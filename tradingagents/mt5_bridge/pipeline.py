"""Trading Pipeline — orchestrates the full signal-to-execution flow.

Flow:
  1. Consume signal from webhook queue
  2. Validate signal
  3. Optional: Run TradingAgents LLM analysis for confirmation
  4. Execute on MT5 (or paper trade)
  5. Notify via Discord
  6. Log trade to database

Modes:
  - "paper": Log trades without executing (safe testing)
  - "live": Execute on MT5 with real orders
  - "confirm": LLM confirms signal before execution
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .signal_validator import TradingSignal, TradingViewSignalParser
from .mt5_executor import MT5HttpExecutor, OrderResult

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Persistent record of a processed trade."""
    id: str
    signal: dict  # raw signal
    decision: str  # BUY, SELL, CLOSE, REJECTED, ERROR
    mt5_result: dict | None = None
    timestamp: str = ""
    mode: str = "paper"
    llm_confirmed: bool = False
    discord_sent: bool = False
    error: str = ""


class TradingPipeline:
    """End-to-end trading pipeline: signal → validate → analyze → execute → notify.

    Usage:
        pipeline = TradingPipeline(
            mode="paper",
            signal_queue=webhook_queue,
            mt5_executor=executor,
        )
        pipeline.process_next()  # Process one signal
        pipeline.run_loop()      # Continuous processing
    """

    def __init__(
        self,
        mode: str = "paper",
        signal_queue: list[dict] | None = None,
        mt5_executor: MT5HttpExecutor | None = None,
        discord_webhook_url: str = "",
        discord_bot_token: str = "",
        discord_channel_id: str = "",
        data_dir: str = "",
        default_lot: float = 0.01,
        max_daily_trades: int = 50,
        max_position_pct: float = 0.02,  # 2% risk per trade
        enable_llm_confirm: bool = False,
        llm_config: dict | None = None,
    ):
        """Initialize the trading pipeline.

        Args:
            mode: "paper", "live", or "confirm"
            signal_queue: Shared signal queue from webhook receiver
            mt5_executor: MT5Executor instance (required for live mode)
            discord_webhook_url: Discord webhook URL for notifications
            discord_bot_token: Discord bot token (alternative to webhook)
            discord_channel_id: Discord channel ID for bot messages
            data_dir: Directory for trade logs (defaults to results/trades)
            default_lot: Default lot size if not specified in signal
            max_daily_trades: Maximum trades per day (risk control)
            max_position_pct: Max % of account risked per trade
            enable_llm_confirm: Use LLM to confirm signals before execution
            llm_config: Config dict for LLM analysis (provider, model, url)
        """
        self.mode = mode
        self._queue = signal_queue if signal_queue is not None else []
        self._mt5 = mt5_executor
        self._parser = TradingViewSignalParser()
        self._default_lot = default_lot
        self._max_daily_trades = max_daily_trades
        self._max_position_pct = max_position_pct
        self._enable_llm_confirm = enable_llm_confirm
        self._llm_config = llm_config or {}

        # Discord config
        self._webhook_url = discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
        self._bot_token = discord_bot_token or os.getenv("DISCORD_TOKEN", "")
        self._channel_id = discord_channel_id or os.getenv("DISCORD_TRADE_CHANNEL", "")

        # Trade log
        self._data_dir = Path(data_dir) if data_dir else Path(os.path.expanduser(
            "~/TradingAgents/results/trades"
        ))
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._trades_file = self._data_dir / "bridge_trades.jsonl"
        self._daily_count = 0
        self._daily_date = ""

        # Stats
        self._stats = {
            "total_processed": 0,
            "total_executed": 0,
            "total_rejected": 0,
            "total_errors": 0,
        }

    @property
    def stats(self) -> dict:
        return {**self._stats, "queue_size": len(self._queue)}

    def process_next(self) -> TradeRecord | None:
        """Process the next signal from the queue.

        Returns the TradeRecord, or None if queue is empty.
        """
        if not self._queue:
            return None

        # Dequeue
        signal_data = self._queue.pop(0)
        log.info("Processing signal: %s", json.dumps(signal_data)[:200])

        # Parse signal
        signal = self._parser.parse_json(signal_data)
        if signal is None:
            # Try text parsing — look for text/message/alert fields, or stringify
            text = (
                signal_data.get("text", "")
                or signal_data.get("message", "")
                or signal_data.get("alert", "")
                or signal_data.get("description", "")
            )
            if not text:
                # Try to stringify the whole payload if it looks like plain text
                if len(signal_data) == 1:
                    first_val = next(iter(signal_data.values()))
                    if isinstance(first_val, str) and len(first_val) > 3:
                        text = first_val
            if text:
                signal = self._parser.parse_text(text)

        self._stats["total_processed"] += 1

        if signal is None:
            record = self._log_trade(signal_data, "REJECTED", error="could not parse signal")
            self._stats["total_rejected"] += 1
            return record

        # Validate
        is_valid, reason = self._parser.validate(signal)
        if not is_valid:
            record = self._log_trade(signal_data, "REJECTED", error=reason)
            self._stats["total_rejected"] += 1
            log.warning("Signal rejected: %s — %s", signal.symbol, reason)
            return record

        # Daily trade limit
        self._check_daily_reset()
        if self._daily_count >= self._max_daily_trades:
            record = self._log_trade(
                signal_data, "REJECTED",
                error=f"daily trade limit reached ({self._max_daily_trades})",
            )
            self._stats["total_rejected"] += 1
            return record

        # Optional: LLM confirmation
        if self._enable_llm_confirm and self.mode != "paper":
            confirmed = self._llm_confirm_signal(signal)
            if not confirmed:
                record = self._log_trade(
                    signal_data, "REJECTED",
                    error="LLM rejected signal", llm_confirmed=False,
                )
                self._stats["total_rejected"] += 1
                return record

        # Execute
        decision = signal.side
        mt5_result = None

        if self.mode == "paper":
            log.info(
                "PAPER TRADE: %s %s @ %s [strategy=%s]",
                decision, signal.symbol, signal.price, signal.strategy,
            )
            mt5_result = {"success": True, "mode": "paper", "price": signal.price}
            self._daily_count += 1
        elif self.mode == "live" and self._mt5 and self._mt5.is_connected:
            mt5_result = self._execute_on_mt5(signal)
            if mt5_result["success"]:
                self._daily_count += 1
            else:
                decision = "ERROR"
        elif self.mode == "live":
            log.error("Live mode but MT5 not connected — falling back to paper")
            mt5_result = {"success": False, "mode": "paper", "error": "MT5 not connected"}
            decision = "ERROR"
        else:
            mt5_result = {"success": False, "mode": "unknown", "error": "invalid mode"}

        if decision not in ("REJECTED", "ERROR"):
            self._stats["total_executed"] += 1
        else:
            self._stats["total_errors"] += 1

        # Notify Discord
        discord_ok = self._send_discord_notification(signal, decision, mt5_result)

        record = self._log_trade(
            signal_data, decision,
            mt5_result=mt5_result,
            llm_confirmed=self._enable_llm_confirm,
            discord_sent=discord_ok,
            mode=self.mode,
        )

        return record

    def run_loop(self, interval: float = 5.0, max_iterations: int | None = None):
        """Run continuous signal processing loop.

        Args:
            interval: Seconds between queue checks.
            max_iterations: Stop after N iterations (None = infinite).
        """
        log.info(
            "Pipeline started: mode=%s, queue=%d, interval=%.1fs",
            self.mode, len(self._queue), interval,
        )
        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                record = self.process_next()
                if record:
                    log.info("Trade processed: %s", record.id)
                time.sleep(interval)
                iteration += 1
        except KeyboardInterrupt:
            log.info("Pipeline stopped by user")
        finally:
            if self._mt5:
                self._mt5.shutdown()

    def _execute_on_mt5(self, signal: TradingSignal) -> dict:
        """Execute a validated signal on MT5."""
        if not self._mt5 or not self._mt5.is_connected:
            return {"success": False, "error": "MT5 not connected"}

        lot = signal.lot or self._default_lot

        # Check for existing position (avoid duplicates)
        existing = self._mt5.get_positions(signal.symbol)
        if existing and signal.side != "CLOSE":
            log.warning(
                "Position already exists for %s (%d positions), skipping",
                signal.symbol, len(existing),
            )
            return {"success": False, "error": "position already exists"}

        try:
            if signal.side == "BUY":
                result = self._mt5.market_buy(
                    symbol=signal.symbol,
                    lot=lot,
                    sl=signal.sl,
                    tp=signal.tp,
                    comment=f"TV:{signal.strategy}",
                )
            elif signal.side == "SELL":
                result = self._mt5.market_sell(
                    symbol=signal.symbol,
                    lot=lot,
                    sl=signal.sl,
                    tp=signal.tp,
                    comment=f"TV:{signal.strategy}",
                )
            elif signal.side == "CLOSE":
                if existing:
                    results = []
                    for pos in existing:
                        r = self._mt5.close_position(pos.ticket)
                        results.append({"ticket": pos.ticket, **asdict(r)})
                    # Return first result's status
                    all_ok = all(r["success"] for r in results)
                    return {
                        "success": all_ok,
                        "action": "close",
                        "positions_closed": len(results),
                        "results": results,
                    }
                return {"success": True, "action": "close", "positions_closed": 0}
            else:
                return {"success": False, "error": f"unknown side: {signal.side}"}

            return asdict(result)

        except Exception as e:
            log.exception("MT5 execution error: %s", e)
            return {"success": False, "error": str(e)}

    def _llm_confirm_signal(self, signal: TradingSignal) -> bool:
        """Use LLM to confirm or reject a signal (optional risk layer)."""
        if not self._llm_config:
            return True  # No LLM configured — auto-approve

        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=self._llm_config.get("model", "glm-4.7"),
                base_url=self._llm_config.get("url", ""),
                api_key=self._llm_config.get("api_key", ""),
                temperature=0,
            )

            prompt = (
                f"You are a risk manager. Confirm or reject this trading signal.\n"
                f"Symbol: {signal.symbol}\n"
                f"Side: {signal.side}\n"
                f"Price: {signal.price}\n"
                f"Strategy: {signal.strategy}\n"
                f"SL: {signal.sl}, TP: {signal.tp}\n"
                f"Lot: {signal.lot}\n"
                f"Timeframe: {signal.timeframe}\n\n"
                f"Respond with ONLY 'CONFIRM' or 'REJECT' followed by a one-line reason."
            )

            response = llm.invoke([{"role": "user", "content": prompt}])
            decision = response.content.strip().upper()

            if "CONFIRM" in decision:
                log.info("LLM confirmed signal: %s %s", signal.side, signal.symbol)
                return True
            else:
                log.warning("LLM rejected signal: %s", response.content[:100])
                return False

        except Exception as e:
            log.error("LLM confirmation failed: %s — auto-rejecting", e)
            return False

    def _send_discord_notification(self, signal: TradingSignal,
                                   decision: str,
                                   mt5_result: dict | None) -> bool:
        """Send trade notification to Discord."""
        # Color coding
        color_map = {
            "BUY": 0x57F287, "SELL": 0xED4245, "CLOSE": 0xFEE75C,
            "HOLD": 0x5865F2, "ERROR": 0xED4245, "REJECTED": 0xFF0000,
        }
        color = color_map.get(decision, 0x5865F2)

        emoji_map = {
            "BUY": "🟢", "SELL": "🔴", "CLOSE": "🟡",
            "HOLD": "🟣", "ERROR": "⚠️", "REJECTED": "🚫",
        }
        emoji = emoji_map.get(decision, "⚪")

        mode_tag = f"[{self.mode.upper()}] " if self.mode != "live" else ""

        title = f"{mode_tag}{decision}: {signal.symbol}"

        price_str = f"@${signal.price:.5f}" if signal.price else "@market"
        desc_lines = [
            f"**Strategy:** {signal.strategy}",
            f"**Price:** {price_str}",
        ]
        if signal.sl:
            desc_lines.append(f"**SL:** {signal.sl}")
        if signal.tp:
            desc_lines.append(f"**TP:** {signal.tp}")
        if signal.lot:
            desc_lines.append(f"**Lot:** {signal.lot}")
        if signal.timeframe:
            desc_lines.append(f"**Timeframe:** {signal.timeframe}")

        if mt5_result:
            if mt5_result.get("error"):
                desc_lines.append(f"⚠️ {mt5_result['error']}")
            if mt5_result.get("ticket"):
                desc_lines.append(f"🎫 Ticket: {mt5_result['ticket']}")

        description = "\n".join(desc_lines)

        # Try webhook first
        if self._webhook_url and "placeholder" not in self._webhook_url:
            if self._send_webhook(title, description, color):
                return True

        # Try bot token
        if self._bot_token and self._channel_id:
            if self._send_bot_message(title, description, color):
                return True

        # Fallback: write to notification queue file (Hermes can pick this up)
        self._write_notification_file(f"{emoji} {title}", description, color)
        log.info("Discord send failed — wrote notification to file")
        return False

    def _send_webhook(self, title: str, description: str, color: int) -> bool:
        """Send via Discord webhook."""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "embeds": [{
                "title": title,
                "description": description[:4096],
                "color": color,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "footer": {"text": "TradingAgents MT5 Bridge"},
            }]
        }).encode("utf-8")

        req = urllib.request.Request(
            self._webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            log.error("Discord webhook failed: %s", e)
            return False

    def _send_bot_message(self, title: str, description: str, color: int) -> bool:
        """Send via Discord bot API."""
        import urllib.request

        url = f"https://discord.com/api/v10/channels/{self._channel_id}/messages"
        payload = json.dumps({
            "embeds": [{
                "title": title,
                "description": description[:4096],
                "color": color,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }]
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bot {self._bot_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            log.error("Discord bot message failed: %s", e)
            return False

    def _write_notification_file(self, title: str, description: str, color: int):
        """Write notification to a JSONL file for Hermes to pick up."""
        notif_dir = self._data_dir / "notifications"
        notif_dir.mkdir(parents=True, exist_ok=True)
        notif_file = notif_dir / "pending.jsonl"

        entry = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "target": "discord:#trading-signals",
        }
        try:
            with open(notif_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            log.error("Failed to write notification file: %s", e)

    def _log_trade(self, signal_data: dict, decision: str, **kwargs) -> TradeRecord:
        """Append trade record to JSONL log file."""
        import hashlib
        record_id = hashlib.md5(
            f"{signal_data}{time.time()}".encode()
        ).hexdigest()[:12]

        record = TradeRecord(
            id=record_id,
            signal=signal_data,
            decision=decision,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            **kwargs,
        )

        try:
            with open(self._trades_file, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception as e:
            log.error("Failed to log trade: %s", e)

        return record

    def _check_daily_reset(self):
        """Reset daily trade counter at midnight UTC."""
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_count = 0
            self._daily_date = today
            log.info("Daily trade counter reset")
