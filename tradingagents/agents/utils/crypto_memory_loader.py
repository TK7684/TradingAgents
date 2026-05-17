"""Load crypto trade history from hyperliquid-dex for TradingAgents memory."""

import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = "/home/tk578/hyperliquid-dex/trading.db"


class CryptoTradeHistoryLoader:
    """Load closed trades from hyperliquid-dex SQLite database."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

    def load_closed_trades(self, limit: int = 200) -> List[Tuple[str, str]]:
        """Load closed trades and convert to (situation, recommendation) tuples.

        Returns list suitable for FinancialSituationMemory.add_situations().

        Args:
            limit: Maximum number of closed trades to load.

        Returns:
            List of (situation, recommendation) tuples for BM25 memory.
        """
        if not self.db_path.exists():
            log.warning(f"Trade database not found: {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM signals
                WHERE status = 'closed'
                ORDER BY close_time DESC
                LIMIT ?
            """,
                (limit,),
            )

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                log.info("No closed trades found in database")
                return []

            results = []
            for row in rows:
                situation = self._build_situation(row)
                recommendation = self._build_recommendation(row)
                results.append((situation, recommendation))

            log.info(f"Loaded {len(results)} closed trades from {self.db_path}")
            return results

        except Exception as e:
            log.error(f"Failed to load trade history: {e}")
            return []

    def _build_situation(self, row) -> str:
        """Build situation text from trade data for BM25 matching.

        Situation captures the setup conditions when the trade was entered.
        """
        entry = row["entry_price"]
        sl = row["sl_price"]
        tp = row["tp_price"]
        conf = row["confidence"]

        parts = [
            f"Symbol: {row['symbol']}",
            f"Strategy: {row['strategy']}",
            f"Direction: {row['direction']}",
            f"Confidence: {conf:.2f}" if conf else "Confidence: unknown",
        ]
        if entry is not None:
            price_str = f"Entry: ${entry:.2f}"
            if sl is not None:
                price_str += f", SL: ${sl:.2f}"
            if tp is not None:
                price_str += f", TP: ${tp:.2f}"
            parts.append(price_str)
        else:
            parts.append("Entry: unknown")

        # Add market context fields if they exist
        for field in ["regime", "btc_trend", "btc_adx", "volatility_level", "session", "tf_alignment"]:
            try:
                val = row[field]
                if val is not None:
                    if isinstance(val, float):
                        parts.append(f"{field}: {val:.2f}")
                    else:
                        parts.append(f"{field}: {val}")
            except (KeyError, IndexError):
                pass

        try:
            reasoning = row["reasoning"]
            if reasoning and len(reasoning) > 200:
                reasoning = reasoning[:200]
            if reasoning:
                parts.append(f"Reasoning: {reasoning}")
        except (KeyError, IndexError):
            pass

        return " | ".join(parts)

    def _build_recommendation(self, row) -> str:
        """Build recommendation/lesson from trade outcome.

        Recommendation captures what happened and what we should learn.
        """
        pnl = row["pnl"] or 0.0
        pnl_pct = row["pnl_pct"] or 0.0
        outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

        parts = []

        # Result summary
        if outcome == "WIN":
            parts.append(f"Result: {outcome} — P&L: +${pnl:.2f} ({pnl_pct:+.1f}%)")
        elif outcome == "LOSS":
            parts.append(f"Result: {outcome} — P&L: -${abs(pnl):.2f} ({pnl_pct:.1f}%)")
        else:
            entry_val = row['entry_price']
            parts.append(f"Result: {outcome} — Entry at {entry_val:.8f}" if entry_val is not None else f"Result: {outcome}")

        # Strategy performance
        strategy = row["strategy"]
        symbol = row["symbol"]
        direction = row["direction"]
        if pnl > 0:
            parts.append(f"{strategy} {direction} on {symbol} worked well.")
        else:
            parts.append(f"{strategy} {direction} on {symbol} did not work.")

        # Context-specific lesson
        try:
            regime = row["regime"]
            if regime:
                if pnl > 0:
                    parts.append(f"In {regime} regime: this strategy was profitable—maintain approach for similar setups.")
                else:
                    parts.append(f"In {regime} regime: consider tighter stops or skip similar setups.")
        except (KeyError, IndexError, TypeError):
            pass

        # Risk/reward context
        try:
            risk = row["risk_dollars"]
            if risk:
                parts.append(f"Risk: ${risk:.0f}")
        except (KeyError, IndexError, TypeError):
            pass

        return " ".join(parts)
