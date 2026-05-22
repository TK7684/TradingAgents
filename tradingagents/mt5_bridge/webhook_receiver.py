"""TradingView Webhook Receiver — Flask server that receives alert JSON from TradingView.

TradingView sends alerts as HTTP POST with JSON body like:
    {"ticker": "EURUSD", "action": "buy", "price": 1.0850, "timeframe": "1H",
     "strategy": "RSI + MACD Crossover", "indicator_values": {...}}

This server validates, normalizes, and queues signals for the trading pipeline.

Usage:
    python -m tradingagents.mt5_bridge.webhook_receiver
    # Starts Flask on port 5001
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from flask import Flask, request, jsonify

log = logging.getLogger(__name__)

# In-memory signal queue (thread-safe)
_signal_queue: list[dict[str, Any]] = []
_queue_lock = Lock()
MAX_QUEUE_SIZE = 500


def create_webhook_app(queue: list[dict[str, Any]] | None = None,
                       secret: str = "") -> Flask:
    """Create and configure the Flask webhook receiver app.

    Args:
        queue: Shared signal queue (defaults to module-level _signal_queue).
        secret: Optional webhook secret for HMAC verification.
    """
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    if queue is not None:
        _q = queue
    else:
        _q = _signal_queue

    _lock = Lock()

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "queue_size": len(_q),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })

    @app.route("/webhook/tradingview", methods=["POST"])
    def tradingview_webhook():
        """Receive TradingView alert webhook.

        TradingView alert JSON format (configured in alert settings):
        {
            "ticker": "{{ticker}}",
            "exchange": "{{exchange}}",
            "action": "{{strategy.order.action}}",
            "price": "{{close}}",
            "time": "{{timenow}}",
            "interval": "{{interval}}",
            "strategy": "My Strategy Name",
            "volume": "{{volume}}"
        }

        Or custom Pine Script JSON via webhook URL:
        {
            "symbol": "EURUSD",
            "side": "buy",
            "price": 1.0850,
            "sl": 1.0830,
            "tp": 1.0900,
            "lot": 0.1,
            "strategy": "RSI_Divergence",
            "timeframe": "1H"
        }
        """
        try:
            data = request.get_json(force=True, silent=True)
            if not data:
                log.warning("Received empty webhook payload")
                return jsonify({"error": "empty payload"}), 400

            # Optional HMAC verification
            if secret:
                sig = request.headers.get("X-Webhook-Signature", "")
                import hashlib
                import hmac
                expected = hmac.new(
                    secret.encode(), request.get_data(), hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(sig, expected):
                    log.warning("Invalid webhook signature")
                    return jsonify({"error": "invalid signature"}), 403

            # Normalize the signal
            signal = _normalize_signal(data)

            with _lock:
                if len(_q) >= MAX_QUEUE_SIZE:
                    _q.pop(0)  # drop oldest
                signal["received_at"] = datetime.now(tz=timezone.utc).isoformat()
                _q.append(signal)

            log.info(
                "Signal queued: %s %s @ %s [strategy=%s, queue=%d]",
                signal["side"].upper(),
                signal["symbol"],
                signal.get("price", "market"),
                signal.get("strategy", "unknown"),
                len(_q),
            )

            return jsonify({"status": "queued", "queue_size": len(_q)}), 200

        except Exception as e:
            log.exception("Webhook processing error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/signals", methods=["GET"])
    def list_signals():
        """List pending signals in the queue."""
        with _lock:
            return jsonify({"signals": _q, "count": len(_q)})

    @app.route("/signals/clear", methods=["POST"])
    def clear_signals():
        """Clear the signal queue."""
        with _lock:
            count = len(_q)
            _q.clear()
        return jsonify({"status": "cleared", "removed": count})

    @app.route("/signals/next", methods=["POST"])
    def pop_signal():
        """Pop the next signal from the queue (for the pipeline consumer)."""
        with _lock:
            if _q:
                signal = _q.pop(0)
                return jsonify({"status": "dequeued", "signal": signal})
            return jsonify({"status": "empty"}), 204

    return app


def _normalize_signal(raw: dict) -> dict[str, Any]:
    """Normalize TradingView webhook payload into standard signal format.

    Handles both TradingView template format and custom Pine Script JSON.
    """
    symbol = (
        raw.get("symbol") or raw.get("ticker") or raw.get("pair") or "UNKNOWN"
    ).upper()

    # Normalize TradingView ticker format (e.g., "FX:EURUSD" → "EURUSD")
    for prefix in ("FX:", "NASDAQ:", "NYSE:", "BINANCE:", "COINBASE:", "FOREX:"):
        if symbol.startswith(prefix):
            symbol = symbol[len(prefix):]

    # Normalize side
    raw_action = raw.get("action", raw.get("side", raw.get("order", ""))).lower()
    if raw_action in ("buy", "long", "bull", "autocall"):
        side = "BUY"
    elif raw_action in ("sell", "short", "bear"):
        side = "SELL"
    elif raw_action in ("close", "exit", "flat"):
        side = "CLOSE"
    else:
        side = "UNKNOWN"

    # Parse price
    price = None
    for key in ("price", "close", "value"):
        if key in raw:
            try:
                price = float(raw[key])
                break
            except (ValueError, TypeError):
                continue

    # Parse SL/TP
    sl = None
    tp = None
    for key in ("sl", "stop_loss", "stoploss"):
        if key in raw:
            try:
                sl = float(raw[key])
                break
            except (ValueError, TypeError):
                continue
    for key in ("tp", "take_profit", "takeprofit"):
        if key in raw:
            try:
                tp = float(raw[key])
                break
            except (ValueError, TypeError):
                continue

    # Parse lot size
    lot = None
    for key in ("lot", "lot_size", "qty", "quantity", "volume"):
        if key in raw:
            try:
                lot = float(raw[key])
                break
            except (ValueError, TypeError):
                continue

    return {
        "symbol": symbol,
        "side": side,
        "price": price,
        "sl": sl,
        "tp": tp,
        "lot": lot,
        "strategy": raw.get("strategy", raw.get("indicator", "unknown")),
        "timeframe": raw.get("interval", raw.get("timeframe", "")),
        "exchange": raw.get("exchange", ""),
        "raw": raw,  # preserve original for debugging
    }


def run_server(host: str = "0.0.0.0", port: int = 5001,
               secret: str = "", debug: bool = False):
    """Run the webhook receiver server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    webhook_secret = secret or os.getenv("TV_WEBHOOK_SECRET", "")
    app = create_webhook_app(secret=webhook_secret)
    log.info("TradingView webhook receiver starting on %s:%d", host, port)
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_server()
