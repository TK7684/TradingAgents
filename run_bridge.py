#!/usr/bin/env python3
"""MT5 Bridge Runner -- starts the webhook receiver + trading pipeline.

Usage:
    # Paper mode (safe, no real trades):
    python run_bridge.py --mode paper

    # Live mode (requires Windows MT5 bridge server running):
    python run_bridge.py --mode live

    # With LLM confirmation before executing:
    python run_bridge.py --mode confirm

    # Custom port and interval:
    python run_bridge.py --port 5001 --interval 3

Environment variables:
    TV_WEBHOOK_SECRET    - HMAC secret for TradingView webhook verification
    DISCORD_WEBHOOK_URL  - Discord webhook for notifications
    DISCORD_TOKEN        - Discord bot token (alternative to webhook)
    DISCORD_TRADE_CHANNEL - Discord channel ID for bot messages
    OPENAI_API_KEY       - Z.AI API key for LLM confirmation
    OPENAI_BASE_URL      - Z.AI base URL
    MT5_BRIDGE_URL       - Windows MT5 bridge URL (default: http://127.0.0.1:5002)
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="TradingAgents MT5 Bridge -- TradingView -> LLM -> MT5"
    )
    parser.add_argument(
        "--mode", choices=["paper", "live", "confirm"],
        default=os.getenv("BRIDGE_MODE", "paper"),
        help="Trading mode: paper (safe), live (real trades), confirm (LLM + live)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Webhook server host")
    parser.add_argument("--port", type=int, default=5001, help="Webhook server port")
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Queue polling interval in seconds",
    )
    parser.add_argument("--lot", type=float, default=0.01, help="Default lot size")
    parser.add_argument(
        "--max-daily", type=int, default=50,
        help="Maximum trades per day",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--mt5-url", default=None,
        help="Windows MT5 bridge URL (default: http://127.0.0.1:5002)",
    )

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("bridge")

    log.info("=" * 60)
    log.info("TradingAgents MT5 Bridge starting")
    log.info("  Mode: %s", args.mode.upper())
    log.info("  Webhook: %s:%d", args.host, args.port)
    log.info("  Default lot: %.2f", args.lot)
    log.info("  Max daily trades: %d", args.max_daily)
    log.info("=" * 60)

    # Shared signal queue
    signal_queue: list[dict] = []

    # Create pipeline
    from tradingagents.mt5_bridge.pipeline import TradingPipeline
    from tradingagents.mt5_bridge.mt5_executor import MT5HttpExecutor

    # Initialize MT5 executor for live modes
    mt5_executor = None
    mt5_url = args.mt5_url or os.getenv("MT5_BRIDGE_URL", "http://127.0.0.1:5002")
    if args.mode in ("live", "confirm"):
        log.info("Connecting to MT5 bridge at %s ...", mt5_url)
        mt5_executor = MT5HttpExecutor(bridge_url=mt5_url)
        if not mt5_executor.connect():
            if args.mode == "live":
                log.error(
                    "MT5 bridge connection failed. "
                    "Make sure mt5_windows_bridge.py is running on Windows. "
                    "Use --mode paper for testing."
                )
                sys.exit(1)
            else:
                log.warning("MT5 bridge not available -- running paper-only")
                args.mode = "paper"

    # LLM config for confirm mode
    llm_config = None
    if args.mode == "confirm":
        llm_config = {
            "model": "glm-4.7",
            "url": os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
        }
        log.info("LLM confirmation enabled: %s", llm_config["model"])

    pipeline = TradingPipeline(
        mode=args.mode,
        signal_queue=signal_queue,
        mt5_executor=mt5_executor,
        default_lot=args.lot,
        max_daily_trades=args.max_daily,
        enable_llm_confirm=(args.mode == "confirm"),
        llm_config=llm_config,
    )

    # Start Flask webhook server in background thread
    from tradingagents.mt5_bridge.webhook_receiver import create_webhook_app

    webhook_secret = os.getenv("TV_WEBHOOK_SECRET", "")
    app = create_webhook_app(queue=signal_queue, secret=webhook_secret)

    flask_thread = threading.Thread(
        target=app.run,
        kwargs={"host": args.host, "port": args.port, "threaded": True},
        daemon=True,
    )
    flask_thread.start()
    log.info("Webhook receiver running in background on port %d", args.port)

    # Run pipeline loop in main thread
    try:
        pipeline.run_loop(interval=args.interval)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        if mt5_executor:
            mt5_executor.shutdown()
        log.info("Bridge stopped")


if __name__ == "__main__":
    main()
