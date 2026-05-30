#!/usr/bin/env python3
"""Batch analysis utilities for TradingAgents.

Provides functions to analyze multiple tickers and manage results.
Used by scan_all_sectors.py and discord_bot.py.
"""
import os
import sys
import json
import time
import logging
import argparse
import glob
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.expanduser("~"), "TradingAgents", ".env"))

BASE_DIR = os.path.expanduser("~/TradingAgents")
RESULTS_DIR = os.path.join(BASE_DIR, "results/daily")

log = logging.getLogger(__name__)


def get_latest_trading_date():
    """Get the most recent trading date (skip weekends)."""
    today = datetime.now()
    # Go back to last weekday
    while today.weekday() >= 5:  # Saturday=5, Sunday=6
        today -= timedelta(days=1)
    return today.strftime("%Y-%m-%d")


def analyze_ticker(ticker, date, config=None, timeout_seconds=600):
    """Analyze a single ticker using TradingAgents.
    
    Returns a dict with: ticker, date, decision, status, summary
    
    Args:
        ticker: Stock ticker symbol
        date: Analysis date (YYYY-MM-DD)
        config: TradingAgents config dict
        timeout_seconds: Hard timeout per ticker (default 600s = 10min).
            Prevents hung LLM API calls from blocking the entire daily run.
    """
    try:
        sys.path.insert(0, BASE_DIR)
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG
        
        if config is None:
            # Import config from ec2_deploy
            ec2_config_dir = os.path.dirname(os.path.abspath(__file__))
            if ec2_config_dir not in sys.path:
                sys.path.insert(0, ec2_config_dir)
            try:
                from config import get_config
                config = get_config("turbo")
            except ImportError:
                config = DEFAULT_CONFIG.copy()
        
        ta = TradingAgentsGraph(debug=False, config=config)
        result_state, decision, consensus_data = ta.propagate(ticker, date)
        
        return {
            "ticker": ticker,
            "date": date,
            "decision": decision.strip().upper() if decision else "UNKNOWN",
            "status": "ok",
            "summary": decision[:200] if decision else "",
            "consensus": consensus_data,
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "date": date,
            "decision": None,
            "status": "error",
            "error": str(e),
        }


def analyze_ticker_with_retry(ticker, date, config=None, timeout_seconds=600,
                               retry_enabled=True, max_retries=1, retry_delay=30):
    """Analyze a ticker with retry logic for transient failures (e.g. LLM timeouts).
    
    On failure, waits retry_delay seconds before each retry attempt.
    Skips the ticker entirely after max_retries failed attempts.
    
    Args:
        ticker, date, config, timeout_seconds: passed to analyze_ticker()
        retry_enabled: if False, no retries are attempted
        max_retries: number of retries after the initial attempt (default 1)
        retry_delay: seconds to wait between attempts (default 30)
    """
    attempts = 1 + (max_retries if retry_enabled else 0)
    result = None
    
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            log.warning("RETRY %d/%d for %s (waited %ds) — previous error: %s",
                        attempt - 1, max_retries, ticker, retry_delay,
                        (result or {}).get("error", "unknown"))
            print(f"[RETRY] Attempt {attempt}/{attempts} for {ticker} after {retry_delay}s wait...")
            time.sleep(retry_delay)
        
        result = analyze_ticker(ticker, date, config, timeout_seconds)
        
        if result.get("status") == "ok":
            if attempt > 1:
                log.info("RETRY SUCCEEDED for %s on attempt %d", ticker, attempt)
                print(f"[RETRY] Success for {ticker} on attempt {attempt}")
            return result
        
        if result.get("status") == "error":
            if attempt == 1 and not retry_enabled:
                log.error("Analysis failed for %s (retries disabled): %s",
                          ticker, result.get("error"))
            elif attempt == 1:
                log.warning("Analysis failed for %s on attempt %d/%d: %s — will retry",
                            ticker, attempt, attempts, result.get("error"))
            else:
                log.error("Analysis failed for %s on attempt %d/%d: %s — skipping ticker",
                          ticker, attempt, attempts, result.get("error"))
                print(f"[RETRY] All {attempts} attempts failed for {ticker} — skipping")
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Analyze a single ticker using TradingAgents.")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("date", nargs="?", default=None,
                        help="Analysis date YYYY-MM-DD (default: latest trading date)")
    parser.add_argument("--profile", default="turbo", choices=["turbo", "default", "deep"],
                        help="Config profile (default: turbo)")
    parser.add_argument("--retry", action="store_true", default=True,
                        help="Enable retry on failure (default: True)")
    parser.add_argument("--no-retry", dest="retry", action="store_false",
                        help="Disable retry on failure")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Max retry attempts after initial failure (default: 1)")
    parser.add_argument("--retry-delay", type=int, default=30,
                        help="Seconds to wait between retries (default: 30)")
    args = parser.parse_args()
    
    ticker = args.ticker
    date = args.date or get_latest_trading_date()
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import get_config
    config = get_config(args.profile)
    
    result = analyze_ticker_with_retry(ticker, date, config,
                                        retry_enabled=args.retry,
                                        max_retries=args.max_retries,
                                        retry_delay=args.retry_delay)
    print(json.dumps(result, indent=2))
