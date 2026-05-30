#!/usr/bin/env python3
"""Batch analysis utilities for TradingAgents.

Provides functions to analyze multiple tickers and manage results.
Used by scan_all_sectors.py and discord_bot.py.
"""
import os
import sys
import json
import glob
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.expanduser("~"), "TradingAgents", ".env"))

BASE_DIR = os.path.expanduser("~/TradingAgents")
RESULTS_DIR = os.path.join(BASE_DIR, "results/daily")


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: batch_analyze.py <TICKER> [DATE] [--profile turbo|default|deep]")
        sys.exit(1)
    
    ticker = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else get_latest_trading_date()
    
    profile = "turbo"
    for i, a in enumerate(sys.argv):
        if a == "--profile" and i + 1 < len(sys.argv):
            profile = sys.argv[i + 1]
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import get_config
    config = get_config(profile)
    
    result = analyze_ticker(ticker, date, config)
    print(json.dumps(result, indent=2))
