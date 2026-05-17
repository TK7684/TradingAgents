#!/usr/bin/env python3
"""Daily TradingAgents runner — analyze, paper trade, post to Discord.

Usage:
    python3 daily_run.py                  # analyze tech watchlist
    python3 daily_run.py --sector etf     # analyze specific sector
    python3 daily_run.py --ticker NVDA    # analyze single ticker
    python3 daily_run.py --all            # scan all sectors
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Add ec2_deploy to path for config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.expanduser("~/TradingAgents")
RESULTS_DIR = os.path.join(BASE_DIR, "results/daily")

from config import get_config, WATCHLIST, ALL_WATCHLISTS
from batch_analyze import analyze_ticker, get_latest_trading_date
from discord_signal import send_trade_decision
from paper_trader import execute_trade, load_portfolio, save_portfolio, get_portfolio_value


def run_daily(tickers, profile="turbo"):
    """Run analysis on tickers, paper trade, and post signals."""
    date = get_latest_trading_date()
    config = get_config(profile)
    portfolio = load_portfolio()
    results = []

    print(f"\n{'='*60}")
    print(f"  TradingAgents Daily Run")
    print(f"  Date: {date} | Profile: {profile} | Tickers: {len(tickers)}")
    print(f"{'='*60}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...")
        start = time.time()

        result = analyze_ticker(ticker, date, config)
        elapsed = time.time() - start
        decision = result.get("decision", "ERROR")
        status = result.get("status", "error")

        icon = "OK" if status == "ok" else "FAIL"
        print(f"  {icon} {ticker}: {decision} ({elapsed:.0f}s)")

        if status == "ok" and decision in ("BUY", "SELL", "HOLD"):
            # Post to Discord
            summary = result.get("summary", "")[:500]
            send_trade_decision(ticker, decision, summary)

            # Paper trade (only BUY/SELL)
            if decision in ("BUY", "SELL"):
                trade = execute_trade(portfolio, ticker, decision, date)
                if trade:
                    if trade["action"] == "BUY":
                        print(f"    -> BUY x{trade['shares']} @ ${trade['price']:.2f}")
                    else:
                        pnl_s = "+" if trade["pnl"] > 0 else ""
                        print(f"    -> SELL P&L: {pnl_s}${trade['pnl']:.0f} ({trade['pnl_pct']}%)")

        results.append(result)

        # Rate limit between tickers
        if i < len(tickers):
            time.sleep(3)

    # Save daily results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, f"{date}_daily_{profile}.json")
    with open(outfile, "w") as f:
        json.dump({"date": date, "profile": profile, "results": results}, f, indent=2)

    # Portfolio summary
    total_value = get_portfolio_value(portfolio)
    pnl = total_value - 100000
    pnl_pct = round(pnl / 100000 * 100, 2)
    sign = "+" if pnl >= 0 else ""

    print(f"\n{'='*60}")
    print(f"  DONE — {len(results)} tickers analyzed")
    print(f"  Portfolio: ${total_value:,.0f} ({sign}{pnl_pct}%)")
    print(f"  Results saved: {outfile}")
    print(f"{'='*60}\n")

    # Summary counts
    buys = sum(1 for r in results if r.get("decision") == "BUY")
    sells = sum(1 for r in results if r.get("decision") == "SELL")
    holds = sum(1 for r in results if r.get("decision") == "HOLD")
    errors = sum(1 for r in results if r.get("status") == "error")
    print(f"  🟢 {buys} BUY | 🔴 {sells} SELL | 🟡 {holds} HOLD | ❌ {errors} errors")


if __name__ == "__main__":
    profile = "turbo"
    tickers = None
    sector = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--profile" and i + 1 < len(sys.argv):
            profile = sys.argv[i + 1]
            i += 2
        elif arg == "--ticker" and i + 1 < len(sys.argv):
            tickers = [sys.argv[i + 1]]
            i += 2
        elif arg == "--sector":
            sector = sys.argv[i + 1] if i + 1 < len(sys.argv) else "tech"
            tickers = ALL_WATCHLISTS.get(sector, WATCHLIST)
            i += 2
        elif arg == "--all":
            # Flatten all watchlists
            tickers = []
            for t_list in ALL_WATCHLISTS.values():
                tickers.extend(t_list)
            i += 1
        else:
            i += 1

    if tickers is None:
        tickers = WATCHLIST

    run_daily(tickers, profile)
