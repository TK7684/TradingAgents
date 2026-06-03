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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Add ec2_deploy to path for config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.expanduser("~/TradingAgents")
RESULTS_DIR = os.path.join(BASE_DIR, "results/daily")

from config import get_config, WATCHLIST, ALL_WATCHLISTS
from batch_analyze import analyze_ticker, get_latest_trading_date
from discord_signal import send_trade_decision
from paper_trader import execute_trade, load_portfolio, save_portfolio, get_portfolio_value, check_risk_rules

# Pipeline tuning
INTER_TICKER_DELAY = 15  # Seconds between ticker analyses (was 3, too aggressive for Z.AI)
TICKER_TIMEOUT = 600      # Hard timeout per ticker (10 min)

# Confidence thresholds for consensus voting
MIN_CONFIDENCE_TO_TRADE = 0.5  # Don't paper-trade below this confidence


def run_daily(tickers, profile="turbo"):
    """Run analysis on tickers, paper trade, and post signals."""
    date = get_latest_trading_date()
    config = get_config(profile)
    portfolio = load_portfolio()
    results = []

    print(f"\n{'='*60}")
    print(f"  TradingAgents Daily Run")
    print(f"  Date: {date} | Profile: {profile} | Tickers: {len(tickers)}")
    print(f"  Inter-ticker delay: {INTER_TICKER_DELAY}s | Timeout: {TICKER_TIMEOUT}s")
    print(f"{'='*60}\n")

    # Step 0: Fetch TradingView pre-market briefing ONCE for all tickers
    tv_briefing = None
    print("Fetching TradingView market data...")
    try:
        from tradingagents.tv_screener.client import TVScreener
        from tradingagents.tv_screener.briefing import format_discord_briefing
        tv = TVScreener()
        # Build sectors dict from the tickers being analyzed
        sectors = {"watchlist": tickers}
        tv_briefing = tv.get_pre_market_watchlist(sectors=sectors)
        # Print briefing summary to console
        print(format_discord_briefing(tv_briefing))
        print(f"  ✅ TradingView briefing loaded ({tv.has_realtime and 'real-time' or 'delayed'})\n")
    except Exception as e:
        print(f"  ⚠️  TradingView briefing failed (continuing without): {e}\n")

    # Step 0: Check risk rules on existing positions (stop-loss, take-profit, trailing stop)
    print("Checking risk management rules (stop-loss, take-profit, trailing stop)...")
    risk_trades = check_risk_rules(portfolio, date)
    if risk_trades:
        print(f"  {len(risk_trades)} risk rule(s) triggered")
    else:
        print("  No risk rules triggered")

    # Step 1: Analyze each ticker sequentially with rate-limit-safe delays
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}...")
        start = time.time()

        # Run analysis with a hard timeout to prevent hung LLM calls from blocking
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(analyze_ticker, ticker, date, config, tv_briefing=tv_briefing)
                result = future.result(timeout=TICKER_TIMEOUT)
        except FuturesTimeoutError:
            elapsed = time.time() - start
            print(f"  TIMEOUT {ticker}: None ({elapsed:.0f}s) — LLM call exceeded {TICKER_TIMEOUT}s limit")
            result = {
                "ticker": ticker,
                "date": date,
                "decision": None,
                "status": "error",
                "error": f"Ticker analysis timed out after {TICKER_TIMEOUT} seconds",
            }
            results.append(result)
            # Still delay after timeout to avoid burst on retry
            if i < len(tickers):
                print(f"  ...waiting {INTER_TICKER_DELAY}s before next ticker")
                time.sleep(INTER_TICKER_DELAY)
            continue
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR {ticker}: {str(e)[:100]} ({elapsed:.0f}s)")
            result = {
                "ticker": ticker,
                "date": date,
                "decision": None,
                "status": "error",
                "error": str(e),
            }
            results.append(result)
            if i < len(tickers):
                time.sleep(INTER_TICKER_DELAY)
            continue

        elapsed = time.time() - start
        decision = result.get("decision", "ERROR")
        status = result.get("status", "error")

        icon = "OK" if status == "ok" else "FAIL"
        confidence = result.get("consensus", {}).get("confidence", "N/A")
        recommendation = result.get("consensus", {}).get("recommendation", "")
        print(f"  {icon} {ticker}: {decision} (conf={confidence}) [{recommendation}] ({elapsed:.0f}s)")

        if status == "ok" and decision in ("BUY", "SELL", "HOLD", "OVERWEIGHT", "UNDERWEIGHT"):
            # Normalize signals
            trade_decision = decision
            if decision == "OVERWEIGHT":
                trade_decision = "BUY"
            elif decision == "UNDERWEIGHT":
                trade_decision = "SELL"

            # Post to Discord (always post, even if low confidence — for tracking)
            summary = result.get("summary", "")[:500]
            consensus_info = result.get("consensus")
            if consensus_info:
                summary += f"\n📊 Consensus: {consensus_info['recommendation']} (confidence={consensus_info['confidence']:.2f})"
                if consensus_info.get("source_signals"):
                    sources_str = " | ".join(
                        f"{k}={v}" for k, v in consensus_info["source_signals"].items()
                    )
                    summary += f"\nSignals: {sources_str}"
            send_trade_decision(ticker, decision, summary)

            # Paper trade only when confidence meets threshold
            consensus_confidence = (result.get("consensus") or {}).get("confidence", 1.0)
            if trade_decision in ("BUY", "SELL"):
                if consensus_confidence >= MIN_CONFIDENCE_TO_TRADE:
                    trade = execute_trade(portfolio, ticker, trade_decision, date)
                    if trade:
                        if trade["action"] == "BUY":
                            print(f"    -> BUY x{trade['shares']} @ ${trade['price']:.2f}")
                        else:
                            pnl_s = "+" if trade["pnl"] > 0 else ""
                            print(f"    -> SELL P&L: {pnl_s}${trade['pnl']:.0f} ({trade['pnl_pct']}%)")
                else:
                    print(f"    -> SKIP paper trade (confidence {consensus_confidence:.2f} < {MIN_CONFIDENCE_TO_TRADE})")

        results.append(result)

        # Rate limit between tickers — generous delay to stay under Z.AI quota
        if i < len(tickers):
            print(f"  ...waiting {INTER_TICKER_DELAY}s before next ticker (rate limit)")
            time.sleep(INTER_TICKER_DELAY)

    # Save daily results (include consensus data for backtest)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, f"{date}_daily_{profile}.json")
    # Strip consensus to serializable keys only
    for r in results:
        if "consensus" in r and isinstance(r["consensus"], dict):
            r["consensus"] = {
                "final_signal": r["consensus"].get("final_signal", ""),
                "confidence": r["consensus"].get("confidence", 0),
                "recommendation": r["consensus"].get("recommendation", ""),
                "unanimous": r["consensus"].get("unanimous", False),
                "source_signals": r["consensus"].get("source_signals", {}),
            }
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
