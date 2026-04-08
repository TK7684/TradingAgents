#!/usr/bin/env python3
"""Backtest TradingAgents decisions against actual price movements.

Compares past BUY/SELL/HOLD decisions with what actually happened
to the stock price over 1, 5, and 20 trading days.

Usage:
    python3.12 backtest.py                    # backtest all results
    python3.12 backtest.py --days 5           # 5-day forward returns only
    python3.12 backtest.py --ticker NVDA      # single ticker
"""
import json
import glob
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
import requests

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
RESULTS_DIR = os.path.expanduser("~/TradingAgents/results/daily")
BACKTEST_DIR = os.path.expanduser("~/TradingAgents/results/backtest")


def get_forward_returns(ticker, decision_date, periods=(1, 5, 20)):
    """Get actual returns N days after the decision date."""
    try:
        end = datetime.strptime(decision_date, "%Y-%m-%d") + timedelta(days=max(periods) + 10)
        start = datetime.strptime(decision_date, "%Y-%m-%d") - timedelta(days=1)
        data = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"), progress=False)
        if data.empty:
            return {}

        # Find the decision date price (or next trading day)
        decision_idx = None
        for i, date in enumerate(data.index):
            if date.strftime("%Y-%m-%d") >= decision_date:
                decision_idx = i
                break
        if decision_idx is None:
            return {}

        base_price = float(data.iloc[decision_idx]["Close"].iloc[0]) if hasattr(data.iloc[decision_idx]["Close"], 'iloc') else float(data.iloc[decision_idx]["Close"])
        returns = {}
        for p in periods:
            idx = decision_idx + p
            if idx < len(data):
                future_price = float(data.iloc[idx]["Close"].iloc[0]) if hasattr(data.iloc[idx]["Close"], 'iloc') else float(data.iloc[idx]["Close"])
                returns[p] = round((future_price - base_price) / base_price * 100, 2)
        return returns
    except Exception as e:
        print("  Error fetching {}: {}".format(ticker, e))
        return {}


def score_decision(decision, returns):
    """Score if the decision was correct based on actual returns."""
    if not returns:
        return "unknown"
    # Use 5-day return as primary, fall back to 1-day
    ret = returns.get(5, returns.get(1, 0))
    if decision == "BUY" and ret > 0:
        return "correct"
    elif decision == "SELL" and ret < 0:
        return "correct"
    elif decision == "HOLD" and abs(ret) < 2:
        return "correct"
    elif decision == "BUY" and ret < 0:
        return "wrong"
    elif decision == "SELL" and ret > 0:
        return "wrong"
    else:
        return "neutral"


def run_backtest(ticker_filter=None, period_days=None):
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not files:
        print("No results to backtest")
        return

    periods = (period_days,) if period_days else (1, 5, 20)
    all_scores = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        date = data["date"]
        # Skip if too recent (need forward data)
        days_ago = (datetime.now() - datetime.strptime(date, "%Y-%m-%d")).days
        if days_ago < max(periods) + 5:
            print("Skipping {} (too recent for {}d backtest)".format(date, max(periods)))
            continue

        for r in data["results"]:
            ticker = r["ticker"]
            decision = r.get("decision")
            if not decision or r.get("status") != "ok":
                continue
            if ticker_filter and ticker != ticker_filter:
                continue

            returns = get_forward_returns(ticker, date, periods)
            score = score_decision(decision, returns)
            result = {
                "date": date, "ticker": ticker, "decision": decision,
                "returns": returns, "score": score
            }
            all_scores.append(result)
            ret_str = ", ".join(["{}d: {}%".format(k, v) for k, v in sorted(returns.items())])
            print("  {} {} {} -> {} ({})".format(date, ticker, decision, ret_str, score))

    if not all_scores:
        print("No decisions old enough to backtest")
        return

    # Summary
    correct = sum(1 for s in all_scores if s["score"] == "correct")
    wrong = sum(1 for s in all_scores if s["score"] == "wrong")
    total = correct + wrong
    accuracy = round(correct / total * 100, 1) if total > 0 else 0

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print("Total decisions: {}".format(len(all_scores)))
    print("Correct: {} | Wrong: {} | Accuracy: {}%".format(correct, wrong, accuracy))

    # Per-ticker breakdown
    from collections import defaultdict
    ticker_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "neutral": 0, "unknown": 0})
    for s in all_scores:
        ticker_stats[s["ticker"]][s["score"]] += 1

    print("\n{:<8} {:<8} {:<8} {:<10}".format("Ticker", "Correct", "Wrong", "Accuracy"))
    print("-" * 40)
    for t, stats in sorted(ticker_stats.items()):
        c = stats["correct"]
        w = stats["wrong"]
        acc = round(c / (c + w) * 100, 1) if (c + w) > 0 else 0
        print("{:<8} {:<8} {:<8} {}%".format(t, c, w, acc))

    # Save
    outfile = os.path.join(BACKTEST_DIR, "backtest_{}.json".format(
        datetime.now().strftime("%Y-%m-%d")))
    with open(outfile, "w") as f:
        json.dump({"accuracy": accuracy, "total": len(all_scores),
                    "correct": correct, "wrong": wrong, "scores": all_scores}, f, indent=2)
    print("\nSaved to {}".format(outfile))

    # Discord
    if WEBHOOK_URL and all_scores:
        emoji = "\U0001f3af" if accuracy >= 60 else "\u26a0\ufe0f" if accuracy >= 40 else "\U0001f534"
        color = 0x57F287 if accuracy >= 60 else 0xFEE75C if accuracy >= 40 else 0xED4245
        lines = []
        for t, stats in sorted(ticker_stats.items()):
            c = stats["correct"]
            w = stats["wrong"]
            acc = round(c / (c + w) * 100, 1) if (c + w) > 0 else 0
            lines.append("**{}** — {}% ({}/{})".format(t, acc, c, c + w))
        payload = {"embeds": [{"title": "{} Backtest: {}% accuracy".format(emoji, accuracy),
                   "description": "Total: {} decisions\n\n{}".format(len(all_scores), "\n".join(lines)),
                   "color": color}]}
        requests.post(WEBHOOK_URL, json=payload)

    return accuracy


if __name__ == "__main__":
    ticker = None
    days = None
    for i, a in enumerate(sys.argv):
        if a == "--ticker" and i + 1 < len(sys.argv):
            ticker = sys.argv[i + 1]
        if a == "--days" and i + 1 < len(sys.argv):
            days = int(sys.argv[i + 1])
    run_backtest(ticker_filter=ticker, period_days=days)
