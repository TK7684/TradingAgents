#!/usr/bin/env python3
"""Retroactively score past predictions against actual price movements.

Called daily after market close to grade yesterday's predictions.
Updates the signal_accuracy.db so ConsensusEngine weights improve over time.

Usage:
    python3 score_predictions.py                   # score yesterday's predictions
    python3 score_predictions.py --date 2026-05-28  # score specific date
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

# Add project root
sys.path.insert(0, os.path.expanduser("~/TradingAgents"))

import yfinance as yf


def get_actual_signal(ticker: str, date_str: str) -> str:
    """Determine the 'correct' signal based on next-day price movement.

    Logic:
    - If price went up >1% → BUY was correct
    - If price went down >1% → SELL was correct
    - If price moved <1% → HOLD was correct
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Next trading day (skip weekends)
    next_dt = dt + timedelta(days=1)
    while next_dt.weekday() >= 5:
        next_dt += timedelta(days=1)
    next_str = next_dt.strftime("%Y-%m-%d")

    try:
        hist = yf.Ticker(ticker).history(start=date_str, end=next_dt, auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return "HOLD"  # Can't determine
        close_today = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else float(hist["Close"].iloc[-1])
        close_tomorrow = float(hist["Close"].iloc[-1])

        if close_today == 0:
            return "HOLD"

        pct_change = (close_tomorrow - close_today) / close_today

        if pct_change > 0.01:
            return "BUY"
        elif pct_change < -0.01:
            return "SELL"
        else:
            return "HOLD"
    except Exception as e:
        print(f"  Error getting price for {ticker}: {e}")
        return "HOLD"


def main():
    parser = argparse.ArgumentParser(description="Score predictions against actuals")
    parser.add_argument("--date", default=None, help="Date to score (YYYY-MM-DD)")
    args = parser.parse_args()

    from tradingagents.graph.consensus import ConsensusEngine

    engine = ConsensusEngine()

    if args.date:
        date_str = args.date
    else:
        # Default: yesterday (or last weekday)
        dt = datetime.now() - timedelta(days=1)
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)
        date_str = dt.strftime("%Y-%m-%d")

    # Get ungraded predictions for this date
    import sqlite3
    db_path = os.path.expanduser("~/TradingAgents/results/signal_accuracy.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT DISTINCT ticker FROM predictions WHERE date = ? AND actual_signal IS NULL",
        (date_str,),
    ).fetchall()

    if not rows:
        print(f"No ungraded predictions for {date_str}")
        conn.close()
        engine.close()
        return

    print(f"Scoring {len(rows)} ticker predictions for {date_str}...")
    for row in rows:
        ticker = row["ticker"]
        actual = get_actual_signal(ticker, date_str)
        updated = engine.record_outcome(ticker, date_str, actual)
        print(f"  {ticker}: actual={actual} ({updated} predictions graded)")

    # Show updated stats
    stats = engine.get_source_stats()
    print(f"\nUpdated accuracy stats:")
    for source, s in stats.items():
        acc = s["accuracy"]
        total = s["total"]
        correct = s["correct"]
        print(f"  {source}: {acc:.1%} ({correct}/{total})")

    conn.close()
    engine.close()


if __name__ == "__main__":
    main()
