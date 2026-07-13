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


def get_actual_signal(ticker: str, date_str: str, horizon_days: int = 5) -> tuple[str, dict]:
    """Determine the 'correct' signal based on N-day price movement.

    Uses 5-day horizon by default (better accuracy than 1-day: 38.9% vs 27.2%).
    
    Logic:
    - If price went up >2% over N days → BUY was correct
    - If price went down >2% over N days → SELL was correct
    - If price moved <2% → HOLD was correct
    
    Returns:
        (signal, details_dict) where details has pct_change, horizon, prices, etc.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Fetch enough data: signal date - 2 days to target + 10 buffer
    start = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=horizon_days + 10)).strftime("%Y-%m-%d")

    details = {"ticker": ticker, "date": date_str, "horizon": horizon_days,
               "close_0": None, "close_N": None, "pct_change": None}

    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return "HOLD", details

        # Find close on signal date
        close_0 = None
        for idx, row in hist.iterrows():
            if idx.strftime("%Y-%m-%d") == date_str:
                close_0 = float(row["Close"])
                break
        if close_0 is None:
            return "HOLD", details

        details["close_0"] = close_0

        # Find close at +horizon_days
        target_dt = dt + timedelta(days=horizon_days)
        close_N = None
        for idx, row in hist.iterrows():
            ds = idx.strftime("%Y-%m-%d")
            if ds >= target_dt.strftime("%Y-%m-%d"):
                close_N = float(row["Close"])
                break

        if close_N is None:
            return "HOLD", details

        details["close_N"] = close_N
        pct_change = (close_N - close_0) / close_0
        details["pct_change"] = round(pct_change * 100, 2)

        # Threshold: 2% over 5 days (more robust than 1% over 1 day)
        threshold = 0.02
        if pct_change > threshold:
            return "BUY", details
        elif pct_change < -threshold:
            return "SELL", details
        else:
            return "HOLD", details

    except Exception as e:
        print(f"  Error getting price for {ticker}: {e}")
        return "HOLD", details


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
        actual, details = get_actual_signal(ticker, date_str)
        updated = engine.record_outcome(ticker, date_str, actual)
        pct_str = f"{details['pct_change']:+.2f}%" if details['pct_change'] is not None else "N/A"
        print(f"  {ticker}: actual={actual} ({pct_str} over {details['horizon']}d) — {updated} graded")

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

    # --- Drift scoring (prediction_drift_scorer.py) ---
    # Feed scored predictions into drift evaluator for strategy health tracking
    try:
        sys.path.insert(0, os.path.expanduser("~/TradingAgents"))
        from prediction_drift_scorer import PredictionDriftScorer

        drift_db = os.path.expanduser("~/TradingAgents/results/prediction_drift.db")
        scorer = PredictionDriftScorer(drift_db)

        # Pull all scored predictions for this date and feed to drift scorer
        conn2 = sqlite3.connect(db_path)
        conn2.row_factory = sqlite3.Row
        scored = conn2.execute(
            "SELECT ticker, date, signal as pred_dir, actual_signal FROM predictions WHERE date = ? AND actual_signal IS NOT NULL",
            (date_str,),
        ).fetchall()
        conn2.close()

        for s in scored:
            dir_map = {"BUY": "bullish", "SELL": "bearish", "HOLD": "neutral"}
            pred_dir = dir_map.get(s["actual_signal"], "neutral")
            # Record + score in drift DB
            scorer.record(s["ticker"], s["date"], dir_map.get(s["signal"], "neutral"), 0.0)
            scorer.score(s["ticker"], s["date"], pred_dir, 0.0)

        # Evaluate drift for each ticker scored today
        tickers_today = list(set(s["ticker"] for s in scored))
        if tickers_today:
            print(f"\n=== Prediction Drift Report ({date_str}) ===")
            for ticker in tickers_today:
                report = scorer.evaluate_drift(ticker)
                rec = report.get("recommendation", "")
                print(f"  {ticker}: {report['signal']} ({report['status']}) — {rec}")
    except Exception as e:
        print(f"  Drift scoring skipped: {e}")


if __name__ == "__main__":
    main()
