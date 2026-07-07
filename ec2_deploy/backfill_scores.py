#!/usr/bin/env python3
"""Retroactively grade ALL ungraded predictions against actual price movements.

Fetches next-day prices via yfinance for each (ticker, date) pair that lacks
an actual_signal.  Also computes multi-day returns (3-day, 5-day) for richer
accuracy tracking.

Usage:
    python ec2_deploy/backfill_scores.py              # grade all ungraded
    python ec2_deploy/backfill_scores.py --date 2026-05-19  # specific date
    python ec2_deploy/backfill_scores.py --dry-run     # show what would be graded
"""
import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.expanduser("~/TradingAgents"))

import yfinance as yf

DB_PATH = os.path.expanduser("~/TradingAgents/results/signal_accuracy.db")
PRICE_THRESHOLD = 0.01  # 1% move = directional; below = HOLD


def get_price_data(ticker: str, date_str: str, days_ahead: int = 7) -> dict:
    """Fetch price data for ticker around date_str.

    Returns dict with 'day0_close', 'day1_close', 'day3_close', 'day5_close'
    and corresponding pct_changes, or None on failure.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Need data from date-1 to date+days_ahead+5 (buffer for weekends)
    start = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=days_ahead + 10)).strftime("%Y-%m-%d")

    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return None

        # Find the row for date_str (or closest prior)
        target_dates = [d.strftime("%Y-%m-%d") for d in [dt]]
        result = {"day0_close": None, "day1_close": None, "day3_close": None, "day5_close": None}

        # Get close on signal date
        signal_close = None
        for idx, row in hist.iterrows():
            ds = idx.strftime("%Y-%m-%d")
            if ds == date_str:
                signal_close = float(row["Close"])
                break

        if signal_close is None:
            return None

        result["day0_close"] = signal_close

        # Get close at day+1, +3, +5
        target_offsets = {"day1_close": 1, "day3_close": 3, "day5_close": 5}
        for key, offset in target_offsets.items():
            target_dt = dt + timedelta(days=offset)
            # Find closest trading day on or after target
            for idx, row in hist.iterrows():
                ds = idx.strftime("%Y-%m-%d")
                if ds >= target_dt.strftime("%Y-%m-%d"):
                    result[key] = float(row["Close"])
                    break

        # Compute pct changes
        result["pct_1d"] = (
            (result["day1_close"] - signal_close) / signal_close * 100
            if result["day1_close"] and signal_close else None
        )
        result["pct_3d"] = (
            (result["day3_close"] - signal_close) / signal_close * 100
            if result["day3_close"] and signal_close else None
        )
        result["pct_5d"] = (
            (result["day5_close"] - signal_close) / signal_close * 100
            if result["day5_close"] and signal_close else None
        )

        return result

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def determine_actual_signal(pct_change: float | None) -> str:
    """Determine correct signal based on price movement."""
    if pct_change is None:
        return "HOLD"
    if pct_change > 1.0:
        return "BUY"
    elif pct_change < -1.0:
        return "SELL"
    else:
        return "HOLD"


def main():
    parser = argparse.ArgumentParser(description="Backfill all ungraded predictions")
    parser.add_argument("--date", default=None, help="Specific date to grade")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be graded")
    parser.add_argument("--days", type=int, default=1, choices=[1, 3, 5],
                        help="Use N-day returns for grading (default: 1-day)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get ungraded pairs
    if args.date:
        rows = conn.execute(
            "SELECT DISTINCT ticker, date FROM predictions "
            "WHERE date = ? AND actual_signal IS NULL",
            (args.date,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT ticker, date FROM predictions "
            "WHERE actual_signal IS NULL ORDER BY date"
        ).fetchall()

    if not rows:
        print("No ungraded predictions found.")
        conn.close()
        return

    print(f"Found {len(rows)} ungraded (ticker, date) pairs")

    if args.dry_run:
        for r in rows:
            print(f"  {r['ticker']} @ {r['date']}")
        conn.close()
        return

    # Add columns for multi-day tracking if not present
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN pct_1d REAL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN pct_3d REAL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN pct_5d REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()

    graded = 0
    errors = 0

    for r in rows:
        ticker = r["ticker"]
        date_str = r["date"]

        # Skip obviously bad data (non-dates, pre-2026)
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"  SKIP {ticker} @ {date_str} (invalid date format)")
            continue
        if dt.year < 2026:
            print(f"  SKIP {ticker} @ {date_str} (pre-2026, likely stale)")
            continue

        print(f"  {ticker} @ {date_str}...", end=" ", flush=True)
        price_data = get_price_data(ticker, date_str)
        time.sleep(0.3)  # rate limit

        if price_data is None or price_data["day0_close"] is None:
            print("NO DATA")
            errors += 1
            continue

        pct_key = f"pct_{args.days}d"
        pct = price_data.get(pct_key)
        actual = determine_actual_signal(pct)

        # Grade all 4 sources for this (ticker, date)
        preds = conn.execute(
            "SELECT id, predicted_signal FROM predictions "
            "WHERE ticker = ? AND date = ? AND actual_signal IS NULL",
            (ticker, date_str),
        ).fetchall()

        for p in preds:
            correct = int(p["predicted_signal"].upper() == actual)
            conn.execute(
                "UPDATE predictions SET actual_signal = ?, correct = ?, "
                "pct_1d = ?, pct_3d = ?, pct_5d = ? WHERE id = ?",
                (actual, correct, price_data.get("pct_1d"), price_data.get("pct_3d"),
                 price_data.get("pct_5d"), p["id"]),
            )
            graded += 1

        pct_1d_str = f"{price_data.get('pct_1d', 0):+.2f}%" if price_data.get("pct_1d") else "N/A"
        pct_3d_str = f"{price_data.get('pct_3d', 0):+.2f}%" if price_data.get("pct_3d") else "N/A"
        pct_5d_str = f"{price_data.get('pct_5d', 0):+.2f}%" if price_data.get("pct_5d") else "N/A"
        print(f"actual={actual} (1d={pct_1d_str} 3d={pct_3d_str} 5d={pct_5d_str})")

    conn.commit()

    # Refresh source_stats
    conn.execute("DELETE FROM source_stats")
    conn.execute(
        "INSERT INTO source_stats (source, total_predictions, correct_predictions, accuracy, updated_at) "
        "SELECT source, COUNT(*), SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END), "
        "ROUND(CAST(SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS REAL) / NULLIF(COUNT(*), 0), 4), "
        "datetime('now') FROM predictions WHERE actual_signal IS NOT NULL GROUP BY source"
    )
    conn.commit()

    # Print stats
    print(f"\n=== Results ===")
    print(f"Graded: {graded} predictions")
    print(f"Errors: {errors}")
    print()

    # Source stats
    stats = conn.execute("SELECT source, total_predictions, correct_predictions, accuracy FROM source_stats ORDER BY source").fetchall()
    print("Source accuracy (1-day returns):")
    for s in stats:
        pct = s["accuracy"] * 100 if s["accuracy"] else 0
        print(f"  {s['source']}: {s['correct_predictions']}/{s['total_predictions']} ({pct:.1f}%)")

    # Also compute 3-day and 5-day accuracy
    for days in [3, 5]:
        print(f"\nSource accuracy ({days}-day returns):")
        for source in ["investment_judge", "trader", "risk_judge", "portfolio_manager"]:
            rows_src = conn.execute(
                f"SELECT predicted_signal, pct_{days}d FROM predictions "
                f"WHERE source = ? AND actual_signal IS NOT NULL AND pct_{days}d IS NOT NULL",
                (source,),
            ).fetchall()
            if not rows_src:
                print(f"  {source}: N/A (no data)")
                continue
            correct = 0
            total = len(rows_src)
            for r in rows_src:
                pred = r["predicted_signal"]
                actual_d = determine_actual_signal(r[f"pct_{days}d"])
                if pred == actual_d:
                    correct += 1
            pct = correct / total * 100
            print(f"  {source}: {correct}/{total} ({pct:.1f}%)")

    # Overall stats
    all_graded = conn.execute(
        "SELECT predicted_signal, actual_signal, correct, pct_1d, pct_3d, pct_5d "
        "FROM predictions WHERE actual_signal IS NOT NULL"
    ).fetchall()
    if all_graded:
        total_correct = sum(1 for r in all_graded if r["correct"])
        print(f"\nOverall: {total_correct}/{len(all_graded)} ({total_correct/len(all_graded)*100:.1f}%)")

        # Confusion matrix
        from collections import Counter
        combos = Counter((r["predicted_signal"], r["actual_signal"]) for r in all_graded)
        print("\nConfusion matrix (predicted → actual):")
        print(f"  {'':>8} | {'BUY':>6} {'HOLD':>6} {'SELL':>6}")
        for pred in ["BUY", "HOLD", "SELL"]:
            cells = []
            for act in ["BUY", "HOLD", "SELL"]:
                cells.append(f"{combos.get((pred, act), 0):>6}")
            print(f"  {pred:>8} | {' '.join(cells)}")

    conn.close()


if __name__ == "__main__":
    main()
