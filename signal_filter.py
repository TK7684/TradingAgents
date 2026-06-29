#!/usr/bin/env python3
"""Signal accuracy filter — prevents trading on historically bad predictions.

Reads prediction history from results/signal_accuracy.db and blocks/warns
based on rolling accuracy. Also provides market regime checking via SPY MAs.

Usage:
    python3 signal_filter.py                    # show ticker stats
    python3 signal_filter.py --check NVDA BUY  # test should_trade()
    python3 signal_filter.py --regime          # show market regime
"""
import sqlite3
import sys
import os
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.expanduser("~/TradingAgents")
DB_PATH = os.path.join(BASE_DIR, "results/signal_accuracy.db")

BLOCK_THRESHOLD = 20    # accuracy % below this → block (requires 20+ predictions)
WARN_THRESHOLD = 25     # source accuracy % below this → warn/downweight
MIN_PREDICTIONS = 20    # minimum predictions before blocking a ticker


class SignalFilter:
    """Filter trading signals based on historical prediction accuracy."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # ── Public API ─────────────────────────────────────────────────────

    def should_trade(self, ticker: str, signal: str) -> tuple[bool, str]:
        """Check whether a ticker/signal should be allowed to trade.

        Returns (allowed: bool, reason: str).
        Reasons starting with 'BLOCK' mean the trade is rejected.
        Reasons starting with 'WARN' mean allowed but downweighted.
        """
        accuracy, n = self._ticker_accuracy(ticker.upper())

        # Rule 1: Block tickers with <20% accuracy over 20+ predictions
        if n >= MIN_PREDICTIONS and accuracy < BLOCK_THRESHOLD:
            return False, (
                f"BLOCK: {ticker} has {accuracy:.1f}% accuracy over "
                f"{n} predictions — below {BLOCK_THRESHOLD}% threshold"
            )

        # Rule 3: Check if predicted signal direction is consistently wrong
        if n >= MIN_PREDICTIONS:
            anti_consistency = self._anti_signal_consistency(ticker.upper(), signal)
            if anti_consistency >= 0.6:
                return False, (
                    f"BLOCK: {ticker} predictions disagree with actual direction "
                    f"{anti_consistency:.0%} of the time — signal inversion detected"
                )

        # Rule 2: Check source-level accuracy for warnings
        worst_source, worst_acc = self._worst_source_accuracy()
        if worst_acc is not None and worst_acc < WARN_THRESHOLD:
            return True, (
                f"WARN: worst source '{worst_source}' has {worst_acc:.0f}% accuracy "
                f"(<{WARN_THRESHOLD}%) — downweight position by 50%"
            )

        return True, "OK"

    def get_position_weight(self, ticker: str) -> float:
        """Return position weight (0.0–1.0) based on ticker accuracy.

        < 20% accuracy  → 0.0 (do not trade)
        20–40% accuracy  → 0.5 (half position)
        40–60% accuracy  → 0.75 (reduced position)
        > 60% accuracy  → 1.0 (full position)
        """
        accuracy, n = self._ticker_accuracy(ticker.upper())

        if n < MIN_PREDICTIONS:
            return 0.75  # not enough data — conservative default

        if accuracy < 20:
            return 0.0
        elif accuracy < 40:
            return 0.5
        elif accuracy < 60:
            return 0.75
        else:
            return 1.0

    def get_ticker_stats(self) -> list[dict]:
        """Return accuracy stats for all tickers with 5+ predictions."""
        return self._query_ticker_stats()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _ticker_accuracy(self, ticker: str) -> tuple[float, int]:
        """Return (accuracy_pct, total_predictions) for a ticker."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT COUNT(*), SUM(correct) FROM predictions WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        conn.close()
        if not row or row[0] == 0:
            return 50.0, 0  # no data = assume neutral
        accuracy = (row[1] or 0) / row[0] * 100
        return accuracy, row[0]

    def _anti_signal_consistency(self, ticker: str, signal: str) -> float:
        """Measure how often predictions disagree with actual outcomes.

        A high score means the system is consistently wrong (inverse signal).
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT predicted_signal, actual_signal, correct "
            "FROM predictions WHERE ticker = ? "
            "AND predicted_signal != 'HOLD' AND actual_signal IS NOT NULL "
            "ORDER BY created_at DESC",
            (ticker,),
        ).fetchall()
        conn.close()

        if len(rows) < MIN_PREDICTIONS:
            return 0.0

        # Count cases where predicted direction was opposite to actual direction
        directional = {"BUY": 1, "SELL": -1, "HOLD": 0}
        disagreements = 0
        directional_count = 0
        for pred_sig, act_sig, correct in rows:
            if pred_sig not in directional or act_sig not in directional:
                continue
            directional_count += 1
            if directional.get(pred_sig, 0) * directional.get(act_sig, 0) < 0:
                disagreements += 1

        return disagreements / max(directional_count, 1)

    def _worst_source_accuracy(self) -> tuple[str, float]:
        """Return (worst_source_name, worst_accuracy)."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT source, accuracy FROM source_stats ORDER BY accuracy ASC LIMIT 1"
        ).fetchone()
        conn.close()
        if not row:
            return "unknown", 50.0
        return row[0], row[1] * 100  # stored as 0–1, convert to %

    def _query_ticker_stats(self) -> list[dict]:
        """Query all tickers with 5+ predictions."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT ticker, COUNT(*) as n, SUM(correct) as correct, "
            "ROUND(100.0 * SUM(correct) / COUNT(*), 1) as wr "
            "FROM predictions WHERE ticker NOT IN ('UNKNOWN', 'BTC') "
            "GROUP BY ticker HAVING n >= 5 ORDER BY wr ASC"
        ).fetchall()
        conn.close()
        return [
            {"ticker": r[0], "predictions": r[1], "correct": r[2], "accuracy": r[3]}
            for r in rows
        ]


# ── Market Regime Check ────────────────────────────────────────────────────

def check_market_regime() -> dict:
    """Check SPY regime using 20/50-day moving averages.

    Returns dict with:
        regime: "healthy" | "weak" | "bearish"
        allow_buys: bool
        spy_price: float
        ma20: float
        ma50: float
        reason: str
    """
    import yfinance as yf

    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="3mo")  # enough for 50-day MA

        if hist.empty:
            return {
                "regime": "unknown",
                "allow_buys": False,
                "spy_price": 0, "ma20": 0, "ma50": 0,
                "reason": "no SPY data available",
            }

        current_price = float(hist["Close"].iloc[-1])
        ma20 = float(hist["Close"].rolling(20).mean().iloc[-1])
        ma50 = float(hist["Close"].rolling(50).mean().iloc[-1])

        below_20 = current_price < ma20
        below_50 = current_price < ma50

        if below_50:
            regime = "bearish"
            allow_buys = False
            reason = (
                f"SPY ${current_price:.2f} below 50-day MA ${ma50:.2f} — "
                "bearish regime, no new buys, consider reducing"
            )
        elif below_20:
            regime = "weak"
            allow_buys = False
            reason = (
                f"SPY ${current_price:.2f} below 20-day MA ${ma20:.2f} — "
                "weak regime, no new buys"
            )
        else:
            regime = "healthy"
            allow_buys = True
            reason = (
                f"SPY ${current_price:.2f} above both MAs (20d=${ma20:.2f}, "
                f"50d=${ma50:.2f}) — healthy regime, buys allowed"
            )

        return {
            "regime": regime,
            "allow_buys": allow_buys,
            "spy_price": current_price,
            "ma20": ma20,
            "ma50": ma50,
            "reason": reason,
        }

    except Exception as e:
        return {
            "regime": "unknown",
            "allow_buys": False,
            "spy_price": 0, "ma20": 0, "ma50": 0,
            "reason": f"error fetching SPY data: {e}",
        }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        # Default: show stats
        sf = SignalFilter()
        print("=== Signal Accuracy Stats ===\n")
        stats = sf.get_ticker_stats()
        for s in stats:
            weight = sf.get_position_weight(s["ticker"])
            flag = "🔴 BLOCK" if weight == 0.0 else "🟡 HALF" if weight < 1.0 else "🟢 FULL"
            print(
                f"  {s['ticker']:<8} "
                f"{s['accuracy']:>5.1f}% ({s['correct']}/{s['predictions']})  "
                f"weight={weight:.2f}  {flag}"
            )
        print()

        regime = check_market_regime()
        emoji = "🟢" if regime["allow_buys"] else "🔴"
        print(f"{emoji} Market Regime: {regime['regime']}")
        print(f"   {regime['reason']}")
        return

    if sys.argv[1] == "--check" and len(sys.argv) >= 4:
        sf = SignalFilter()
        ticker, signal = sys.argv[2], sys.argv[3].upper()
        allowed, reason = sf.should_trade(ticker, signal)
        weight = sf.get_position_weight(ticker)
        print(f"  {ticker} {signal}: {'✅' if allowed else '❌'} {reason}")
        print(f"  Position weight: {weight}")
        return

    if sys.argv[1] == "--regime":
        regime = check_market_regime()
        print(f"  Regime: {regime['regime']}")
        print(f"  SPY: ${regime['spy_price']:.2f}")
        print(f"  MA20: ${regime['ma20']:.2f}")
        print(f"  MA50: ${regime['ma50']:.2f}")
        print(f"  Allow buys: {regime['allow_buys']}")
        print(f"  {regime['reason']}")
        return

    print("Usage:")
    print("  python3 signal_filter.py              # show stats")
    print("  python3 signal_filter.py --check NVDA BUY")
    print("  python3 signal_filter.py --regime")


if __name__ == "__main__":
    main()
