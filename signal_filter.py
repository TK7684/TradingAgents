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

# Empirical-Bayes (beta-binomial) shrinkage for ticker accuracy.
# Raw correct/total overfits small samples; shrink toward the 50% neutral
# prior by EB_PRIOR_STRENGTH pseudo-observations. Matches the convention
# already deployed for source weights in tradingagents/graph/consensus.py
# (BAYES_PRIOR_STRENGTH = 5). A 1/1 ticker no longer reads as 100%.
EB_PRIOR_MEAN = 0.5     # neutral prior mean (no data = coin flip)
EB_PRIOR_STRENGTH = 5   # pseudo-observations pulling toward the prior

# Recency weighting (non-stationary bandit convention): each prediction's
# vote decays by half every RECENCY_HALF_LIFE_DAYS, so stale regimes stop
# dominating the gate. Composed with EB shrinkage: weighted correct/total
# feed the same beta-binomial prior (pseudo-observations carry weight 1).
RECENCY_HALF_LIFE_DAYS = 60.0


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
        """Return (recency-weighted, EB-shrunk accuracy_pct, total_predictions).

        Each prediction votes with weight 0.5 ** (age_days /
        RECENCY_HALF_LIFE_DAYS); the weighted correct/total then feed the
        beta-binomial shrinkage:

        accuracy = (w_correct + EB_PRIOR_STRENGTH * EB_PRIOR_MEAN) /
                   (w_total + EB_PRIOR_STRENGTH) * 100

        Equal timestamps reduce to the previous unweighted EB estimate.
        Small samples are still pulled toward 50%.
        """
        w_correct, w_total, n = self._weighted_counts(ticker.upper())
        if n == 0:
            return 50.0, 0  # no data = assume neutral
        shrunk = (w_correct + EB_PRIOR_STRENGTH * EB_PRIOR_MEAN) / (
            w_total + EB_PRIOR_STRENGTH
        )
        return shrunk * 100, n

    def _weighted_counts(self, ticker: str) -> tuple[float, float, int]:
        """Recency-weighted (w_correct, w_total, raw_n) for a ticker.

        Shared by the gate (_ticker_accuracy) and the stats display
        (_query_ticker_stats) so display always equals the decision basis.
        Unparseable/future timestamps fall back to full weight (safe).
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT correct, created_at FROM predictions WHERE ticker = ?",
            (ticker,),
        ).fetchall()
        conn.close()
        w_correct = w_total = 0.0
        n = 0
        for correct, created_at in rows:
            n += 1
            weight = 1.0
            if created_at:
                try:
                    age_days = (now - datetime.fromisoformat(
                        str(created_at).replace(" ", "T"))).total_seconds() / 86400.0
                    if age_days > 0:
                        weight = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
                except ValueError:
                    pass
            w_total += weight
            if correct:
                w_correct += weight
        return w_correct, w_total, n

    def _anti_signal_consistency(self, ticker: str, signal: str) -> float:
        """Measure how often predictions disagree with actual outcomes.

        A high score means the system is consistently wrong (inverse signal).
        Recency-weighted with the same decay as _ticker_accuracy: a ticker
        whose inversion regime is stale (recovered) is not flagged forever,
        while a currently-inverted ticker still is.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT predicted_signal, actual_signal, created_at "
            "FROM predictions WHERE ticker = ? "
            "AND predicted_signal != 'HOLD' AND actual_signal IS NOT NULL "
            "ORDER BY created_at DESC",
            (ticker,),
        ).fetchall()
        conn.close()

        if len(rows) < MIN_PREDICTIONS:
            return 0.0

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Weight cases where predicted direction was opposite to actual
        directional = {"BUY": 1, "SELL": -1, "HOLD": 0}
        disagreements = 0.0
        directional_count = 0.0
        for pred_sig, act_sig, created_at in rows:
            if pred_sig not in directional or act_sig not in directional:
                continue
            weight = 1.0
            if created_at:
                try:
                    age_days = (now - datetime.fromisoformat(
                        str(created_at).replace(" ", "T"))).total_seconds() / 86400.0
                    if age_days > 0:
                        weight = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
                except ValueError:
                    pass
            directional_count += weight
            if directional.get(pred_sig, 0) * directional.get(act_sig, 0) < 0:
                disagreements += weight

        return disagreements / max(directional_count, 1e-9)

    def _worst_source_accuracy(self) -> tuple[str, float]:
        """Return (worst_source_name, worst_accuracy_pct).

        Accuracy is EB-shrunk with the same beta-binomial prior as ticker
        accuracy, and "worst" is ranked by the shrunk value, so a 0/2
        source reads ~35.7% instead of 0% and cannot trigger a spurious
        WARN downweight off a tiny sample.
        """
        prior_correct = EB_PRIOR_STRENGTH * EB_PRIOR_MEAN
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT source, "
            "  100.0 * (correct_predictions + ?) / (total_predictions + ?) "
            "FROM source_stats "
            "ORDER BY (correct_predictions + ?) * 1.0 / (total_predictions + ?) ASC "
            "LIMIT 1",
            (prior_correct, EB_PRIOR_STRENGTH, prior_correct, EB_PRIOR_STRENGTH),
        ).fetchone()
        conn.close()
        if not row:
            return "unknown", 50.0
        return row[0], row[1]

    def _query_ticker_stats(self) -> list[dict]:
        """Query all tickers with 5+ predictions.

        Accuracy is EB-shrunk (same beta-binomial prior as
        _ticker_accuracy) so the displayed number matches the value the
        gating decision actually uses; raw correct/total overstates
        small samples (AGI pattern: small-sample accuracy overfitting).
        """
        conn = sqlite3.connect(self.db_path)
        tickers = [r[0] for r in conn.execute(
            "SELECT DISTINCT ticker FROM predictions "
            "WHERE ticker NOT IN ('UNKNOWN', 'BTC')"
        ).fetchall()]
        conn.close()
        stats = []
        for ticker in tickers:
            w_correct, w_total, n = self._weighted_counts(ticker)
            if n < 5:
                continue
            stats.append({
                "ticker": ticker,
                "predictions": n,
                "correct": round(w_correct, 2),
                "accuracy": round(
                    100.0 * (w_correct + EB_PRIOR_STRENGTH * EB_PRIOR_MEAN)
                    / (w_total + EB_PRIOR_STRENGTH), 1),
            })
        stats.sort(key=lambda s: s["accuracy"])
        return stats


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
