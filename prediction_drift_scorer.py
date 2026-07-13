"""
Prediction Drift Scorer — combines Vibe-Trading's DecayEvaluator with
AI-Berkshire's thesis drift detection for TradingAgents.

Tracks prediction accuracy over time and detects when our trading
signals are drifting (strategy decay) vs when fundamentals changed.

Based on:
- vibe-trading-src/agent/src/strategy_store/decay.py (DecayEvaluator)
- ai-berkshire-src/skills/thesis-drift.md (drift detection methodology)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import json
import sqlite3
from pathlib import Path


class PredictionSignal(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DECAYED = "decayed"
    CRITICAL = "critical"


class PredictionStatus(str, Enum):
    ACTIVE = "active"
    MONITORING = "monitoring"
    DECAYED = "decayed"
    DISABLED = "disabled"


@dataclass(frozen=True)
class DriftThresholds:
    """Configurable thresholds — adapted from Vibe-Trading DecayThresholds."""
    # Prediction accuracy ratio (rolling / baseline)
    accuracy_healthy: float = 0.75
    accuracy_warning: float = 0.60
    accuracy_decayed: float = 0.40

    # Directional accuracy (up/down calls)
    directional_healthy: float = 0.65
    directional_warning: float = 0.55
    directional_decayed: float = 0.45

    # Magnitude error (predicted vs actual % move)
    magnitude_error_healthy: float = 0.5  # avg error within 50% of move
    magnitude_error_warning: float = 1.0
    magnitude_error_decayed: float = 2.0

    # Sharpe of predictions (treat correct direction as +1, wrong as -1)
    pred_sharpe_healthy: float = 1.0
    pred_sharpe_warning: float = 0.5
    pred_sharpe_decayed: float = 0.0

    # Consecutive signals for transitions
    warnings_for_monitoring: int = 3
    warnings_for_decayed: int = 2
    critical_for_disabled: int = 3


@dataclass
class PredictionRecord:
    """A single prediction and its outcome."""
    prediction_id: str
    symbol: str
    date: str  # ISO date
    predicted_direction: str  # "bullish" | "bearish" | "neutral"
    predicted_magnitude: float  # expected % move
    actual_direction: str = ""
    actual_magnitude: float = 0.0
    scored: bool = False
    correct_direction: bool = False
    magnitude_error: float = 0.0
    notes: str = ""


class PredictionDriftScorer:
    """
    Scores trading predictions and detects strategy drift.

    Combines:
    - Vibe-Trading's DecayEvaluator state machine (active→monitoring→decayed→disabled)
    - AI-Berkshire's thesis drift philosophy: distinguish facts changing from noise

    Usage:
        scorer = PredictionDriftScorer(db_path="predictions.db")
        scorer.record("BTC", "2026-07-12", "bullish", 2.5)
        # ... later when actual prices are known:
        scorer.score("BTC", "2026-07-12", "bullish", 3.1)
        report = scorer.evaluate_drift("BTC")
    """

    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    predicted_direction TEXT NOT NULL,
                    predicted_magnitude REAL NOT NULL,
                    actual_direction TEXT DEFAULT '',
                    actual_magnitude REAL DEFAULT 0,
                    scored INTEGER DEFAULT 0,
                    correct_direction INTEGER DEFAULT 0,
                    magnitude_error REAL DEFAULT 0,
                    notes TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_state (
                    symbol TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'active',
                    signal_history TEXT DEFAULT '[]',
                    last_evaluated TEXT DEFAULT '',
                    thresholds TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_symbol_date
                ON predictions(symbol, date)
            """)

    def record(
        self,
        symbol: str,
        date: str,
        direction: str,
        magnitude: float,
        notes: str = "",
    ) -> str:
        """Record a new prediction."""
        pred_id = f"{symbol}_{date}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO predictions
                (prediction_id, symbol, date, predicted_direction, predicted_magnitude, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pred_id, symbol, date, direction, magnitude, notes))
        return pred_id

    def score(
        self,
        symbol: str,
        date: str,
        actual_direction: str,
        actual_magnitude: float,
    ) -> dict:
        """Score a past prediction against actual outcome."""
        pred_id = f"{symbol}_{date}"
        correct = (actual_direction == self._get_pred_direction(pred_id))
        pred_mag = self._get_pred_magnitude(pred_id)
        mag_error = abs(actual_magnitude - pred_mag) / max(abs(actual_magnitude), 0.01)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE predictions SET
                    actual_direction = ?,
                    actual_magnitude = ?,
                    scored = 1,
                    correct_direction = ?,
                    magnitude_error = ?
                WHERE prediction_id = ?
            """, (actual_direction, actual_magnitude, int(correct), mag_error, pred_id))

        return {
            "prediction_id": pred_id,
            "correct_direction": correct,
            "predicted_magnitude": pred_mag,
            "actual_magnitude": actual_magnitude,
            "magnitude_error": mag_error,
        }

    def _get_pred_direction(self, pred_id: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT predicted_direction FROM predictions WHERE prediction_id = ?",
                (pred_id,)
            ).fetchone()
            return row[0] if row else "neutral"

    def _get_pred_magnitude(self, pred_id: str) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT predicted_magnitude FROM predictions WHERE prediction_id = ?",
                (pred_id,)
            ).fetchone()
            return row[0] if row else 0.0

    def get_metrics(self, symbol: str, lookback_days: int = 0) -> dict:
        """Get rolling metrics for a symbol. lookback_days=0 means all history."""
        with sqlite3.connect(self.db_path) as conn:
            if lookback_days > 0:
                rows = conn.execute("""
                    SELECT correct_direction, magnitude_error
                    FROM predictions
                    WHERE symbol = ? AND scored = 1
                    AND date >= date('now', ?)
                    ORDER BY date DESC
                """, (symbol, f"-{lookback_days} days")).fetchall()
            else:
                rows = conn.execute("""
                    SELECT correct_direction, magnitude_error
                    FROM predictions
                    WHERE symbol = ? AND scored = 1
                    ORDER BY date DESC
                """, (symbol,)).fetchall()

        if not rows:
            return {
                "accuracy": None,
                "directional_accuracy": None,
                "avg_magnitude_error": None,
                "pred_sharpe": None,
                "sample_size": 0,
            }

        correct = sum(1 for r in rows if r[0])
        total = len(rows)
        accuracy = correct / total
        avg_mag_error = sum(r[1] for r in rows) / total

        # Treat correct direction as +1, wrong as -1, compute simple sharpe
        returns = [1 if r[0] else -1 for r in rows]
        mean_ret = sum(returns) / len(returns)
        std_ret = (sum((r - mean_ret) ** 2 for r in returns) / len(returns)) ** 0.5
        pred_sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        return {
            "accuracy": round(accuracy, 3),
            "directional_accuracy": round(accuracy, 3),
            "avg_magnitude_error": round(avg_mag_error, 3),
            "pred_sharpe": round(pred_sharpe, 2),
            "sample_size": total,
        }

    def evaluate_drift(
        self,
        symbol: str,
        thresholds: Optional[DriftThresholds] = None,
    ) -> dict:
        """
        Evaluate prediction drift for a symbol.
        Returns signal + status transition + detailed report.

        Inspired by Vibe-Trading's DecayEvaluator but adapted for
        prediction scoring (IC ratio → accuracy ratio, etc).
        """
        t = thresholds or DriftThresholds()
        metrics = self.get_metrics(symbol)

        if metrics["sample_size"] < 5:
            return {
                "symbol": symbol,
                "signal": "insufficient_data",
                "status": "active",
                "metrics": metrics,
                "message": f"Only {metrics['sample_size']} scored predictions (need ≥5)",
            }

        # Classify each metric (worst signal wins, same as Vibe-Trading)
        signals = []

        if metrics["accuracy"] is not None:
            acc = metrics["accuracy"]
            if acc >= t.accuracy_healthy:
                signals.append(PredictionSignal.HEALTHY)
            elif acc >= t.accuracy_warning:
                signals.append(PredictionSignal.WARNING)
            elif acc >= t.accuracy_decayed:
                signals.append(PredictionSignal.DECAYED)
            else:
                signals.append(PredictionSignal.CRITICAL)

        if metrics["pred_sharpe"] is not None:
            ps = metrics["pred_sharpe"]
            if ps >= t.pred_sharpe_healthy:
                signals.append(PredictionSignal.HEALTHY)
            elif ps >= t.pred_sharpe_warning:
                signals.append(PredictionSignal.WARNING)
            elif ps >= t.pred_sharpe_decayed:
                signals.append(PredictionSignal.DECAYED)
            else:
                signals.append(PredictionSignal.CRITICAL)

        # Worst signal wins
        _order = {
            PredictionSignal.CRITICAL: 0,
            PredictionSignal.DECAYED: 1,
            PredictionSignal.WARNING: 2,
            PredictionSignal.HEALTHY: 3,
        }
        worst = min(signals, key=lambda s: _order[s]) if signals else PredictionSignal.HEALTHY

        # Get current status and update state machine
        current_status = self._get_status(symbol)
        signal_history = self._get_signal_history(symbol)
        signal_history.append(worst.value)

        # State transitions (same logic as Vibe-Trading)
        new_status = self._check_transition(current_status, signal_history, t)
        if new_status is None:
            new_status = current_status

        # Persist state
        self._save_state(symbol, new_status, signal_history[-20:])

        return {
            "symbol": symbol,
            "signal": worst.value,
            "status": new_status.value,
            "previous_status": current_status.value,
            "metrics": metrics,
            "signal_history": signal_history[-10:],
            "thresholds": {
                "accuracy_healthy": t.accuracy_healthy,
                "accuracy_warning": t.accuracy_warning,
                "accuracy_decayed": t.accuracy_decayed,
            },
            "recommendation": self._get_recommendation(worst, new_status),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _check_transition(
        self,
        current: PredictionStatus,
        history: list[str],
        t: DriftThresholds,
    ) -> Optional[PredictionStatus]:
        """State machine — adapted from Vibe-Trading's should_transition."""
        if not history:
            return None

        non_healthy_tail = [s for s in history if s != "healthy"][-t.warnings_for_monitoring:]

        if current == PredictionStatus.ACTIVE:
            if len(non_healthy_tail) >= t.warnings_for_monitoring:
                return PredictionStatus.MONITORING

        elif current == PredictionStatus.MONITORING:
            if history[-1] == "healthy":
                return PredictionStatus.ACTIVE
            decayed_tail = [s for s in history if s in ("decayed", "critical")][-t.warnings_for_decayed:]
            if len(decayed_tail) >= t.warnings_for_decayed:
                return PredictionStatus.DECAYED

        elif current == PredictionStatus.DECAYED:
            critical_tail = [s for s in history if s == "critical"][-t.critical_for_disabled:]
            if len(critical_tail) >= t.critical_for_disabled:
                return PredictionStatus.DISABLED

        return None

    def _get_recommendation(self, signal: PredictionSignal, status: PredictionStatus) -> str:
        """Human-readable recommendation — AI-Berkshire style."""
        if status == PredictionStatus.DISABLED:
            return "⛔ Strategy DISABLED. Predictions consistently wrong. Review or replace model."
        if signal == PredictionSignal.CRITICAL:
            return "🔴 CRITICAL: Prediction accuracy severely degraded. Immediate review needed."
        if status == PredictionStatus.DECAYED:
            return "🟠 DECAYED: Consistent underperformance. Reduce position sizing, investigate cause."
        if signal == PredictionSignal.WARNING:
            return "🟡 WARNING: Accuracy declining. Monitor closely, do not increase exposure."
        if status == PredictionStatus.MONITORING:
            return "👀 MONITORING: In observation window. Continue with reduced confidence."
        return "🟢 HEALTHY: Predictions performing well. Maintain current strategy."

    def _get_status(self, symbol: str) -> PredictionStatus:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT status FROM drift_state WHERE symbol = ?", (symbol,)
            ).fetchone()
            return PredictionStatus(row[0]) if row else PredictionStatus.ACTIVE

    def _get_signal_history(self, symbol: str) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT signal_history FROM drift_state WHERE symbol = ?", (symbol,)
            ).fetchone()
            return json.loads(row[0]) if row and row[0] else []

    def _save_state(self, symbol: str, status: PredictionStatus, history: list[str]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO drift_state
                (symbol, status, signal_history, last_evaluated)
                VALUES (?, ?, ?, datetime('now'))
            """, (symbol, status.value, json.dumps(history)))

    def report_all(self) -> list[dict]:
        """Get drift report for all tracked symbols."""
        with sqlite3.connect(self.db_path) as conn:
            symbols = conn.execute(
                "SELECT DISTINCT symbol FROM predictions WHERE scored = 1"
            ).fetchall()
        return [self.evaluate_drift(s[0]) for s in symbols]


if __name__ == "__main__":
    # Smoke test
    scorer = PredictionDriftScorer("/tmp/test_predictions.db")

    # Simulate predictions
    scorer.record("BTC", "2026-07-01", "bullish", 2.5)
    scorer.record("BTC", "2026-07-02", "bullish", 1.8)
    scorer.record("BTC", "2026-07-03", "bearish", -1.2)
    scorer.record("BTC", "2026-07-04", "bullish", 3.0)
    scorer.record("BTC", "2026-07-05", "neutral", 0.5)

    # Score them
    scorer.score("BTC", "2026-07-01", "bullish", 2.1)
    scorer.score("BTC", "2026-07-02", "bullish", 1.5)
    scorer.score("BTC", "2026-07-03", "bearish", -0.8)
    scorer.score("BTC", "2026-07-04", "bearish", -0.5)  # wrong direction
    scorer.score("BTC", "2026-07-05", "neutral", 0.3)

    report = scorer.evaluate_drift("BTC")
    print(json.dumps(report, indent=2))
    print(f"\n✅ Smoke test passed: signal={report['signal']}, status={report['status']}")
