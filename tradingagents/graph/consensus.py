"""Consensus voting and confidence scoring for TradingAgents multi-agent decisions.

Extracts independent signals from all 4 decision points in the pipeline,
weights them by historical accuracy, and computes a confidence-scored
consensus recommendation.  Persists accuracy tracking in SQLite so that
weights improve over time.

Usage::

    from tradingagents.graph.consensus import ConsensusEngine

    engine = ConsensusEngine()
    result = engine.evaluate(log_states_dict)
    # result.final_signal, result.confidence, result.recommendation
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid raw signals produced by agents.
VALID_SIGNALS = {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}

#: Normalised signals used for voting.
NORMALISED_SIGNALS = {"BUY", "HOLD", "SELL"}

#: Mapping from verbose signals to the canonical 3-way set.
SIGNAL_NORMALISATION: Dict[str, str] = {
    "BUY": "BUY",
    "OVERWEIGHT": "BUY",
    "HOLD": "HOLD",
    "UNDERWEIGHT": "SELL",
    "SELL": "SELL",
}

#: The four signal sources extracted from the pipeline.
SOURCES = [
    "investment_judge",
    "trader",
    "risk_judge",
    "portfolio_manager",
]

#: Minimum floor for any source weight (no source is ever fully zeroed).
MIN_WEIGHT: float = 0.1

#: Minimum total recorded predictions before accuracy-based weighting kicks in.
MIN_PREDICTIONS_FOR_WEIGHTING: int = 10

#: Confidence threshold above which a signal is tradeable.
CONFIDENCE_TRADE_THRESHOLD: float = 0.6

#: Confidence + unanimity threshold for a STRONG signal.
CONFIDENCE_STRONG_THRESHOLD: float = 0.85

# Anti-BUY-bias correction
#: When all sources agree on BUY, apply this discount to reflect base-rate overconfidence.
BUY_UNANIMITY_DISCOUNT: float = 0.15

#: When investment_judge disagrees with execution block, this penalty applies to the majority.
DISAGREEMENT_PENALTY: float = 0.25

#: Combined weight of the 3 execution sources (trader, risk_judge, PM) vs investment_judge.
EXECUTION_BLOCK_WEIGHT: float = 0.40
INVESTMENT_JUDGE_WEIGHT: float = 0.60

# ---------------------------------------------------------------------------
# Regex patterns (mirrors signal_processing.py)
# ---------------------------------------------------------------------------

_RE_RATING = re.compile(
    r"(?:\*\*)?Rating(?:\*\*)?\s*[:：]\s*(?:\*\*)?(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)(?:\*\*)?",
    re.IGNORECASE,
)

_RE_PROPOSAL = re.compile(
    r"FINAL\s+TRANSACTION\s+PROPOSAL\s*[:：]\s*\*\*(BUY|HOLD|SELL)\*\*",
    re.IGNORECASE,
)

_RE_NUMBERED_RATING = re.compile(
    r"1\.\s*\*\*Rating\*\*\s*[:：]\s*(?:\*\*)?(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)(?:\*\*)?",
    re.IGNORECASE,
)

_RE_JUDGE_DECISION = re.compile(
    r"(?:\*\*)?Decision(?:\*\*)?\s*[:：]\s*(?:\*\*)?(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)(?:\*\*)?",
    re.IGNORECASE,
)

_RE_VERDICT = re.compile(
    r"(?:\*\*)?Verdict(?:\*\*)?\s*[:：]\s*(?:\*\*)?(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)(?:\*\*)?",
    re.IGNORECASE,
)

#: Ordered list of regex patterns tried in sequence.
_EXTRACTION_PATTERNS = [
    ("rating", _RE_RATING),
    ("proposal", _RE_PROPOSAL),
    ("numbered_rating", _RE_NUMBERED_RATING),
    ("judge_decision", _RE_JUDGE_DECISION),
    ("verdict", _RE_VERDICT),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Result of extracting a single signal from one source."""

    signal: str  # BUY, HOLD, or SELL (normalised)
    raw_text: str
    extraction_method: str  # which pattern or fallback method matched


@dataclass
class ConsensusResult:
    """Aggregated consensus output for a single ticker/date evaluation."""

    final_signal: str  # BUY, HOLD, or SELL
    confidence: float  # 0.0 – 1.0
    source_signals: Dict[str, str]  # source -> normalised signal
    weights: Dict[str, float]  # source -> weight used
    unanimous: bool
    recommendation: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    extractions: Dict[str, ExtractionResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Signal Extraction
# ---------------------------------------------------------------------------

class ConsensusSignalExtractor:
    """Extract BUY/HOLD/SELL signals from each of the four pipeline sources.

    Each extraction returns an :class:`ExtractionResult` so that callers
    can inspect *how* the signal was derived.
    """

    @staticmethod
    def _extract_raw_signal(text: str) -> Tuple[str, str]:
        """Attempt regex extraction; return (signal, method) or ("HOLD", "fallback_none")."""
        if not text or not isinstance(text, str) or not text.strip():
            return "HOLD", "fallback_empty"

        # 1. Structured regex patterns
        for method_name, pattern in _EXTRACTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).upper(), f"regex_{method_name}"

        # 2. Word-boundary frequency scan (same logic as SignalProcessor)
        signal_counts: Dict[str, int] = {}
        for sig in VALID_SIGNALS:
            matches = re.findall(rf"\b{sig}\b", text, re.IGNORECASE)
            if matches:
                signal_counts[sig] = len(matches)

        if signal_counts:
            best = max(signal_counts, key=signal_counts.get)
            # Prefer directional over HOLD when tied or more frequent
            if best == "HOLD" and len(signal_counts) > 1:
                for preferred in ["SELL", "UNDERWEIGHT", "BUY", "OVERWEIGHT"]:
                    if preferred in signal_counts and signal_counts[preferred] >= signal_counts[best]:
                        best = preferred
                        break
            return best, "frequency_scan"

        return "HOLD", "fallback_no_match"

    @staticmethod
    def _normalise(signal: str) -> str:
        """Map OVERWEIGHT→BUY, UNDERWEIGHT→SELL, passthrough for others."""
        return SIGNAL_NORMALISATION.get(signal.upper(), "HOLD")

    def extract_from_state(self, state: Dict[str, Any]) -> Dict[str, ExtractionResult]:
        """Extract signals from all four sources given a single date's state dict.

        Args:
            state: The value from ``log_states_dict[trade_date]``, i.e. the
                   dict produced by :meth:`TradingAgentsGraph._log_state`.

        Returns:
            Mapping of source name to :class:`ExtractionResult`.
        """
        results: Dict[str, ExtractionResult] = {}

        # 1. Investment Debate Judge
        raw = ""
        try:
            raw = (state.get("investment_debate_state") or {}).get("judge_decision", "")
        except (AttributeError, TypeError):
            log.warning("Could not access investment_debate_state.judge_decision")
        results["investment_judge"] = self._make_extraction(raw)

        # 2. Trader (investment plan)
        raw = ""
        try:
            raw = state.get("trader_investment_decision", "")
        except (AttributeError, TypeError):
            log.warning("Could not access trader_investment_decision")
        results["trader"] = self._make_extraction(raw)

        # 3. Risk Debate Judge
        raw = ""
        try:
            raw = (state.get("risk_debate_state") or {}).get("judge_decision", "")
        except (AttributeError, TypeError):
            log.warning("Could not access risk_debate_state.judge_decision")
        results["risk_judge"] = self._make_extraction(raw)

        # 4. Portfolio Manager
        raw = ""
        try:
            raw = state.get("final_trade_decision", "")
        except (AttributeError, TypeError):
            log.warning("Could not access final_trade_decision")
        results["portfolio_manager"] = self._make_extraction(raw)

        return results

    def _make_extraction(self, raw: Any) -> ExtractionResult:
        """Build an :class:`ExtractionResult` from raw text."""
        raw_str = str(raw) if raw else ""
        signal, method = self._extract_raw_signal(raw_str)
        normalised = self._normalise(signal)
        return ExtractionResult(
            signal=normalised,
            raw_text=raw_str[:500],  # cap for logging/storage
            extraction_method=method,
        )


# ---------------------------------------------------------------------------
# Accuracy Tracker (SQLite-backed)
# ---------------------------------------------------------------------------

class AccuracyTracker:
    """Persist prediction accuracy in SQLite and compute dynamic weights.

    The database lives at ``~/TradingAgents/results/signal_accuracy.db``.
    """

    _DDL_PREDICTIONS = """
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            predicted_signal TEXT NOT NULL,
            actual_signal    TEXT,
            correct     INTEGER,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );
    """

    _DDL_STATS = """
        CREATE TABLE IF NOT EXISTS source_stats (
            source             TEXT PRIMARY KEY,
            total_predictions  INTEGER NOT NULL DEFAULT 0,
            correct_predictions INTEGER NOT NULL DEFAULT 0,
            accuracy           REAL NOT NULL DEFAULT 0.0,
            updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.expanduser(
                "~/TradingAgents/results/signal_accuracy.db"
            )
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    # -- connection management ------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            # Allow concurrent readers from other processes
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_tables(self) -> None:
        conn = self._get_conn()
        conn.executescript(self._DDL_PREDICTIONS + self._DDL_STATS)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- recording ------------------------------------------------------------

    def record_prediction(
        self,
        ticker: str,
        date_str: str,
        source: str,
        predicted_signal: str,
    ) -> None:
        """Insert a prediction record.

        Args:
            ticker: e.g. ``"AAPL"``.
            date_str: e.g. ``"2026-05-30"``.
            source: one of ``SOURCES``.
            predicted_signal: ``"BUY"``, ``"HOLD"``, or ``"SELL"``.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO predictions (ticker, date, source, predicted_signal) "
                "VALUES (?, ?, ?, ?)",
                (ticker, date_str, source, predicted_signal.upper()),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            log.warning(
                "Duplicate prediction for %s/%s/%s — skipping insert",
                ticker, date_str, source,
            )
        except sqlite3.Error as exc:
            log.error("Failed to record prediction: %s", exc)

    def record_outcome(
        self,
        ticker: str,
        date_str: str,
        actual_signal: str,
    ) -> int:
        """Mark all predictions for this ticker/date as correct/incorrect.

        Updates the ``predictions`` table and refreshes ``source_stats``.

        Returns:
            Number of prediction rows updated.
        """
        conn = self._get_conn()
        actual = actual_signal.upper()

        # Fetch pending predictions (no actual_signal yet)
        rows = conn.execute(
            "SELECT id, predicted_signal FROM predictions "
            "WHERE ticker = ? AND date = ? AND actual_signal IS NULL",
            (ticker, date_str),
        ).fetchall()

        updated = 0
        for row in rows:
            correct = int(row["predicted_signal"] == actual)
            conn.execute(
                "UPDATE predictions SET actual_signal = ?, correct = ? WHERE id = ?",
                (actual, correct, row["id"]),
            )
            updated += 1

        conn.commit()
        self._refresh_stats()
        return updated

    # -- weight computation ----------------------------------------------------

    def _refresh_stats(self) -> None:
        """Recompute ``source_stats`` from ``predictions``."""
        conn = self._get_conn()
        conn.execute("DELETE FROM source_stats")

        conn.execute(
            "INSERT INTO source_stats (source, total_predictions, correct_predictions, accuracy, updated_at) "
            "SELECT "
            "  source, "
            "  COUNT(*), "
            "  SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END), "
            "  ROUND(CAST(SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS REAL) "
            "        / NULLIF(COUNT(*), 0), 4), "
            "  datetime('now') "
            "FROM predictions "
            "WHERE actual_signal IS NOT NULL "
            "GROUP BY source"
        )
        conn.commit()

    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return accuracy statistics per source.

        Returns:
            ``{source: {"total": int, "correct": int, "accuracy": float}}``
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source, total_predictions, correct_predictions, accuracy "
            "FROM source_stats"
        ).fetchall()
        return {
            row["source"]: {
                "total": row["total_predictions"],
                "correct": row["correct_predictions"],
                "accuracy": row["accuracy"],
            }
            for row in rows
        }

    def get_ticker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return accuracy statistics per ticker.

        Returns:
            ``{ticker: {"total": int, "correct": int, "accuracy": float,
                       "buy_correct": int, "buy_total": int}}``
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT ticker, COUNT(*) as total, "
            "SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct, "
            "ROUND(CAST(SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS REAL) "
            "/ NULLIF(COUNT(*), 0), 4) as accuracy, "
            "SUM(CASE WHEN predicted_signal = 'BUY' AND correct = 1 THEN 1 ELSE 0 END) as buy_correct, "
            "SUM(CASE WHEN predicted_signal = 'BUY' THEN 1 ELSE 0 END) as buy_total, "
            "SUM(CASE WHEN predicted_signal = 'SELL' AND correct = 1 THEN 1 ELSE 0 END) as sell_correct, "
            "SUM(CASE WHEN predicted_signal = 'SELL' THEN 1 ELSE 0 END) as sell_total "
            "FROM predictions WHERE actual_signal IS NOT NULL "
            "GROUP BY ticker ORDER BY total DESC"
        ).fetchall()
        return {
            row["ticker"]: {
                "total": row["total"],
                "correct": row["correct"],
                "accuracy": row["accuracy"],
                "buy_correct": row["buy_correct"],
                "buy_total": row["buy_total"],
                "sell_correct": row["sell_correct"],
                "sell_total": row["sell_total"],
            }
            for row in rows
        }

    def get_weights(self) -> Dict[str, float]:
        """Compute accuracy-weighted weights for each source.

        Rules:
        * If fewer than ``MIN_PREDICTIONS_FOR_WEIGHTING`` total graded
          predictions exist, returns equal weights.
        * Each source weight is floored at ``MIN_WEIGHT``.
        * Weights are normalised so they sum to 1.0.

        Returns:
            ``{source: weight}`` for all sources in ``SOURCES``.
        """
        stats = self.get_source_stats()
        total_graded = sum(s["total"] for s in stats.values())

        if total_graded < MIN_PREDICTIONS_FOR_WEIGHTING:
            log.debug(
                "Only %d graded predictions (< %d), using equal weights",
                total_graded,
                MIN_PREDICTIONS_FOR_WEIGHTING,
            )
            equal = 1.0 / len(SOURCES)
            return {src: round(equal, 4) for src in SOURCES}

        # Build raw weights from accuracy; default to 0.5 for unseen sources
        raw: Dict[str, float] = {}
        for src in SOURCES:
            acc = stats.get(src, {}).get("accuracy", 0.5)
            raw[src] = max(float(acc), MIN_WEIGHT)

        # Normalise
        total = sum(raw.values())
        if total == 0:
            equal = 1.0 / len(SOURCES)
            return {src: round(equal, 4) for src in SOURCES}

        return {src: round(w / total, 4) for src, w in raw.items()}


# ---------------------------------------------------------------------------
# Confidence Scorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """Compute consensus and confidence from per-source signals.

    Uses :class:`AccuracyTracker` for dynamic weighting or falls back to
    equal weights when insufficient history exists.
    """

    def __init__(self, accuracy_tracker: AccuracyTracker):
        self._tracker = accuracy_tracker

    def score(
        self,
        source_signals: Dict[str, str],
        source_extractions: Optional[Dict[str, ExtractionResult]] = None,
    ) -> ConsensusResult:
        """Score a set of per-source signals with anti-bias corrections.

        Two voting tiers:
        1. **Investment judge** (weight=0.60) — the analytical voice that sees
           fundamentals.  Historically the most accurate source (30.4% vs 26.1%).
        2. **Execution block** (trader+risk_judge+portfolio_manager, weight=0.40) —
           the operational voices that tend to echo each other.  Their signal is
           derived by majority vote within the block.

        Anti-BUY-bias corrections:
        - BUY unanimity discount: if both tiers say BUY, confidence is reduced
          because the system's base BUY rate is 80% (prior doesn't add information).
        - Disagreement penalty: if tiers disagree, the majority signal loses
          confidence (the minority voice has historically been valuable).
        """
        # --- Tiered voting ---
        ij_signal = source_signals.get("investment_judge", "HOLD")
        exec_signals = [
            source_signals.get("trader", "HOLD"),
            source_signals.get("risk_judge", "HOLD"),
            source_signals.get("portfolio_manager", "HOLD"),
        ]
        # Majority vote within execution block
        exec_tally: Dict[str, int] = {}
        for s in exec_signals:
            exec_tally[s] = exec_tally.get(s, 0) + 1
        exec_signal = max(exec_tally, key=lambda k: (exec_tally[k], k))

        # Weighted tally (2 tiers)
        tally: Dict[str, float] = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
        tally[ij_signal] += INVESTMENT_JUDGE_WEIGHT
        tally[exec_signal] += EXECUTION_BLOCK_WEIGHT

        final_signal = max(tally, key=lambda k: tally[k])
        consensus_level = tally[final_signal]  # 0.0 – 1.0 since weights sum to 1.0

        # --- Unanimity check (original 4-source) ---
        signals_set = set(source_signals.get(src, "HOLD") for src in SOURCES)
        unanimous = len(signals_set) == 1

        # --- Anti-BUY-bias corrections ---
        tiers_disagree = (ij_signal != exec_signal)
        buy_correction = 0.0

        if final_signal == "BUY":
            if unanimous:
                # All 4 sources say BUY: discount because system is biased toward BUY
                buy_correction += BUY_UNANIMITY_DISCOUNT
            if tiers_disagree:
                # Tiers disagree and BUY won by weight: penalise for ignoring the skeptic
                buy_correction += DISAGREEMENT_PENALTY * 0.5
        elif final_signal == "SELL" and tiers_disagree:
            # SELL won despite disagreement — mildly reward contrarian conviction
            buy_correction -= DISAGREEMENT_PENALTY * 0.25

        confidence = max(0.0, min(1.0, round(consensus_level - buy_correction, 4)))

        recommendation = self._build_recommendation(final_signal, confidence, unanimous)

        # Build display weights (show effective tier weights, not per-source)
        display_weights = {
            "investment_judge": INVESTMENT_JUDGE_WEIGHT,
            "execution_block": EXECUTION_BLOCK_WEIGHT,
        }

        log.info(
            "Consensus: signal=%s confidence=%.2f unanimous=%s recommendation=%s | "
            "ij=%s exec=%s tiers_agree=%s buy_correction=%.2f | tally=%s",
            final_signal, confidence, unanimous, recommendation,
            ij_signal, exec_signal, not tiers_disagree, buy_correction,
            tally,
        )

        return ConsensusResult(
            final_signal=final_signal,
            confidence=confidence,
            source_signals=dict(source_signals),
            weights=display_weights,
            unanimous=unanimous,
            recommendation=recommendation,
            extractions=source_extractions or {},
        )

    @staticmethod
    def _build_recommendation(signal: str, confidence: float, unanimous: bool) -> str:
        """Map signal + confidence to a human-readable recommendation.

        Thresholds (adjusted for anti-BUY-bias):
        * confidence >= 0.85 AND unanimous → STRONG_<signal>
        * confidence >= 0.6 → <signal> as-is
        * BUY with confidence 0.5-0.6 → CAUTIOUS_BUY (explicit low-conviction label)
        * confidence < 0.5 → HOLD (too uncertain)
        """
        if confidence >= CONFIDENCE_STRONG_THRESHOLD and unanimous:
            return f"STRONG_{signal}"

        if confidence >= CONFIDENCE_TRADE_THRESHOLD:
            return signal

        if signal == "BUY" and confidence >= 0.45:
            return "CAUTIOUS_BUY"

        return "HOLD"


# ---------------------------------------------------------------------------
# Top-level Engine
# ---------------------------------------------------------------------------

class ConsensusEngine:
    """High-level facade that combines extraction, scoring, and tracking.

    Typical usage::

        engine = ConsensusEngine()
        result = engine.evaluate(log_states_dict)
        print(result.recommendation)
    """

    def __init__(self, db_path: Optional[str] = None):
        self.extractor = ConsensusSignalExtractor()
        self.tracker = AccuracyTracker(db_path=db_path)
        self.scorer = ConfidenceScorer(self.tracker)

    def evaluate(
        self,
        log_states_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, ConsensusResult]:
        """Run consensus evaluation over all dates in *log_states_dict*.

        Args:
            log_states_dict: The dict produced by
                :meth:`TradingAgentsGraph._log_state`, mapping
                ``str(trade_date)`` → state dict.

        Returns:
            Mapping of ``date_str`` → :class:`ConsensusResult`.
            Each result's extractions are populated with full detail.
        """
        results: Dict[str, ConsensusResult] = {}

        for date_str, state in log_states_dict.items():
            try:
                extractions = self.extractor.extract_from_state(state)
                source_signals = {src: ext.signal for src, ext in extractions.items()}

                result = self.scorer.score(source_signals, extractions)
                results[date_str] = result

                # Persist predictions for future accuracy tracking
                ticker = state.get("company_of_interest", "UNKNOWN")
                self._record_predictions(ticker, date_str, source_signals)

            except Exception as exc:
                log.error(
                    "Consensus evaluation failed for date %s: %s",
                    date_str, exc,
                    exc_info=True,
                )
                # Produce a safe fallback result
                results[date_str] = ConsensusResult(
                    final_signal="HOLD",
                    confidence=0.0,
                    source_signals={},
                    weights={},
                    unanimous=False,
                    recommendation="HOLD",
                )

        return results

    def _record_predictions(
        self,
        ticker: str,
        date_str: str,
        source_signals: Dict[str, str],
    ) -> None:
        """Persist predictions to the accuracy tracker."""
        for source, signal in source_signals.items():
            try:
                self.tracker.record_prediction(ticker, date_str, source, signal)
            except Exception as exc:
                log.error("Failed to record prediction (%s/%s/%s): %s", ticker, date_str, source, exc)

    def record_outcome(self, ticker: str, date_str: str, actual_signal: str) -> int:
        """Record the actual outcome for a ticker/date, updating accuracy.

        This should be called after the market move is known (e.g. next-day
        close vs prior close, or whatever definition of "actual" you use).

        Returns:
            Number of prediction rows updated.
        """
        return self.tracker.record_outcome(ticker, date_str, actual_signal)

    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Convenience wrapper around :class:`AccuracyTracker`."""
        return self.tracker.get_source_stats()

    def get_ticker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Convenience wrapper for per-ticker accuracy."""
        return self.tracker.get_ticker_stats()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.tracker.close()
