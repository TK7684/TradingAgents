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
import math
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

#: Additional BUY penalty when market regime is declining (SPY down >2% in 5 days).
REGIME_DECLINE_BUY_PENALTY: float = 0.15

#: Threshold for "declining market" regime (SPY 5-day return below this = declining).
REGIME_DECLINE_THRESHOLD: float = -0.02

#: Threshold for "strong decline" (extra penalty).
REGIME_STRONG_DECLINE_THRESHOLD: float = -0.05
REGIME_STRONG_DECLINE_PENALTY: float = 0.25

#: Combined weight of the 3 execution sources (trader, risk_judge, PM) vs investment_judge.
EXECUTION_BLOCK_WEIGHT: float = 0.40
INVESTMENT_JUDGE_WEIGHT: float = 0.60

# DRL (Q-learning) constants
# DRL_ALPHA: Blending factor: final_weight = (1-alpha)*accuracy_weight + alpha*drl_weight
# DRL_LEARNING_RATE: How fast Q-values update on reward
# DRL_DISCOUNT_FACTOR: Temporal discount for future rewards
# DRL_DECAY_LAMBDA: Exponential decay lambda for old reward history (half-life ~10 entries)
# DRL_MAX_WEIGHT_ADJUST: Clamp per-source adjustment to +/-this value
DRL_ALPHA: float = 0.3
DRL_LEARNING_RATE: float = 0.15
DRL_DISCOUNT_FACTOR: float = 0.9
DRL_DECAY_LAMBDA: float = 0.07
DRL_MAX_WEIGHT_ADJUST: float = 0.15

# ---------------------------------------------------------------------------
# Regex patterns (mirrors signal_processing.py)

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


def get_market_regime() -> float:
    """Return SPY 5-day return as a regime indicator.

    Uses yfinance to fetch SPY close from 10 days ago and 5 days ago.
    Returns the 5-day return fraction (e.g. -0.03 for -3% decline).
    Returns 0.0 on any error (neutral regime).
    """
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        hist = spy.history(period="10d", auto_adjust=True)
        if hist is None or len(hist) < 6:
            return 0.0
        close_5d_ago = float(hist["Close"].iloc[-6])
        close_now = float(hist["Close"].iloc[-1])
        return (close_now - close_5d_ago) / close_5d_ago
    except Exception:
        return 0.0

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

    _DDL_DRL_REWARDS = """
        CREATE TABLE IF NOT EXISTS drl_rewards (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            date            TEXT    NOT NULL,
            source          TEXT    NOT NULL,
            regime_bucket   TEXT    NOT NULL DEFAULT 'neutral',
            streak_correct  INTEGER NOT NULL DEFAULT 0,
            predicted_signal TEXT NOT NULL,
            actual_signal   TEXT,
            reward          REAL NOT NULL DEFAULT 0.0,
            weight_adjust   REAL NOT NULL DEFAULT 0.0,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );
    """

    _DDL_DRL_QTABLE = """
        CREATE TABLE IF NOT EXISTS drl_qtable (
            regime_bucket   TEXT    NOT NULL,
            source          TEXT    NOT NULL,
            streak_bucket   INTEGER NOT NULL,
            q_value         REAL    NOT NULL DEFAULT 0.0,
            PRIMARY KEY (regime_bucket, source, streak_bucket)
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
        conn.executescript(
            self._DDL_PREDICTIONS + self._DDL_STATS + self._DDL_DRL_REWARDS + self._DDL_DRL_QTABLE
        )
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

        # --- Market regime correction ---
        regime_ret = get_market_regime()
        regime_correction = 0.0
        if final_signal == "BUY" and regime_ret < REGIME_DECLINE_THRESHOLD:
            if regime_ret < REGIME_STRONG_DECLINE_THRESHOLD:
                regime_correction = REGIME_STRONG_DECLINE_PENALTY
            else:
                regime_correction = REGIME_DECLINE_BUY_PENALTY
            log.info(
                "Regime correction: SPY 5d return=%.2f%%, BUY penalty=%.2f",
                regime_ret * 100, regime_correction,
            )

        confidence = max(0.0, min(1.0, round(consensus_level - buy_correction - regime_correction, 4)))

        recommendation = self._build_recommendation(final_signal, confidence, unanimous)

        # Build display weights (show effective tier weights, not per-source)
        display_weights = {
            "investment_judge": INVESTMENT_JUDGE_WEIGHT,
            "execution_block": EXECUTION_BLOCK_WEIGHT,
        }

        log.info(
            "Consensus: signal=%s confidence=%.2f unanimous=%s recommendation=%s | "
            "ij=%s exec=%s tiers_agree=%s buy_correction=%.2f regime=%.2f%% | tally=%s",
            final_signal, confidence, unanimous, recommendation,
            ij_signal, exec_signal, not tiers_disagree, buy_correction,
            regime_ret * 100, tally,
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
# DRL-Weighted Scorer (Q-learning style)
# ---------------------------------------------------------------------------

# Regime bucket boundaries for discretizing SPY 5-day return.
_REGIME_BUCKETS: list = [
    ("strong_decline", -1.0, REGIME_STRONG_DECLINE_THRESHOLD),
    ("decline", REGIME_STRONG_DECLINE_THRESHOLD, REGIME_DECLINE_THRESHOLD),
    ("neutral", REGIME_DECLINE_THRESHOLD, REGIME_DECLINE_THRESHOLD),  # special: catch zero
    ("moderate_growth", REGIME_DECLINE_THRESHOLD, 0.03),
    ("strong_growth", 0.03, 1.0),
]

# Streak bucket boundaries for discretizing last-5 correctness count.
_STREAK_BUCKETS: list = [
    (0, 0),    # 0/5 correct
    (1, 1),    # 1/5 correct
    (2, 2),    # 2/5 correct
    (3, 3),    # 3/5 correct
    (4, 5),    # 4-5/5 correct
]


def _discretize_regime(spy_return: float) -> str:
    """Map a continuous SPY return to a discrete regime bucket string."""
    if spy_return < REGIME_STRONG_DECLINE_THRESHOLD:
        return "strong_decline"
    if spy_return < REGIME_DECLINE_THRESHOLD:
        return "decline"
    if spy_return < 0.03:
        return "neutral"
    return "strong_growth"


def _discretize_streak(correct_count: int) -> int:
    """Map a correctness count (0-5) to a streak bucket index."""
    correct_count = max(0, min(5, correct_count))
    for lo, hi in _STREAK_BUCKETS:
        if lo <= correct_count <= hi:
            return lo
    return 0


class DRLWeightedScorer:
    """Wrap :class:`ConfidenceScorer` with Q-learning based weight adjustment.

    The scorer maintains a Q-table in SQLite keyed by
    ``(regime_bucket, source, streak_bucket)`` and uses reward feedback
    from consensus outcomes to learn per-source weight adjustments.

    Final weights are an alpha-blend of accuracy-based and DRL-based weights::

        final_weight = (1 - alpha) * accuracy_weight + alpha * drl_weight

    This keeps the system safe: when there is no DRL history, it falls back
    to pure accuracy weighting.  As reward history accumulates, the DRL
    component gradually shifts weights toward what has worked recently in
    the current market regime.
    """

    def __init__(
        self,
        accuracy_tracker: AccuracyTracker,
        drl_alpha: float = DRL_ALPHA,
        learning_rate: float = DRL_LEARNING_RATE,
        discount_factor: float = DRL_DISCOUNT_FACTOR,
        decay_lambda: float = DRL_DECAY_LAMBDA,
        max_weight_adjust: float = DRL_MAX_WEIGHT_ADJUST,
    ):
        self._tracker = accuracy_tracker
        self._base_scorer = ConfidenceScorer(accuracy_tracker)
        self.drl_alpha = drl_alpha
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.decay_lambda = decay_lambda
        self.max_weight_adjust = max_weight_adjust

    # -- public interface ------------------------------------------------------

    def score(
        self,
        source_signals: Dict[str, str],
        source_extractions: Optional[Dict[str, ExtractionResult]] = None,
        regime_return: Optional[float] = None,
    ) -> ConsensusResult:
        """Score signals, blending DRL-adjusted weights into the result.

        If *regime_return* is None, calls :func:`get_market_regime` to fetch
        the SPY 5-day return.  Pass an explicit value in tests or when
        mocking market data.

        Returns:
            A :class:`ConsensusResult` whose ``weights`` dict contains the
            DRL-blended per-source weights.
        """
        # Run the base scorer first (unchanged logic)
        result = self._base_scorer.score(source_signals, source_extractions)

        # Compute DRL-blended weights
        drl_weights = self.get_blended_weights(
            source_signals=source_signals,
            regime_return=regime_return,
        )

        result.weights = drl_weights
        return result

    def get_blended_weights(
        self,
        source_signals: Dict[str, str],
        regime_return: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute the alpha-blended per-source weights.

        Args:
            source_signals: Current source signals (used to determine which
                sources voted with the consensus).
            regime_return: SPY 5-day return.  Fetched live if None.

        Returns:
            Dict mapping each source name to its blended weight.
        """
        # Get accuracy-based weights
        accuracy_weights = self._tracker.get_weights()

        # Get DRL weight adjustments
        if regime_return is None:
            regime_return = get_market_regime()
        regime_bucket = _discretize_regime(regime_return)

        drl_adjustments = self._get_drl_weight_adjustments(regime_bucket, source_signals)

        # Convert adjustments into raw DRL weights:
        # Start from equal, apply adjustments, then normalise.
        equal_weight = 1.0 / len(SOURCES)
        raw_drl: Dict[str, float] = {}
        for src in SOURCES:
            raw_drl[src] = max(
                MIN_WEIGHT,
                equal_weight + drl_adjustments.get(src, 0.0),
            )

        # Normalise DRL weights to sum to 1
        drl_total = sum(raw_drl.values())
        if drl_total > 0:
            drl_weights = {src: w / drl_total for src, w in raw_drl.items()}
        else:
            drl_weights = {src: equal_weight for src in SOURCES}

        # Alpha blend
        blended: Dict[str, float] = {}
        for src in SOURCES:
            aw = accuracy_weights.get(src, equal_weight)
            dw = drl_weights.get(src, equal_weight)
            blended[src] = round(
                (1.0 - self.drl_alpha) * aw + self.drl_alpha * dw, 4
            )

        return blended

    def record_reward(
        self,
        ticker: str,
        date_str: str,
        source_signals: Dict[str, str],
        predicted_signal: str,
        actual_signal: str,
        regime_return: float = 0.0,
    ) -> None:
        """Record a reward for the DRL Q-table and update weights.

        Reward: +1 for correct consensus, -1 for incorrect, 0 for HOLD outcomes.

        Updates the ``drl_rewards`` table and then performs a Q-learning
        update on the ``drl_qtable``.
        """
        predicted = predicted_signal.upper()
        actual = actual_signal.upper()

        # Compute reward
        if predicted == actual:
            reward = 1.0
        elif predicted == "HOLD" or actual == "HOLD":
            reward = 0.0
        else:
            reward = -1.0

        regime_bucket = _discretize_regime(regime_return)

        conn = self._tracker._get_conn()

        # Get recent streak for each source (last 5 predictions)
        for source in SOURCES:
            streak = self._get_source_recent_correct(source, limit=5)
            streak_bucket = _discretize_streak(streak)

            # Fetch current Q-value
            row = conn.execute(
                "SELECT q_value FROM drl_qtable "
                "WHERE regime_bucket = ? AND source = ? AND streak_bucket = ?",
                (regime_bucket, source, streak_bucket),
            ).fetchone()

            old_q = row["q_value"] if row else 0.0

            # Q-learning (TD) update: Q(s,a) <- Q(s,a) + lr * (r + gamma * max_Q(s',a') - Q(s,a))
            # Bootstrap the next-state value as the best Q across streak
            # buckets for the same (regime, source) -- a proxy for the best
            # attainable future value of this source in the current regime.
            next_max_row = conn.execute(
                "SELECT MAX(q_value) FROM drl_qtable "
                "WHERE regime_bucket = ? AND source = ?",
                (regime_bucket, source),
            ).fetchone()
            max_next_q = (
                next_max_row[0]
                if next_max_row and next_max_row[0] is not None
                else 0.0
            )
            new_q = old_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - old_q
            )

            # Compute weight adjustment from Q-value, clamped
            weight_adjust = max(
                -self.max_weight_adjust,
                min(self.max_weight_adjust, new_q * 0.05),
            )

            # Upsert Q-table
            conn.execute(
                "INSERT INTO drl_qtable (regime_bucket, source, streak_bucket, q_value) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(regime_bucket, source, streak_bucket) "
                "DO UPDATE SET q_value = excluded.q_value",
                (regime_bucket, source, streak_bucket, new_q),
            )

            # Record reward history
            conn.execute(
                "INSERT INTO drl_rewards "
                "(ticker, date, source, regime_bucket, streak_correct, "
                "predicted_signal, actual_signal, reward, weight_adjust) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ticker, date_str, source, regime_bucket, streak,
                    predicted, actual, reward, weight_adjust,
                ),
            )

        conn.commit()
        log.debug(
            "DRL reward recorded: ticker=%s date=%s reward=%.1f regime=%s",
            ticker, date_str, reward, regime_bucket,
        )

    def get_reward_history(self, limit: int = 50) -> list:
        """Return the most recent DRL reward records."""
        conn = self._tracker._get_conn()
        rows = conn.execute(
            "SELECT * FROM drl_rewards ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def decay_old_rewards(self) -> int:
        """Apply exponential decay to all DRL Q-values.

        Reduces each Q-value by ``exp(-decay_lambda)`` to make older
        learned values less influential.  Called periodically to prevent
        stale Q-values from dominating.

        Returns:
            Number of Q-table rows updated.
        """
        conn = self._tracker._get_conn()
        decay_factor = math.exp(-self.decay_lambda)
        cursor = conn.execute(
            "UPDATE drl_qtable SET q_value = q_value * ? WHERE q_value != 0",
            (decay_factor,),
        )
        conn.commit()
        return cursor.rowcount

    def reset_qtable(self) -> None:
        """Reset all Q-values to 0 (for testing)."""
        conn = self._tracker._get_conn()
        conn.execute("UPDATE drl_qtable SET q_value = 0.0")
        conn.commit()

    # -- private helpers ------------------------------------------------------

    def _get_drl_weight_adjustments(
        self,
        regime_bucket: str,
        source_signals: Dict[str, str],
    ) -> Dict[str, float]:
        """Look up the Q-table for current state and return weight adjustments."""
        conn = self._tracker._get_conn()
        adjustments: Dict[str, float] = {}

        for source in SOURCES:
            streak = self._get_source_recent_correct(source, limit=5)
            streak_bucket = _discretize_streak(streak)

            row = conn.execute(
                "SELECT q_value FROM drl_qtable "
                "WHERE regime_bucket = ? AND source = ? AND streak_bucket = ?",
                (regime_bucket, source, streak_bucket),
            ).fetchone()

            if row:
                # Convert Q-value to weight adjustment, clamped
                adjustments[source] = max(
                    -self.max_weight_adjust,
                    min(self.max_weight_adjust, row["q_value"] * 0.05),
                )
            else:
                adjustments[source] = 0.0

        return adjustments

    def _get_source_recent_correct(self, source: str, limit: int = 5) -> int:
        """Count how many of the last *limit* predictions for *source* were correct."""
        conn = self._tracker._get_conn()
        row = conn.execute(
            "SELECT SUM(correct) as cnt FROM ("
            "  SELECT correct FROM predictions "
            "  WHERE source = ? AND actual_signal IS NOT NULL "
            "  ORDER BY id DESC LIMIT ?"
            ")",
            (source, limit),
        ).fetchone()
        return int(row["cnt"] or 0)


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

    def __init__(self, db_path: Optional[str] = None, drl_alpha: float = DRL_ALPHA):
        self.extractor = ConsensusSignalExtractor()
        self.tracker = AccuracyTracker(db_path=db_path)
        self.scorer = ConfidenceScorer(self.tracker)
        self.drl_scorer = DRLWeightedScorer(self.tracker, drl_alpha=drl_alpha)

    def evaluate(
        self,
        log_states_dict: Dict[str, Dict[str, Any]],
        use_drl: bool = True,
    ) -> Dict[str, ConsensusResult]:
        """Run consensus evaluation over all dates in *log_states_dict*.

        Args:
            log_states_dict: The dict produced by
                :meth:`TradingAgentsGraph._log_state`, mapping
                ``str(trade_date)`` → state dict.
            use_drl: If True (default), use the DRL-weighted scorer which
                blends accuracy-based weights with Q-learned adjustments.
                Set to False to use the original scorer unchanged.

        Returns:
            Mapping of ``date_str`` → :class:`ConsensusResult`.
            Each result's extractions are populated with full detail.
        """
        results: Dict[str, ConsensusResult] = {}
        scorer = self.drl_scorer if use_drl else self.scorer

        for date_str, state in log_states_dict.items():
            try:
                extractions = self.extractor.extract_from_state(state)
                source_signals = {src: ext.signal for src, ext in extractions.items()}

                result = scorer.score(source_signals, extractions)
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

    def record_drl_reward(
        self,
        ticker: str,
        date_str: str,
        predicted_signal: str,
        actual_signal: str,
        regime_return: float = 0.0,
    ) -> None:
        """Record a DRL reward for a consensus outcome.

        Convenience wrapper around :meth:`DRLWeightedScorer.record_reward`.
        """
        # Reconstruct approximate source signals from the predictions table
        source_signals: Dict[str, str] = {}
        conn = self.tracker._get_conn()
        rows = conn.execute(
            "SELECT source, predicted_signal FROM predictions "
            "WHERE ticker = ? AND date = ?",
            (ticker, date_str),
        ).fetchall()
        for row in rows:
            source_signals[row["source"]] = row["predicted_signal"]

        if not source_signals:
            source_signals = {src: "HOLD" for src in SOURCES}

        self.drl_scorer.record_reward(
            ticker=ticker,
            date_str=date_str,
            source_signals=source_signals,
            predicted_signal=predicted_signal,
            actual_signal=actual_signal,
            regime_return=regime_return,
        )

    def get_drl_reward_history(self, limit: int = 50) -> list:
        """Return recent DRL reward records."""
        return self.drl_scorer.get_reward_history(limit=limit)

    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Convenience wrapper around :class:`AccuracyTracker`."""
        return self.tracker.get_source_stats()

    def get_ticker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Convenience wrapper for per-ticker accuracy."""
        return self.tracker.get_ticker_stats()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.tracker.close()
