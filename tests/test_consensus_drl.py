"""Tests for DRL-weighted consensus scoring.

Tests the Q-learning weight adjustment, alpha-blending, decay, and
integration with ConsensusEngine.
"""

import math
import os
import sqlite3
import tempfile

import pytest

from tradingagents.graph.consensus import (
    DRLWeightedScorer,
    AccuracyTracker,
    ConsensusEngine,
    ConsensusResult,
    ExtractionResult,
    _discretize_regime,
    _discretize_streak,
    DRL_ALPHA,
    DRL_LEARNING_RATE,
    DRL_DECAY_LAMBDA,
    DRL_MAX_WEIGHT_ADJUST,
    MIN_WEIGHT,
    SOURCES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Return a temporary SQLite DB path for each test."""
    return str(tmp_path / "test_drl.db")


@pytest.fixture
def tracker(db_path):
    """AccuracyTracker with a fresh temp DB."""
    t = AccuracyTracker(db_path=db_path)
    yield t
    t.close()


@pytest.fixture
def drl_scorer(tracker):
    """DRLWeightedScorer with default alpha."""
    s = DRLWeightedScorer(tracker)
    yield s


@pytest.fixture
def engine(db_path):
    """Full ConsensusEngine with DRL enabled."""
    e = ConsensusEngine(db_path=db_path, drl_alpha=DRL_ALPHA)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Discretization helpers
# ---------------------------------------------------------------------------

class TestDiscretize:
    def test_regime_strong_decline(self):
        assert _discretize_regime(-0.10) == "strong_decline"
        # -0.05 is exactly at the threshold (exclusive), so it's "decline"
        assert _discretize_regime(-0.051) == "strong_decline"
        assert _discretize_regime(-0.05) == "decline"

    def test_regime_decline(self):
        assert _discretize_regime(-0.03) == "decline"

    def test_regime_neutral(self):
        assert _discretize_regime(0.0) == "neutral"
        assert _discretize_regime(0.01) == "neutral"

    def test_regime_strong_growth(self):
        assert _discretize_regime(0.05) == "strong_growth"
        assert _discretize_regime(0.10) == "strong_growth"

    def test_streak_clamp(self):
        assert _discretize_streak(-1) == 0
        assert _discretize_streak(10) == 4  # clamped to 4-5 bucket

    def test_streak_values(self):
        assert _discretize_streak(0) == 0
        assert _discretize_streak(1) == 1
        assert _discretize_streak(2) == 2
        assert _discretize_streak(3) == 3
        assert _discretize_streak(4) == 4
        assert _discretize_streak(5) == 4


# ---------------------------------------------------------------------------
# DRLWeightedScorer initialization
# ---------------------------------------------------------------------------

class TestDRLInit:
    def test_default_params(self, drl_scorer):
        assert drl_scorer.drl_alpha == DRL_ALPHA
        assert drl_scorer.learning_rate == DRL_LEARNING_RATE

    def test_custom_alpha(self, tracker):
        s = DRLWeightedScorer(tracker, drl_alpha=0.5)
        assert s.drl_alpha == 0.5

    def test_tables_created(self, tracker):
        """DRL tables should be created during AccuracyTracker init."""
        conn = tracker._get_conn()
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "drl_rewards" in tables
        assert "drl_qtable" in tables


# ---------------------------------------------------------------------------
# Alpha-blended weights
# ---------------------------------------------------------------------------

class TestBlendedWeights:
    def test_weights_sum_to_one(self, drl_scorer):
        """Blended weights must always sum to 1.0 (within float precision)."""
        signals = {src: "BUY" for src in SOURCES}
        for regime in [-0.10, -0.03, 0.0, 0.05]:
            weights = drl_scorer.get_blended_weights(signals, regime_return=regime)
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-4, f"sum={total} for regime={regime}"

    def test_weights_no_source_missing(self, drl_scorer):
        signals = {src: "HOLD" for src in SOURCES}
        weights = drl_scorer.get_blended_weights(signals, regime_return=0.0)
        for src in SOURCES:
            assert src in weights
            assert weights[src] >= MIN_WEIGHT

    def test_all_sources_have_weights(self, drl_scorer):
        signals = {"investment_judge": "BUY", "trader": "SELL",
                    "risk_judge": "HOLD", "portfolio_manager": "BUY"}
        weights = drl_scorer.get_blended_weights(signals, regime_return=0.01)
        assert len(weights) == len(SOURCES)

    def test_fallback_to_equal_weights(self, drl_scorer):
        """With no DRL history, blended weights should be close to accuracy weights."""
        signals = {src: "BUY" for src in SOURCES}
        # With no history, accuracy weights are equal (0.25 each)
        weights = drl_scorer.get_blended_weights(signals, regime_return=0.0)
        # No Q-table entries exist, so DRL adjustments are 0, drl_weights = equal
        # Blended = (1-0.3)*0.25 + 0.3*0.25 = 0.25 for each
        for src in SOURCES:
            assert abs(weights[src] - 0.25) < 1e-4, f"{src}: {weights[src]}"

    def test_alpha_1_pure_drl(self, tracker):
        """alpha=1.0 should use pure DRL weights (equal without history)."""
        s = DRLWeightedScorer(tracker, drl_alpha=1.0)
        signals = {src: "BUY" for src in SOURCES}
        weights = s.get_blended_weights(signals, regime_return=0.0)
        for src in SOURCES:
            assert abs(weights[src] - 0.25) < 1e-4

    def test_alpha_0_pure_accuracy(self, tracker):
        """alpha=0.0 should use pure accuracy weights."""
        s = DRLWeightedScorer(tracker, drl_alpha=0.0)
        signals = {src: "BUY" for src in SOURCES}
        weights = s.get_blended_weights(signals, regime_return=0.0)
        # Pure accuracy weights = equal weights (no history)
        for src in SOURCES:
            assert abs(weights[src] - 0.25) < 1e-4


# ---------------------------------------------------------------------------
# Score integration
# ---------------------------------------------------------------------------

class TestScoreIntegration:
    def test_score_returns_consensus_result(self, drl_scorer):
        signals = {src: "BUY" for src in SOURCES}
        result = drl_scorer.score(signals, regime_return=0.0)
        assert isinstance(result, ConsensusResult)
        assert result.final_signal in {"BUY", "HOLD", "SELL"}
        assert 0.0 <= result.confidence <= 1.0

    def test_score_weights_populated(self, drl_scorer):
        signals = {src: "BUY" for src in SOURCES}
        result = drl_scorer.score(signals, regime_return=0.0)
        assert len(result.weights) == len(SOURCES)

    def test_score_with_mixed_signals(self, drl_scorer):
        signals = {
            "investment_judge": "BUY",
            "trader": "SELL",
            "risk_judge": "HOLD",
            "portfolio_manager": "HOLD",
        }
        result = drl_scorer.score(signals, regime_return=0.01)
        assert isinstance(result, ConsensusResult)


# ---------------------------------------------------------------------------
# Reward recording
# ---------------------------------------------------------------------------

class TestRewardRecording:
    def test_record_correct_reward(self, drl_scorer):
        signals = {src: "BUY" for src in SOURCES}
        drl_scorer.record_reward(
            ticker="AAPL", date_str="2026-06-10",
            source_signals=signals,
            predicted_signal="BUY", actual_signal="BUY",
            regime_return=0.01,
        )
        history = drl_scorer.get_reward_history(limit=10)
        assert len(history) == len(SOURCES)  # One reward per source
        assert all(r["reward"] == 1.0 for r in history)

    def test_record_incorrect_reward(self, drl_scorer):
        signals = {src: "BUY" for src in SOURCES}
        drl_scorer.record_reward(
            ticker="AAPL", date_str="2026-06-10",
            source_signals=signals,
            predicted_signal="BUY", actual_signal="SELL",
            regime_return=0.01,
        )
        history = drl_scorer.get_reward_history(limit=10)
        assert all(r["reward"] == -1.0 for r in history)

    def test_record_hold_reward(self, drl_scorer):
        signals = {src: "BUY" for src in SOURCES}
        drl_scorer.record_reward(
            ticker="AAPL", date_str="2026-06-10",
            source_signals=signals,
            predicted_signal="BUY", actual_signal="HOLD",
            regime_return=0.01,
        )
        history = drl_scorer.get_reward_history(limit=10)
        assert all(r["reward"] == 0.0 for r in history)

    def test_reward_updates_qtable(self, drl_scorer):
        signals = {src: "BUY" for src in SOURCES}
        drl_scorer.record_reward(
            ticker="AAPL", date_str="2026-06-10",
            source_signals=signals,
            predicted_signal="BUY", actual_signal="BUY",
            regime_return=0.01,
        )
        conn = drl_scorer._tracker._get_conn()
        rows = conn.execute("SELECT * FROM drl_qtable").fetchall()
        assert len(rows) == len(SOURCES)
        # Q-values should be positive (reward = +1)
        for row in rows:
            assert row["q_value"] > 0


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_decay_reduces_qvalues(self, drl_scorer):
        """After recording positive rewards, decay should shrink Q-values."""
        signals = {src: "BUY" for src in SOURCES}
        # Record several positive rewards to build up Q-values
        for i in range(5):
            drl_scorer.record_reward(
                ticker="AAPL", date_str=f"2026-06-{10+i:02d}",
                source_signals=signals,
                predicted_signal="BUY", actual_signal="BUY",
                regime_return=0.01,
            )

        # Get Q-values before decay
        conn = drl_scorer._tracker._get_conn()
        before = {row["source"]: row["q_value"]
                  for row in conn.execute("SELECT source, q_value FROM drl_qtable").fetchall()}

        # Apply decay
        count = drl_scorer.decay_old_rewards()
        assert count > 0

        # Get Q-values after decay
        after = {row["source"]: row["q_value"]
                 for row in conn.execute("SELECT source, q_value FROM drl_qtable").fetchall()}

        # All should be smaller
        for src in SOURCES:
            assert after[src] < before[src], f"{src}: {after[src]} >= {before[src]}"

    def test_decay_factor_value(self, drl_scorer):
        """Verify the decay factor is exp(-lambda)."""
        expected = math.exp(-DRL_DECAY_LAMBDA)
        assert expected < 1.0  # Must actually decay
        assert expected > 0.9   # Should be mild decay per tick


# ---------------------------------------------------------------------------
# ConsensusEngine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_engine_uses_drl_by_default(self, engine):
        """Engine should use DRL scorer by default."""
        assert engine.drl_scorer is not None
        assert isinstance(engine.drl_scorer, DRLWeightedScorer)

    def test_evaluate_with_drl(self, engine):
        log_states = {
            "2026-06-10": {
                "company_of_interest": "AAPL",
                "investment_debate_state": {"judge_decision": "Rating: BUY"},
                "trader_investment_decision": "FINAL TRANSACTION PROPOSAL: **BUY**",
                "risk_debate_state": {"judge_decision": "Decision: HOLD"},
                "final_trade_decision": "Rating: BUY",
            },
        }
        results = engine.evaluate(log_states, use_drl=True)
        assert "2026-06-10" in results
        result = results["2026-06-10"]
        assert isinstance(result, ConsensusResult)
        assert len(result.weights) == len(SOURCES)

    def test_evaluate_without_drl(self, engine):
        """use_drl=False should use original ConfidenceScorer (tier weights)."""
        log_states = {
            "2026-06-10": {
                "company_of_interest": "AAPL",
                "investment_debate_state": {"judge_decision": "Rating: BUY"},
                "trader_investment_decision": "FINAL TRANSACTION PROPOSAL: **BUY**",
                "risk_debate_state": {"judge_decision": "Decision: BUY"},
                "final_trade_decision": "Rating: BUY",
            },
        }
        results = engine.evaluate(log_states, use_drl=False)
        result = results["2026-06-10"]
        # Non-DRL scorer uses tier display weights, not per-source
        assert "investment_judge" in result.weights or "execution_block" in result.weights

    def test_evaluate_predictions_persisted(self, engine, db_path):
        log_states = {
            "2026-06-10": {
                "company_of_interest": "MSFT",
                "investment_debate_state": {"judge_decision": "Rating: SELL"},
                "trader_investment_decision": "FINAL TRANSACTION PROPOSAL: **HOLD**",
                "risk_debate_state": {"judge_decision": "Decision: HOLD"},
                "final_trade_decision": "Rating: HOLD",
            },
        }
        engine.evaluate(log_states, use_drl=True)
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT * FROM predictions WHERE ticker='MSFT' AND date='2026-06-10'"
        ).fetchall()
        assert len(rows) == len(SOURCES)
        conn.close()

    def test_record_drl_reward_via_engine(self, engine):
        log_states = {
            "2026-06-10": {
                "company_of_interest": "GOOG",
                "investment_debate_state": {"judge_decision": "Rating: BUY"},
                "trader_investment_decision": "FINAL TRANSACTION PROPOSAL: **BUY**",
                "risk_debate_state": {"judge_decision": "Decision: BUY"},
                "final_trade_decision": "Rating: BUY",
            },
        }
        engine.evaluate(log_states, use_drl=True)
        engine.record_drl_reward(
            ticker="GOOG", date_str="2026-06-10",
            predicted_signal="BUY", actual_signal="BUY",
            regime_return=0.02,
        )
        history = engine.get_drl_reward_history(limit=10)
        assert len(history) == len(SOURCES)
        assert all(r["reward"] == 1.0 for r in history)

    def test_get_drl_reward_history_empty(self, engine):
        assert engine.get_drl_reward_history() == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_signals(self, drl_scorer):
        """Empty signal dict should not crash."""
        weights = drl_scorer.get_blended_weights({}, regime_return=0.0)
        assert len(weights) == len(SOURCES)

    def test_malformed_signals_tolerates_valid(self, drl_scorer):
        """Valid BUY/HOLD/SELL values with disagreement should not crash."""
        signals = {"investment_judge": "SELL", "trader": "HOLD",
                     "risk_judge": "HOLD", "portfolio_manager": "BUY"}
        result = drl_scorer.score(signals, regime_return=0.0)
        assert isinstance(result, ConsensusResult)
        assert result.recommendation != ""

    def test_extreme_regime(self, drl_scorer):
        """Very negative regime should produce valid weights."""
        signals = {src: "SELL" for src in SOURCES}
        weights = drl_scorer.get_blended_weights(signals, regime_return=-0.50)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_reset_qtable(self, drl_scorer):
        signals = {src: "BUY" for src in SOURCES}
        drl_scorer.record_reward(
            ticker="AAPL", date_str="2026-06-10",
            source_signals=signals,
            predicted_signal="BUY", actual_signal="BUY",
            regime_return=0.01,
        )
        drl_scorer.reset_qtable()
        conn = drl_scorer._tracker._get_conn()
        rows = conn.execute("SELECT q_value FROM drl_qtable").fetchall()
        assert all(r["q_value"] == 0.0 for r in rows)


# ---------------------------------------------------------------------------
# Forward-Gate Tests
# ---------------------------------------------------------------------------

class TestForwardGate:
    """Tests for forward-gated weight deployment."""

    def test_forward_gate_passes_with_no_history(self, drl_scorer):
        """With < FORWARD_GATE_MIN_PREDICTIONS graded predictions, gate is bypassed (blended deployed)."""
        signals = {src: "BUY" for src in SOURCES}
        weights = drl_scorer.get_blended_weights(signals, regime_return=0.01)
        # With no DRL history, blended == accuracy weights (equal), gate should pass
        assert abs(sum(weights.values()) - 1.0) < 1e-4
        assert all(w >= MIN_WEIGHT for w in weights.values())

    def test_forward_gate_blocks_degenerate_drl(self, tracker):
        """When DRL weights consistently pick wrong, gate should fall back to accuracy weights.

        Setup: risk_judge is the only correct source (SELL), but DRL Q-table
        boosts investment_judge (always wrong, says BUY).  With alpha=1.0
        (pure DRL), the boosted wrong source flips consensus to BUY.
        The gate must detect this and deploy accuracy-only weights instead.
        """
        # Use alpha=1.0 so DRL weights dominate the blend
        scorer = DRLWeightedScorer(tracker, drl_alpha=1.0)

        # Record 24 graded predictions (4 sources x 6 rounds), enough for FORWARD_GATE_MIN_PREDICTIONS
        # investment_judge and trader always say BUY (wrong), risk_judge says SELL (right)
        for i in range(6):
            date = f"2026-06-{10+i:02d}"
            tracker.record_prediction("TEST", date, "investment_judge", "BUY")
            tracker.record_prediction("TEST", date, "trader", "BUY")
            tracker.record_prediction("TEST", date, "risk_judge", "SELL")
            tracker.record_prediction("TEST", date, "portfolio_manager", "HOLD")
            tracker.record_outcome("TEST", date, "SELL")

        # Inject adversarial Q-values: boost the wrong source, suppress the right one
        conn = tracker._get_conn()
        conn.execute(
            "INSERT INTO drl_qtable (regime_bucket, source, streak_bucket, q_value) "
            "VALUES ('neutral', 'investment_judge', 0, 1.0)"
        )
        conn.execute(
            "INSERT INTO drl_qtable (regime_bucket, source, streak_bucket, q_value) "
            "VALUES ('neutral', 'risk_judge', 0, -1.0)"
        )
        conn.commit()

        signals = {"investment_judge": "BUY", "trader": "BUY",
                    "risk_judge": "SELL", "portfolio_manager": "HOLD"}
        weights = scorer.get_blended_weights(signals, regime_return=0.0)

        # Gate should have blocked the degenerate DRL weights
        # Verify by comparing to accuracy weights (what the gate falls back to)
        acc_weights = tracker.get_weights()
        for src in SOURCES:
            assert abs(weights[src] - acc_weights[src]) < 1e-3, \
                f"Forward-gate should have blocked DRL for {src}: got {weights[src]}, expected {acc_weights[src]}"

    def test_forward_gate_min_predictions_threshold(self, drl_scorer):
        """Gate should not activate with fewer than FORWARD_GATE_MIN_PREDICTIONS."""
        from tradingagents.graph.consensus import FORWARD_GATE_MIN_PREDICTIONS
        assert FORWARD_GATE_MIN_PREDICTIONS >= 10

    def test_forward_gate_returns_valid_weights_dict(self, drl_scorer):
        """Forward-gate output must always be a valid normalized weight dict."""
        signals = {src: "BUY" for src in SOURCES}
        for _ in range(3):
            weights = drl_scorer.get_blended_weights(signals, regime_return=0.02)
            assert set(weights.keys()) == set(SOURCES)
            assert abs(sum(weights.values()) - 1.0) < 1e-3
            assert all(0 < w < 1.0 for w in weights.values())
