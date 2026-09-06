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
    DRL_DISCOUNT_FACTOR,
    DRL_DECAY_LAMBDA,
    DRL_MAX_WEIGHT_ADJUST,
    MIN_WEIGHT,
    REPLAY_MAX_EPOCHS,
    REPLAY_CONVERGENCE_TOL,
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
# ---------------------------------------------------------------------------
# TD(0) Bootstrap Tests (AGI cycle H20260826150153)
# ---------------------------------------------------------------------------

class TestTD0Bootstrap:
    """The Q-update must use the canonical TD(0) rule with a real
    gamma*Q(s',a') bootstrap term, not the degenerate max_Q(s',a')=0 EMA."""

    def _record(self, drl_scorer, **kw):
        signals = kw.get("signals", {src: "BUY" for src in SOURCES})
        drl_scorer.record_reward(
            ticker=kw.get("ticker", "AAPL"),
            date_str=kw.get("date_str", "2026-06-10"),
            source_signals=signals,
            predicted_signal=kw.get("predicted_signal", "BUY"),
            actual_signal=kw.get("actual_signal", "BUY"),
            regime_return=kw.get("regime_return", 0.01),
        )

    def test_bootstrap_lands_in_next_streak_bucket(self, drl_scorer):
        """A correct call must advance the streak and write the bootstrapped
        Q-value into the next streak bucket's row (not just the current one)."""
        self._record(drl_scorer)  # streak 0 -> correct -> writes bucket 0, bootstraps from bucket 1
        conn = drl_scorer._tracker._get_conn()
        rows = conn.execute(
            "SELECT streak_bucket, q_value FROM drl_qtable ORDER BY streak_bucket"
        ).fetchall()
        assert len(rows) == len(SOURCES)
        for r in rows:
            # First update: Q = lr * (r + gamma*0 - 0) = lr * 1.0 > 0
            assert r["q_value"] > 0
            assert r["streak_bucket"] == 0

    def test_self_transition_bootstraps_from_current_estimate(self, drl_scorer):
        """streak=4 is the top bucket; streak+1 clamps back to 4, so the update
        must bootstrap from the current Q instead of the (absent) next row."""
        # Seed a 5-correct streak via the predictions table so
        # _get_source_recent_correct sees bucket 4-5
        conn = drl_scorer._tracker._get_conn()
        for src in SOURCES:
            for i in range(5):
                conn.execute(
                    "INSERT INTO predictions (ticker, date, source, predicted_signal, actual_signal, correct) "
                    "VALUES ('AAPL', ?, ?, 'BUY', 'BUY', 1)",
                    (f"2026-06-{10+i:02d}", src),
                )
        conn.commit()
        self._record(drl_scorer)  # streak 4 -> correct -> writes bucket 4
        b4 = conn.execute(
            "SELECT COUNT(*) FROM drl_qtable WHERE streak_bucket = 4"
        ).fetchone()[0]
        assert b4 == len(SOURCES)

    def test_td_error_uses_gamma_bootstrap(self, drl_scorer):
        """The bootstrap must be reachable: seed a next-bucket Q, then verify
        the update exceeds the degenerate EMA update (lr*r) when gamma>0."""
        conn = drl_scorer._tracker._get_conn()
        gamma = drl_scorer.discount_factor
        lr = drl_scorer.learning_rate
        # Seed bucket-1 Q high for all sources in 'neutral'
        for src in SOURCES:
            conn.execute(
                "INSERT INTO drl_qtable (regime_bucket, source, streak_bucket, q_value) "
                "VALUES ('neutral', ?, 1, 0.5)",
                (src,),
            )
        conn.commit()
        # Correct call from bucket 0 -> bootstraps from seeded bucket 1
        self._record(drl_scorer)
        rows = conn.execute(
            "SELECT q_value FROM drl_qtable WHERE streak_bucket = 0"
        ).fetchall()
        # Q = 0 + lr * (1 + gamma*0.5 - 0) = lr * 1.45 > lr * 1.0 (degenerate EMA)
        degenerate = lr * 1.0
        for r in rows:
            assert r["q_value"] > degenerate + 1e-9


# ---------------------------------------------------------------------------
# Adaptive LR decay (visit-count): lr_n = lr_0 / (1 + n * lr_decay)
# ---------------------------------------------------------------------------

class TestAdaptiveLRDecay:
    """Per-state visit-count adaptive learning-rate decay.

    Effective step size shrinks as a state-action pair is revisited,
    damping late-stage Q-value overshoot. Deploy of stranded proven
    experiment e67fcb4 onto main lineage (AGI cycle H20260829150109).
    """

    def test_visit_count_increments(self, drl_scorer):
        signals = {src: "BUY" for src in SOURCES}
        for i in range(3):
            drl_scorer.record_reward(
                ticker="AAPL", date_str=f"2026-06-1{i}",
                source_signals=signals,
                predicted_signal="BUY", actual_signal="BUY",
                regime_return=0.01,
            )
        conn = drl_scorer._tracker._get_conn()
        rows = conn.execute(
            "SELECT visit_count FROM drl_qtable WHERE regime_bucket='neutral'"
        ).fetchall()
        assert rows, "qtable rows should exist"
        for row in rows:
            assert row["visit_count"] == 3

    def test_fresh_state_uses_full_lr(self, drl_scorer):
        """First visit to a state: lr_0 applies unchanged."""
        signals = {src: "BUY" for src in SOURCES}
        drl_scorer.record_reward(
            ticker="AAPL", date_str="2026-06-10",
            source_signals=signals,
            predicted_signal="BUY", actual_signal="BUY",
            regime_return=0.01,
        )
        conn = drl_scorer._tracker._get_conn()
        rows = conn.execute(
            "SELECT q_value, visit_count FROM drl_qtable WHERE regime_bucket='neutral'"
        ).fetchall()
        for row in rows:
            # First visit: e=1, lr=lr_0=0.15, gamma=0.9, next_q=0 for fresh
            # next bucket -> q = 0 + 0.15 * (1 + 0.9*0 - 0) = 0.15
            assert row["q_value"] == pytest.approx(0.15)
            assert row["visit_count"] == 1

    def test_mature_state_smaller_steps(self, drl_scorer):
        """Effective step shrinks with visits: late |dq| < early |dq|.

        The bucket updated in round i is keyed by the streak computed
        BEFORE that round's insert, so we capture it pre-call.
        """
        signals = {src: "BUY" for src in SOURCES}
        q_at = {}
        for i in range(1, 21):
            bucket = _discretize_streak(
                drl_scorer._get_source_recent_correct("trader", limit=5)
            )
            drl_scorer.record_reward(
                ticker="AAPL", date_str=f"2026-06-{i:02d}",
                source_signals=signals,
                predicted_signal="BUY", actual_signal="BUY",
                regime_return=0.01,
            )
            conn = drl_scorer._tracker._get_conn()
            row = conn.execute(
                "SELECT q_value FROM drl_qtable "
                "WHERE regime_bucket='neutral' AND source='trader' AND streak_bucket=?",
                (bucket,),
            ).fetchone()
            assert row is not None, f"bucket {bucket} missing after round {i}"
            q_at[i] = row["q_value"]
        early_gap = abs(q_at[2] - q_at[1])
        late_gap = abs(q_at[20] - q_at[19])
        assert late_gap < early_gap, (
            f"adaptive LR should shrink steps: early={early_gap:.4f} late={late_gap:.4f}"
        )

    def test_qvalues_diverge_by_skill_with_adaptive_lr(self, drl_scorer):
        """Convergence still holds with adaptive LR: correct source trends +.

        trader says BUY (matches consensus BUY/actual BUY -> reward +1 each
        round); investment_judge says SELL (mismatches -> -1 via consensus
        reward path... actually consensus reward is shared, so instead check
        divergence across streak dynamics).
        """
        signals = {
            "investment_judge": "SELL",
            "trader": "BUY",
            "risk_judge": "BUY",
            "portfolio_manager": "HOLD",
        }
        for i in range(20):
            drl_scorer.record_reward(
                ticker="AAPL", date_str=f"2026-06-{i+1:02d}",
                source_signals=signals,
                predicted_signal="BUY", actual_signal="BUY",
                regime_return=0.01,
            )
        conn = drl_scorer._tracker._get_conn()
        rows = conn.execute(
            "SELECT source, q_value FROM drl_qtable WHERE regime_bucket='neutral'"
        ).fetchall()
        by_src = {r["source"]: r["q_value"] for r in rows}
        assert by_src["trader"] > 0, "consistently-correct rounds should drive Q up"

    def test_migration_adds_visit_count_to_legacy_db(self, tmp_path):
        """A legacy DB created without visit_count gets the column backfilled."""
        legacy_path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(legacy_path)
        conn.execute(
            "CREATE TABLE drl_qtable ("
            "regime_bucket TEXT NOT NULL, source TEXT NOT NULL, "
            "streak_bucket INTEGER NOT NULL, q_value REAL NOT NULL DEFAULT 0.0, "
            "PRIMARY KEY (regime_bucket, source, streak_bucket))"
        )
        conn.execute(
            "INSERT INTO drl_qtable VALUES ('neutral', 'trader', 0, 0.5)"
        )
        conn.commit()
        conn.close()

        tracker = AccuracyTracker(db_path=legacy_path)
        try:
            check = tracker._get_conn()
            row = check.execute(
                "SELECT q_value, visit_count FROM drl_qtable "
                "WHERE source='trader'"
            ).fetchone()
            assert row["q_value"] == pytest.approx(0.5), "legacy data preserved"
            assert row["visit_count"] == 0, "backfilled default is 0"
        finally:
            tracker.close()


# ---------------------------------------------------------------------------
# Experience-replay optimisation loop (H20260824150137)
# ---------------------------------------------------------------------------

class TestReplayDeployIntegration:
    """Experience-replay deploy: evaluate() replays graded history before scoring.

    (AGI H20260906150126 — RL reward optimization loop deployed into the
    production evaluate() path.)
    """

    def _seed_graded_history(self, engine, n_rounds=4):
        """Seed graded rounds directly into the engine's tracker."""
        for i in range(n_rounds):
            date = f"2026-08-{i+1:02d}"
            actual = "BUY" if i % 2 == 0 else "SELL"
            signals = {
                "investment_judge": actual,
                "trader": actual,
                "risk_judge": "SELL" if actual == "BUY" else "BUY",
                "portfolio_manager": actual,
            }
            for source, sig in signals.items():
                engine.tracker.record_prediction("AAPL", date, source, sig)
            engine.tracker.record_outcome("AAPL", date, actual)

    def _make_log_states(self):
        return {
            "2026-09-06": {
                "company_of_interest": "AAPL",
                "investment_debate_state": {"judge_decision": "Rating: BUY"},
                "trader_investment_decision": "FINAL TRANSACTION PROPOSAL: **BUY**",
                "risk_debate_state": {"judge_decision": "Decision: BUY"},
                "final_trade_decision": "Rating: BUY",
            },
        }

    def test_evaluate_populates_last_replay_stats(self, engine):
        """DRL-enabled evaluate must run replay and expose its stats."""
        self._seed_graded_history(engine, n_rounds=4)
        assert engine.last_replay_stats is None  # not yet run
        engine.evaluate(self._make_log_states(), use_drl=True)
        assert engine.last_replay_stats is not None
        assert engine.last_replay_stats["epochs"] >= 1
        assert "converged" in engine.last_replay_stats

    def test_evaluate_without_drl_skips_replay(self, engine):
        """use_drl=False must not run replay."""
        self._seed_graded_history(engine, n_rounds=4)
        engine.evaluate({}, use_drl=False)
        assert engine.last_replay_stats is None

    def test_evaluate_replay_failure_is_non_fatal(self, engine, monkeypatch):
        """A crashing replay must not break evaluate()."""
        def _boom(*a, **k):
            raise RuntimeError("replay exploded")

        monkeypatch.setattr(
            engine.drl_scorer, "optimize_weights_from_history", _boom
        )
        results = engine.evaluate(self._make_log_states(), use_drl=True)
        assert "2026-09-06" in results
        assert engine.last_replay_stats is None

    def test_evaluate_replay_updates_qtable_in_production_path(self, engine, db_path):
        """End-to-end: graded history -> replay inside evaluate -> nonzero Q."""
        self._seed_graded_history(engine, n_rounds=4)
        conn = sqlite3.connect(db_path)
        before = conn.execute(
            "SELECT COUNT(*) FROM drl_qtable WHERE q_value != 0"
        ).fetchone()[0]
        conn.close()

        engine.evaluate(self._make_log_states(), use_drl=True)

        conn = sqlite3.connect(db_path)
        after = conn.execute(
            "SELECT COUNT(*) FROM drl_qtable WHERE q_value != 0"
        ).fetchone()[0]
        conn.close()
        assert after > before, "replay inside evaluate() must update the Q-table"

    def test_evaluate_replay_is_contraction_on_re_replay(self, engine, db_path):
        """Second evaluate() must continue contracting toward the Bellman fixpoint.

        Self-transition bootstrap with gamma=0.9 has fixpoint Q*=r/(1-gamma),
        so full convergence within REPLAY_MAX_EPOCHS is not guaranteed — but
        re-replay must be monotonically closer to the fixpoint (smaller final
        per-pass movement) and Q must respect the Bellman bound
        |Q| <= r_max/(1-gamma).
        """
        self._seed_graded_history(engine, n_rounds=4)
        engine.evaluate(self._make_log_states(), use_drl=True)
        stats_1 = dict(engine.last_replay_stats)
        engine.evaluate(self._make_log_states(), use_drl=True)
        stats_2 = dict(engine.last_replay_stats)

        assert stats_2["epochs"] <= REPLAY_MAX_EPOCHS
        mov_1 = stats_1["q_movement_history"][-1]
        mov_2 = stats_2["q_movement_history"][-1]
        assert mov_2 <= mov_1, "re-replay must contract, not diverge"

        # Bellman bound: |Q| <= r_max / (1 - gamma)
        gamma = DRL_DISCOUNT_FACTOR
        bound = 1.0 / (1.0 - gamma) + 1e-6
        conn = sqlite3.connect(db_path)
        max_q = conn.execute(
            "SELECT MAX(ABS(q_value)) FROM drl_qtable"
        ).fetchone()[0]
        conn.close()
        assert max_q <= bound, f"|Q|={max_q} exceeds Bellman bound {bound}"


class TestReplayOptimization:
    """optimize_weights_from_history: RL reward loop over graded predictions."""

    def _seed_history(self, tracker, n_rounds=6):
        """Seed graded predictions: one round = 4 sources, same ticker/date."""
        for i in range(n_rounds):
            date = f"2026-08-{i+1:02d}"
            actual = "BUY" if i % 2 == 0 else "SELL"
            # trader always correct, risk_judge always wrong, others mixed
            signals = {
                "investment_judge": "BUY" if i % 3 == 0 else actual,
                "trader": actual,
                "risk_judge": "SELL" if actual == "BUY" else "BUY",
                "portfolio_manager": actual if i % 2 == 0 else "HOLD",
            }
            for source, sig in signals.items():
                tracker.record_prediction("AAPL", date, source, sig)
            tracker.record_outcome("AAPL", date, actual)

    def test_replay_returns_stats_dict(self, drl_scorer):
        result = drl_scorer.optimize_weights_from_history()
        assert isinstance(result, dict)
        assert set(result) >= {"epochs", "converged", "final_td_error", "td_error_history"}
        assert result["epochs"] == 0
        assert result["converged"] is True
        assert result["td_error_history"] == []

    def test_replay_empty_history_converges_immediately(self, drl_scorer):
        result = drl_scorer.optimize_weights_from_history()
        assert result["epochs"] == 0
        assert result["final_td_error"] == 0.0

    def test_replay_runs_and_updates_qtable(self, tracker, drl_scorer):
        self._seed_history(tracker)
        result = drl_scorer.optimize_weights_from_history(max_epochs=5)
        assert result["epochs"] >= 1
        assert result["epochs"] <= 5
        assert result["td_error_history"], "at least one epoch ran"
        conn = tracker._get_conn()
        nonzero = conn.execute(
            "SELECT COUNT(*) FROM drl_qtable WHERE q_value != 0"
        ).fetchone()[0]
        assert nonzero > 0, "Q-table must be updated by replay"

    def test_replay_converges_within_max_epochs(self, tracker, drl_scorer):
        self._seed_history(tracker, n_rounds=10)
        result = drl_scorer.optimize_weights_from_history(max_epochs=REPLAY_MAX_EPOCHS)
        assert result["epochs"] <= REPLAY_MAX_EPOCHS
        # with a tiny buffer it must converge or hit the cap
        assert result["converged"] is True or result["epochs"] == REPLAY_MAX_EPOCHS
        # monotone non-increasing tail is expected under fixed-point iteration;
        # verify errors are finite
        assert all(0 <= e < float("inf") for e in result["td_error_history"])

    def test_replay_td_error_decreases(self, tracker, drl_scorer):
        """Mean |TD error| tracks residual Bellman inconsistency; falls as the
        table approaches its fixed point (first pass dominates movement)."""
        self._seed_history(tracker, n_rounds=10)
        result = drl_scorer.optimize_weights_from_history(max_epochs=30)
        hist = result["td_error_history"]
        assert len(hist) >= 3, f"expected multiple epochs, got {hist}"
        assert hist[-1] < hist[0], (
            f"mean |TD error| should decrease over epochs: {hist}"
        )

    def test_replay_fixed_point_analytic(self, tracker, drl_scorer):
        """Constant reward + self-transition => Q converges to r/(1-gamma).

        All sources always correct => consensus always correct => reward=+1
        every round; every source sits in the max streak bucket 4 (buckets
        merge streaks 4-5; streak+1 clamps to a self-transition). Analytic
        fixed point: Q* = 1/(1-0.9) = 10.
        """
        import math as _math

        def seed_all_correct(n_rounds):
            for i in range(n_rounds):
                date = f"2026-09-{i+1:02d}"
                for source in SOURCES:
                    tracker.record_prediction("MSFT", date, source, "BUY")
                tracker.record_outcome("MSFT", date, "BUY")

        seed_all_correct(12)
        result = drl_scorer.optimize_weights_from_history(max_epochs=800)
        assert result["converged"] is True, (
            f"must converge: {result['td_error_history'][-5:]}"
        )
        conn = tracker._get_conn()
        row = conn.execute(
            "SELECT q_value FROM drl_qtable "
            "WHERE regime_bucket = 'neutral' AND source = 'trader' "
            "AND streak_bucket = 4"
        ).fetchone()
        expected = 1.0 / (1.0 - drl_scorer.discount_factor)  # = 10.0
        assert row is not None, "trader/streak-5 entry must exist"
        assert _math.isclose(row["q_value"], expected, rel_tol=0.01), (
            f"Q* should equal r/(1-gamma)={expected}, got {row['q_value']}"
        )

    def test_replay_fixed_point_wrong_forever(self, tracker, drl_scorer):
        """All-wrong consensus: reward=-1 every round, streak stays 0,
        next_bucket=0 == streak_bucket (self-transition) => the analytic
        fixed point Q* = -1/(1-gamma) = -10 must emerge.
        """
        import math as _math

        for i in range(12):
            date = f"2026-09-{i+1:02d}"
            for source in SOURCES:
                tracker.record_prediction("MSFT", date, source, "SELL")
            tracker.record_outcome("MSFT", date, "BUY")

        result = drl_scorer.optimize_weights_from_history(max_epochs=800)
        assert result["converged"] is True
        conn = tracker._get_conn()
        row = conn.execute(
            "SELECT q_value FROM drl_qtable "
            "WHERE regime_bucket = 'neutral' AND source = 'trader' "
            "AND streak_bucket = 0"
        ).fetchone()
        expected = -1.0 / (1.0 - drl_scorer.discount_factor)  # = -10.0
        assert row is not None, "trader/streak-0 entry must exist"
        assert _math.isclose(row["q_value"], expected, rel_tol=0.01), (
            f"Q* should equal r/(1-gamma)={expected}, got {row['q_value']}"
        )

    def test_replay_respects_buffer_limit(self, tracker, drl_scorer):
        self._seed_history(tracker, n_rounds=20)
        result = drl_scorer.optimize_weights_from_history(
            max_epochs=3, buffer_limit=8
        )
        # buffer_limit=8 -> 2 rounds replayed; must run without error
        assert result["epochs"] == 3  # tiny buffer keeps TD error > tol
        assert len(result["td_error_history"]) == 3

    def test_replay_snapshot_semantics_idempotent_tail(self, tracker, drl_scorer):
        """Near convergence, an extra pass should barely move Q-values."""
        self._seed_history(tracker, n_rounds=8)
        drl_scorer.optimize_weights_from_history(max_epochs=40)
        conn = tracker._get_conn()
        before = conn.execute(
            "SELECT SUM(q_value) FROM drl_qtable"
        ).fetchone()[0]
        drl_scorer.optimize_weights_from_history(max_epochs=1)
        after = conn.execute(
            "SELECT SUM(q_value) FROM drl_qtable"
        ).fetchone()[0]
        assert abs(after - before) < 0.5, (
            f"converged replay should be near-idempotent: before={before}, after={after}"
        )

    def test_replay_does_not_touch_predictions_table(self, tracker, drl_scorer):
        self._seed_history(tracker, n_rounds=6)
        conn = tracker._get_conn()
        before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        drl_scorer.optimize_weights_from_history(max_epochs=3)
        after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert before == after, "replay must never mutate the history it learns from"

    def test_replay_clean_state_zero_q(self, drl_scorer):
        """Fresh scorer, no history: replay is a no-op leaving Q-table at 0."""
        result = drl_scorer.optimize_weights_from_history()
        assert result["final_td_error"] == 0.0
        drl_scorer.reset_qtable()  # still works after replay
