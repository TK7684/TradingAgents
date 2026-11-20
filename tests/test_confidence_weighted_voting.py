"""Confidence-weighted voting (arxiv:2611.18150v1, H20261120150051-3).

Tier votes are scaled by the mean Wilson-LB panel weight of each tier's
seated members.  Equal weights (thin history) cancel in the normalisation,
so the legacy fixed 0.60/0.40 split is preserved exactly until the track
records justify a shift.
"""

import pytest

from tradingagents.graph.consensus import (
    AccuracyTracker,
    ConfidenceScorer,
    SOURCES,
)

N = 18  # graded per source: above bench bar (15) and weighting bar (10)

# IJ says HOLD; execution block majority (trader + risk_judge) says BUY.
SIGNALS = {
    "investment_judge": "HOLD",
    "trader": "BUY",
    "risk_judge": "BUY",
    "portfolio_manager": "HOLD",
}


def _seed_src(tracker, src, n_correct, offset):
    """n_correct of N predictions right; remainder wrong."""
    ticker = f"TK{offset}"
    for i in range(N):
        pred = "BUY" if i < n_correct else "SELL"
        tracker.record_prediction(ticker, f"2026-04-{i + 1:02d}", src, pred)
        tracker.record_outcome(ticker, f"2026-04-{i + 1:02d}", "BUY")


@pytest.fixture
def scorer(tmp_path):
    t = AccuracyTracker(db_path=str(tmp_path / "cwv.db"))
    yield ConfidenceScorer(t), t
    t.close()


class TestConfidenceWeightedVoting:
    def test_thin_history_keeps_fixed_tiers(self, scorer):
        """No graded history -> equal panel weights cancel -> legacy split."""
        sc, _t = scorer
        r = sc.score(dict(SIGNALS))
        assert r.final_signal == "HOLD"          # 0.60 beats 0.40
        assert r.confidence == pytest.approx(0.6)
        assert r.weights == {"investment_judge": 0.6, "execution_block": 0.4}

    def test_weak_judge_loses_vote_to_strong_exec(self, scorer):
        """IJ seated but unproven vs a strong exec block -> BUY flips the call."""
        sc, t = scorer
        _seed_src(t, "investment_judge", 11, 1)          # 61% raw, Wilson LB ~0.39
        for k, src in enumerate(("trader", "risk_judge", "portfolio_manager"), 2):
            _seed_src(t, src, 16, k)                      # 89% raw, Wilson LB ~0.74
        r = sc.score(dict(SIGNALS))
        # Legacy fixed tiers would give HOLD (0.60 vs 0.40); the learned
        # confidence shift must let the execution block win instead.
        assert r.final_signal == "BUY"
        assert r.weights["execution_block"] > r.weights["investment_judge"]

    def test_strong_judge_strengthens_hold(self, scorer):
        """Proven IJ vs mediocre exec block -> HOLD margin widens past 0.60."""
        sc, t = scorer
        _seed_src(t, "investment_judge", 16, 1)
        for k, src in enumerate(("trader", "risk_judge", "portfolio_manager"), 2):
            _seed_src(t, src, 11, k)
        r = sc.score(dict(SIGNALS))
        assert r.final_signal == "HOLD"
        assert r.weights["investment_judge"] > 0.6
        assert r.confidence > 0.6

    def test_all_sources_seated_in_flip_case(self, scorer):
        """The shift comes from weighting, not benching: full panel seated."""
        sc, t = scorer
        _seed_src(t, "investment_judge", 11, 1)
        for k, src in enumerate(("trader", "risk_judge", "portfolio_manager"), 2):
            _seed_src(t, src, 16, k)
        seated, _w = t.get_panel()
        assert sorted(seated) == sorted(SOURCES)
