"""Tests for task-conditioned panel selection (arxiv:2611.22998v1).

A source is benched when it has >= PANEL_MIN_GRADED graded predictions and
Wilson lower-bound accuracy < PANEL_BENCH_WILSON.  Insufficient evidence or
over-benching falls back to the full 4-source panel.
"""

import pytest

from tradingagents.graph.consensus import (
    AccuracyTracker,
    ConfidenceScorer,
    PANEL_MIN_GRADED,
    PANEL_BENCH_WILSON,
    SOURCES,
    _wilson_lower_bound,
)


N = PANEL_MIN_GRADED + 3  # comfortably above the benching evidence bar


def _seed(tracker, correct_by_source):
    """Seed one prediction/outcome pair per source per day."""
    day = 0
    for src, n_correct in correct_by_source.items():
        n_wrong = N - n_correct
        for i in range(n_correct):
            day += 1
            tracker.record_prediction("AAPL", f"2026-01-{day:02d}", src, "BUY")
            tracker.record_outcome("AAPL", f"2026-01-{day:02d}", "BUY")
        for i in range(n_wrong):
            day += 1
            tracker.record_prediction("AAPL", f"2026-01-{day:02d}", src, "BUY")
            tracker.record_outcome("AAPL", f"2026-01-{day:02d}", "SELL")


@pytest.fixture
def tracker(tmp_path):
    t = AccuracyTracker(db_path=str(tmp_path / "panel.db"))
    yield t
    t.close()


class TestGetPanel:
    def test_fresh_tracker_full_panel(self, tracker):
        """No graded history -> everyone seats (legacy behaviour)."""
        seated, weights = tracker.get_panel()
        assert seated == list(SOURCES)
        assert len(weights) == 4

    def test_proven_bad_source_is_benched(self, tracker):
        """trader graded 5/18 (~28%) is benched; skilled sources stay."""
        _seed(tracker, {
            "investment_judge": 13,
            "trader": 5,
            "risk_judge": 12,
            "portfolio_manager": 11,
        })
        seated, weights = tracker.get_panel()
        assert "trader" not in seated
        assert seated == ["investment_judge", "risk_judge", "portfolio_manager"]
        # weights renormalised over seated only, sum to ~1
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)
        assert "trader" not in weights

    def test_small_sample_source_not_benched(self, tracker):
        """A source with < PANEL_MIN_GRADED predictions never benches,
        even at 0% accuracy (Wilson shrinkage protects the unproven)."""
        # others carry the totals; trader has only 3 graded predictions
        _seed_partial(tracker)
        seated, _ = tracker.get_panel()
        assert "trader" in seated

    def test_over_benching_falls_back_to_full_panel(self, tracker):
        """If benching would leave <2 seated, the full panel is retained."""
        _seed(tracker, {src: 4 for src in SOURCES})  # everyone ~22%
        seated, weights = tracker.get_panel()
        assert seated == list(SOURCES)
        assert len(weights) == 4

    def test_skilled_panel_unchanged(self, tracker):
        """Everyone above the bench threshold -> full panel, no regression."""
        _seed(tracker, {
            "investment_judge": 14,
            "trader": 13,
            "risk_judge": 12,
            "portfolio_manager": 13,
        })
        seated, weights = tracker.get_panel()
        assert seated == list(SOURCES)
        for src in SOURCES:
            assert _wilson_lower_bound(
                *self._stats(tracker, src)
            ) >= PANEL_BENCH_WILSON or src not in seated

    @staticmethod
    def _stats(tracker, src):
        s = tracker.get_source_stats()[src]
        return int(s["correct"]), int(s["total"])


def _seed_partial(tracker):
    """Seed 18 predictions for 3 sources; only 3 graded for trader."""
    day = 0
    for src in ("investment_judge", "risk_judge", "portfolio_manager"):
        for i in range(14):
            day += 1
            tracker.record_prediction("AAPL", f"2026-02-{day:02d}", src, "BUY")
            tracker.record_outcome("AAPL", f"2026-02-{day:02d}", "BUY")
    for i in range(3):
        day += 1
        tracker.record_prediction("AAPL", f"2026-02-{day:02d}", "trader", "BUY")
        tracker.record_outcome("AAPL", f"2026-02-{day:02d}", "SELL")


class TestScorerPanelBehaviour:
    def test_benched_source_signal_is_ignored(self, tracker):
        """The benched trader's BUY cannot drag the execution block."""
        _seed(tracker, {
            "investment_judge": 13,
            "trader": 5,
            "risk_judge": 12,
            "portfolio_manager": 11,
        })
        scorer = ConfidenceScorer(tracker)
        result = scorer.score({
            "investment_judge": "SELL",
            "trader": "BUY",           # benched voice
            "risk_judge": "SELL",
            "portfolio_manager": "SELL",
        })
        # Without benching: exec block = BUY,SELL,SELL -> SELL; ij=SELL -> SELL
        # unanimous-check differs: trader's BUY no longer breaks unanimity.
        assert result.final_signal == "SELL"
        assert result.unanimous is True  # benched source doesn't count
        assert result.weights["investment_judge"] > 0
        assert result.weights["execution_block"] > 0

    def test_benched_trader_cannot_flip_exec_majority(self, tracker):
        """trader+one exec say BUY, other exec says SELL; ij benched too.
        With trader benched the exec block majority is SELL."""
        _seed(tracker, {
            "investment_judge": 4,   # benched
            "trader": 4,             # benched
            "risk_judge": 13,
            "portfolio_manager": 12,
        })
        scorer = ConfidenceScorer(tracker)
        result = scorer.score({
            "investment_judge": "BUY",
            "trader": "BUY",
            "risk_judge": "SELL",
            "portfolio_manager": "SELL",
        })
        # ij benched -> exec block is risk+PM (both SELL) -> SELL with full weight
        assert result.final_signal == "SELL"
        assert result.weights["investment_judge"] == 0.0
        assert result.weights["execution_block"] == pytest.approx(1.0)

    def test_fresh_tracker_matches_legacy_tier_weights(self, tracker):
        """No history -> original 0.6/0.4 tier split is preserved."""
        scorer = ConfidenceScorer(tracker)
        result = scorer.score({src: "HOLD" for src in SOURCES})
        assert result.final_signal == "HOLD"
        assert result.weights["investment_judge"] == pytest.approx(0.6)
        assert result.weights["execution_block"] == pytest.approx(0.4)
