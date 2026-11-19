"""Task-conditioned panel routing (arxiv:2611.18150v1, H20261119150056-3).

Seating is decided per task type: a source proven bad *within a task* is
benched for that task only, while keeping its seat everywhere else.  Thin
per-task evidence falls back to the global panel (never bench on noise).
"""

from unittest.mock import patch

import pytest

from tradingagents.graph.consensus import (
    AccuracyTracker,
    ConfidenceScorer,
    MIN_PREDICTIONS_FOR_WEIGHTING,
    PANEL_MIN_GRADED,
    SOURCES,
)


N = PANEL_MIN_GRADED + 3  # comfortably above the benching evidence bar


def _seed(tracker, ticker, correct_by_source, task_type="default"):
    """Seed one graded prediction per source per day for a given task."""
    day = 0
    for src, n_correct in correct_by_source.items():
        for _ in range(n_correct):
            day += 1
            tracker.record_prediction(ticker, f"2026-01-{day:02d}", src, "BUY", task_type)
            tracker.record_outcome(ticker, f"2026-01-{day:02d}", "BUY")
        for _ in range(N - n_correct):
            day += 1
            tracker.record_prediction(ticker, f"2026-01-{day:02d}", src, "BUY", task_type)
            tracker.record_outcome(ticker, f"2026-01-{day:02d}", "SELL")


@pytest.fixture
def tracker(tmp_path):
    t = AccuracyTracker(db_path=str(tmp_path / "task_panel.db"))
    yield t
    t.close()


class TestTaskConditionedSeating:
    def test_default_task_column_backfilled(self, tracker):
        """record_prediction without task_type stores 'default'."""
        tracker.record_prediction("AAPL", "2026-02-01", "trader", "BUY")
        row = tracker._get_conn().execute(
            "SELECT task_type FROM predictions WHERE ticker = 'AAPL'"
        ).fetchone()
        assert row["task_type"] == "default"

    def test_bad_in_task_seats_globally(self, tracker):
        """trader proven bad on 'vol_scan' is benched there, seated elsewhere."""
        # Healthy global history for everyone (72% raw, Wilson LB ~0.50)
        _seed(tracker, "AAPL", {src: 13 for src in SOURCES})
        # Task history: trader is terrible on volatility scans, others unproven
        _seed(tracker, "VOLX", {"trader": 4}, task_type="vol_scan")

        seated_task, w_task = tracker.get_panel(task_type="vol_scan")
        assert "trader" not in seated_task
        assert set(seated_task) == {"investment_judge", "risk_judge", "portfolio_manager"}
        assert sum(w_task.values()) == pytest.approx(1.0, abs=0.001)

        # The same source keeps its seat with no task conditioning
        seated_global, _ = tracker.get_panel()
        assert "trader" in seated_global
        assert seated_global == list(SOURCES)

    def test_task_split_panels(self, tracker):
        """Two tasks can bench two different sources simultaneously."""
        _seed(tracker, "AAPL", {src: 13 for src in SOURCES})
        _seed(tracker, "TKA", {"trader": 4}, task_type="vol_scan")
        _seed(tracker, "TKB", {"risk_judge": 3}, task_type="earnings")

        assert "trader" not in tracker.get_panel(task_type="vol_scan")[0]
        assert "risk_judge" not in tracker.get_panel(task_type="earnings")[0]
        # ...and each keeps its seat for the *other* task
        assert "risk_judge" in tracker.get_panel(task_type="vol_scan")[0]
        assert "trader" in tracker.get_panel(task_type="earnings")[0]

    def test_thin_task_evidence_falls_back_to_global(self, tracker):
        """< MIN_PREDICTIONS_FOR_WEIGHTING graded in-task -> global panel."""
        _seed(tracker, "AAPL", {src: 13 for src in SOURCES})
        # Only 3 graded rows inside the task (1/3 correct for trader)
        for i in range(3):
            d = f"2026-03-{i + 1:02d}"
            tracker.record_prediction("VOLX", d, "trader", "BUY", "vol_scan")
            tracker.record_outcome("VOLX", d, "SELL" if i else "BUY")
        assert 3 < MIN_PREDICTIONS_FOR_WEIGHTING

        seated, _ = tracker.get_panel(task_type="vol_scan")
        assert seated == list(SOURCES)  # global panel: everyone healthy

    def test_over_benching_in_task_retains_full_panel(self, tracker):
        """Benching 3+ sources within a task leaves <2 seated -> full panel."""
        _seed(tracker, "AAPL", {src: 13 for src in SOURCES})
        _seed(
            tracker, "VOLX",
            {src: 4 for src in SOURCES},  # everyone ~22% in-task
            task_type="vol_scan",
        )
        seated, _ = tracker.get_panel(task_type="vol_scan")
        assert seated == list(SOURCES)


class TestScorerRouting:
    def test_scorer_passes_task_type_to_panel(self, tracker):
        """ConfidenceScorer.score routes task_type into get_panel."""
        scorer = ConfidenceScorer(tracker)
        signals = {src: "BUY" for src in SOURCES}
        with patch.object(
            tracker, "get_panel", wraps=tracker.get_panel
        ) as spy:
            result = scorer.score(signals, task_type="vol_scan")
        assert spy.call_count == 1
        assert spy.call_args.kwargs.get("task_type") == "vol_scan"
        assert result.final_signal == "BUY"

    def test_benched_source_excluded_from_task_consensus(self, tracker):
        """Unanimity/unanimity-discount reflects seated members only."""
        _seed(tracker, "AAPL", {src: 13 for src in SOURCES})
        _seed(tracker, "VOLX", {"trader": 4}, task_type="vol_scan")
        scorer = ConfidenceScorer(tracker)

        signals = {src: "BUY" for src in SOURCES}
        res_task = scorer.score(signals, task_type="vol_scan")
        res_global = scorer.score(signals)
        # Identical signal sets, but different panels -> same BUY, both valid
        assert res_task.final_signal == "BUY"
        assert res_global.final_signal == "BUY"
