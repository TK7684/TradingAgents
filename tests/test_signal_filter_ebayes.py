"""Tests for EB (empirical-Bayes) shrinkage in SignalFilter._ticker_accuracy.

Pattern: small-sample accuracy overfitting — raw correct/total reads a 1/1
ticker as 100% accurate; beta-binomial shrinkage toward 50% with strength 5
pulls it to (1+2.5)/(1+5) = 58.3%.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from signal_filter import SignalFilter, EB_PRIOR_MEAN, EB_PRIOR_STRENGTH


@pytest.fixture()
def db(tmp_path):
    """Fresh signal_accuracy.db with the production schema."""
    path = tmp_path / "signal_accuracy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            predicted_signal TEXT NOT NULL,
            actual_signal    TEXT,
            correct     INTEGER,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        , pct_1d REAL, pct_3d REAL, pct_5d REAL);
        CREATE TABLE source_stats (
            source             TEXT PRIMARY KEY,
            total_predictions  INTEGER NOT NULL DEFAULT 0,
            correct_predictions INTEGER NOT NULL DEFAULT 0,
            accuracy           REAL NOT NULL DEFAULT 0.0,
            updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def _add_prediction(conn, ticker, correct, source="analyst_a", signal="BUY"):
    conn.execute(
        "INSERT INTO predictions (ticker, date, source, predicted_signal, "
        "actual_signal, correct) VALUES (?, '2026-01-01', ?, ?, 'SELL', ?)",
        (ticker, source, signal, correct),
    )


class TestEBShrinkage:
    def test_no_data_returns_neutral(self, db):
        sf = SignalFilter(db_path=db)
        acc, n = sf._ticker_accuracy("NVDA")
        assert n == 0
        assert acc == pytest.approx(50.0)

    def test_tiny_perfect_sample_is_shrunk_not_100(self, db):
        # 1/1 correct: raw = 100%, EB = (1 + 2.5) / (1 + 5) = 58.33%
        sf = SignalFilter(db_path=db)
        conn = sqlite3.connect(db)
        _add_prediction(conn, "NVDA", 1)
        conn.commit(); conn.close()
        acc, n = sf._ticker_accuracy("NVDA")
        assert n == 1
        assert acc == pytest.approx((1 + EB_PRIOR_STRENGTH * EB_PRIOR_MEAN)
                                    / (1 + EB_PRIOR_STRENGTH) * 100)

    def test_tiny_bad_sample_is_shrunk_not_0(self, db):
        # 0/2 correct: raw = 0%, EB = 2.5 / 7 = 35.7% — not catastrophically 0
        sf = SignalFilter(db_path=db)
        conn = sqlite3.connect(db)
        _add_prediction(conn, "AMD", 0)
        _add_prediction(conn, "AMD", 0)
        conn.commit(); conn.close()
        acc, n = sf._ticker_accuracy("AMD")
        assert n == 2
        assert acc == pytest.approx(35.714, abs=0.01)
        assert acc > 20  # above BLOCK threshold: no premature hard-block

    def test_large_sample_converges_to_raw(self, db):
        # 60/100: EB = 62.5/105 = 59.5% ≈ raw 60%
        sf = SignalFilter(db_path=db)
        conn = sqlite3.connect(db)
        for i in range(60):
            _add_prediction(conn, "T", 1)
        for i in range(40):
            _add_prediction(conn, "T", 0)
        conn.commit(); conn.close()
        acc, n = sf._ticker_accuracy("T")
        assert n == 100
        assert acc == pytest.approx(59.52, abs=0.01)
        assert abs(acc - 60.0) < 1.0  # close to raw

    def test_blocking_still_works_on_genuinely_bad_history(self, db):
        # 3/25 = 12% raw; EB = 5.5/30 = 18.3% — still below BLOCK_THRESHOLD=20
        sf = SignalFilter(db_path=db)
        conn = sqlite3.connect(db)
        for i in range(3):
            _add_prediction(conn, "BAD", 1)
        for i in range(22):
            _add_prediction(conn, "BAD", 0)
        conn.commit(); conn.close()
        allowed, reason = sf.should_trade("BAD", "BUY")
        assert allowed is False
        assert reason.startswith("BLOCK")

    def test_position_weight_uses_shrunk_accuracy(self, db):
        # 1/1 → EB 58.3% → falls in 40–60% band → weight 0.75, not 1.0
        sf = SignalFilter(db_path=db)
        conn = sqlite3.connect(db)
        _add_prediction(conn, "NVDA", 1)
        conn.commit(); conn.close()
        w = sf.get_position_weight("NVDA")
        assert w == pytest.approx(0.75)


class TestCLIUnchanged:
    def test_module_constants_match_consensus_convention(self):
        # Consistency with consensus.py BAYES_PRIOR_STRENGTH = 5
        assert EB_PRIOR_STRENGTH == 5
        assert EB_PRIOR_MEAN == 0.5


class TestSourceEBShrinkage:
    """Source-level WARN gate must use EB-shrunk accuracy, not raw.

    Raw 0/2 reads as 0% and would spuriously WARN-downweight every trade;
    shrunk it reads (0+2.5)/(2+5) = 35.7% — above the 25% WARN threshold.
    """

    def _stats(self, db, rows):
        conn = sqlite3.connect(db)
        conn.executemany(
            "INSERT INTO source_stats (source, total_predictions, "
            "correct_predictions, accuracy, updated_at) "
            "VALUES (?, ?, ?, ?, '2026-01-01')",
            rows,
        )
        conn.commit()
        conn.close()

    def test_tiny_bad_source_not_zero(self, db):
        sf = SignalFilter(db_path=db)
        self._stats(db, [("weak", 2, 0, 0.0)])
        src, acc = sf._worst_source_accuracy()
        assert src == "weak"
        assert acc == pytest.approx((0 + EB_PRIOR_STRENGTH * EB_PRIOR_MEAN)
                                    / (2 + EB_PRIOR_STRENGTH) * 100)

    def test_tiny_bad_source_does_not_trigger_warn(self, db):
        sf = SignalFilter(db_path=db)
        self._stats(db, [("weak", 2, 0, 0.0)])
        allowed, reason = sf.should_trade("NVDA", "BUY")
        assert allowed is True
        assert not reason.startswith("WARN")

    def test_large_bad_sample_still_warns(self, db):
        sf = SignalFilter(db_path=db)
        self._stats(db, [("bad", 50, 5, 0.10)])
        # shrunk = (5 + 2.5) / (50 + 5) = 13.6% < 25% -> WARN
        allowed, reason = sf.should_trade("NVDA", "BUY")
        assert allowed is True
        assert reason.startswith("WARN")
        assert "bad" in reason

    def test_empty_source_stats_no_warn(self, db):
        sf = SignalFilter(db_path=db)
        src_name, acc = sf._worst_source_accuracy()
        assert src_name == "unknown"
        assert acc == 50.0
        allowed, reason = sf.should_trade("NVDA", "BUY")
        assert allowed is True
        assert reason == "OK"
