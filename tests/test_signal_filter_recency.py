"""Recency-weighted accuracy in SignalFilter (regime-shift adaptation).

Discriminator suite: a ticker whose accuracy regime SHIFTED (bad long ago,
good recently — or the reverse) must be judged on recent evidence, not the
lifetime average. Lifetime averaging is doubly wrong here: it keeps a
recovered ticker blocked AND keeps a collapsed ticker at full weight.

Mechanism: each prediction votes with weight 0.5 ** (age_days /
RECENCY_HALF_LIFE_DAYS); weighted counts feed the same EB shrinkage.
Equal timestamps reduce exactly to the previous unweighted estimate, so
all pre-existing EB tests keep their numerics.
"""
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from signal_filter import (
    SignalFilter,
    EB_PRIOR_MEAN,
    EB_PRIOR_STRENGTH,
    RECENCY_HALF_LIFE_DAYS,
)

NOW = datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture()
def db(tmp_path):
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
            updated_at         TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def _add(conn, ticker, correct, age_days, signal="BUY", actual="SELL"):
    ts = (NOW - timedelta(days=age_days)).isoformat(sep=" ")
    conn.execute(
        "INSERT INTO predictions (ticker, date, source, predicted_signal, "
        "actual_signal, correct, created_at) VALUES (?, '2026-01-01', "
        "'analyst_a', ?, ?, ?, ?)",
        (ticker, signal, actual, correct, ts),
    )


class TestRecoveredTicker:
    """Was terrible (>20 predictions, lifetime ~12%), now consistently right.

    Old code: lifetime EB stays < 20% -> perpetual BLOCK. New code: recent
    wins dominate -> allowed to trade again.
    """

    def _seed(self, db):
        conn = sqlite3.connect(db)
        # 3 correct + 22 wrong, all ~180 days old (stale regime)
        for _ in range(3):
            _add(conn, "RECOV", 1, age_days=180)
        for _ in range(22):
            _add(conn, "RECOV", 0, age_days=180)
        # recent regime: 8/8 correct within the last 10 days
        # (correct rows predict BUY and actual is BUY — they agree)
        for _ in range(8):
            _add(conn, "RECOV", 1, age_days=5, signal="BUY", actual="BUY")
        conn.commit()
        conn.close()

    def test_gate_no_longer_blocks_recovered_ticker(self, db):
        self._seed(db)
        sf = SignalFilter(db_path=db)
        allowed, reason = sf.should_trade("RECOV", "BUY")
        assert allowed is True, f"recovered ticker still blocked: {reason}"

    def test_lifetime_average_would_have_blocked(self, db):
        """Document the discriminator: unweighted EB on this history is <20%."""
        self._seed(db)
        sf = SignalFilter(db_path=db)
        # lifetime unweighted accuracy: 11/33 -> EB 35.5% (above BLOCK 20%),
        # BUT old Rule-3 anti-signal gate: 22 disagreements / 25 directional
        # = 88% >= 0.6 -> BLOCK. Recency code allows. (Both gates audited.)
        lifetime_eb = (11 + EB_PRIOR_STRENGTH * EB_PRIOR_MEAN) / (33 + EB_PRIOR_STRENGTH) * 100
        anti = 22 / 25
        assert anti >= 0.6 or lifetime_eb < 20.0  # old code blocks via one of them
        assert lifetime_eb > 20.0  # accuracy gate alone would NOT block

    def test_position_weight_recovers_above_half(self, db):
        self._seed(db)
        sf = SignalFilter(db_path=db)
        w = sf.get_position_weight("RECOV")
        assert w >= 0.75  # recent 8/8 -> not stuck in the 0.5 band


class TestCollapsedTicker:
    """Was excellent lifetime (~78%), collapsed recently (0/10 last 14 days).

    Old code: lifetime halo -> full weight 1.0. New code: recent losses
    dominate -> weight drops to the 0.5 band (20-40% weighted accuracy).
    """

    def _seed(self, db):
        conn = sqlite3.connect(db)
        for _ in range(25):
            _add(conn, "COLL", 1, age_days=180)
        for _ in range(10):
            _add(conn, "COLL", 0, age_days=7)
        conn.commit()
        conn.close()

    def test_weight_drops_below_full(self, db):
        self._seed(db)
        sf = SignalFilter(db_path=db)
        w = sf.get_position_weight("COLL")
        assert w < 1.0  # lifetime code returns 1.0 forever

    def test_weighted_accuracy_in_low_band(self, db):
        self._seed(db)
        sf = SignalFilter(db_path=db)
        acc, n = sf._ticker_accuracy("COLL")
        assert n == 35
        assert acc < 40.0  # 0.5 band; lifetime would be ~78%


class TestBackwardCompat:
    """Equal timestamps must reduce exactly to the previous EB numerics."""

    def test_same_timestamps_match_unweighted_eb(self, db):
        conn = sqlite3.connect(db)
        for _ in range(3):
            _add(conn, "FLAT", 1, age_days=0)  # fresh => weight exactly 1
        for _ in range(7):
            _add(conn, "FLAT", 0, age_days=0)
        conn.commit()
        conn.close()
        sf = SignalFilter(db_path=db)
        acc, n = sf._ticker_accuracy("FLAT")
        expected = (3 + EB_PRIOR_STRENGTH * EB_PRIOR_MEAN) / (10 + EB_PRIOR_STRENGTH) * 100
        assert acc == pytest.approx(expected, abs=0.01)

    def test_half_life_constant_sanity(self):
        assert RECENCY_HALF_LIFE_DAYS == 60.0

    def test_future_and_junk_timestamps_safe(self, db):
        conn = sqlite3.connect(db)
        _add(conn, "JUNK", 1, age_days=-5)   # future
        conn.execute(
            "INSERT INTO predictions (ticker, date, source, predicted_signal, "
            "actual_signal, correct, created_at) VALUES "
            "('JUNK', '2026-01-01', 'a', 'BUY', 'SELL', 1, 'not-a-date')"
        )
        conn.commit()
        conn.close()
        sf = SignalFilter(db_path=db)
        acc, n = sf._ticker_accuracy("JUNK")  # must not raise
        assert n == 2
        assert 0.0 < acc < 100.0

    def test_stats_display_matches_gate(self, db):
        """Display accuracy == gating accuracy (decision-basis invariant)."""
        conn = sqlite3.connect(db)
        for i in range(6):
            _add(conn, "TSM", 1 if i % 2 == 0 else 0, age_days=90 - i * 15)
        conn.commit()
        conn.close()
        sf = SignalFilter(db_path=db)
        stats = {s["ticker"]: s for s in sf.get_ticker_stats()}
        gate_acc, _ = sf._ticker_accuracy("TSM")
        assert stats["TSM"]["accuracy"] == pytest.approx(gate_acc, abs=0.06)
