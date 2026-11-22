"""Tests for TradingAgentsGraph._evidence_gate (H20261122-evidence-gate).

Transfer of H20261121150120 ("route by evidence-backed confidence, not
static priors", proven in hermes agi_system tooling) to the main trading
graph: the LLM's directional signal is multiplied by the ticker's
EB-shrunk realized-accuracy ledger before release. BLOCK -> HOLD,
WARN -> pass-through, ledger failure -> fail-open.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from signal_filter import SignalFilter  # noqa: E402


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


SCHEMA = """
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            predicted_signal TEXT NOT NULL,
            actual_signal    TEXT,
            correct     INTEGER,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        , pct_1d REAL, pct_3d REAL, pct_5d REAL);
        CREATE TABLE IF NOT EXISTS source_stats (
            source             TEXT PRIMARY KEY,
            total_predictions  INTEGER NOT NULL DEFAULT 0,
            correct_predictions INTEGER NOT NULL DEFAULT 0,
            accuracy           REAL NOT NULL DEFAULT 0.0,
            updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """


def _bad_ticker_db(db, ticker="BAD"):
    """25 predictions at 12% accuracy -> BLOCK (below 20%, n >= 20)."""
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    for i in range(25):
        correct = 1 if i < 3 else 0
        conn.execute(
            "INSERT INTO predictions (ticker, date, source, predicted_signal, "
            "actual_signal, correct) VALUES (?, date('now', ?), 'analyst_a', "
            "'BUY', 'SELL', ?)",
            (ticker, f"-{i} days", correct),
        )
    conn.commit()
    conn.close()
    return db


class _GateMixin:
    @staticmethod
    def _graph():
        """TradingAgentsGraph is heavy (LLM keys); only the gate method is
        under test, so bind it unbound-style onto a bare object."""
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        obj = object.__new__(TradingAgentsGraph)
        return obj


class TestEvidenceGate(_GateMixin):
    def test_block_downgrades_directional_to_hold(self, tmp_path):
        db = _bad_ticker_db(str(tmp_path / "sadb.db"))
        sf = SignalFilter(db_path=db)
        allowed, reason = sf.should_trade("BAD", "BUY")
        assert not allowed, reason
        gate = self._graph()._evidence_gate
        assert gate("BAD", "BUY", signal_filter=sf) == "HOLD"
        assert gate("BAD", "OVERWEIGHT", signal_filter=sf) == "HOLD"
        assert gate("BAD", "SELL", signal_filter=sf) == "HOLD"

    def test_clean_ticker_passes_directional_signal(self, db):
        sf = SignalFilter(db_path=db)  # empty ledger -> no block
        gate = self._graph()._evidence_gate
        assert gate("NVDA", "BUY", signal_filter=sf) == "BUY"
        assert gate("NVDA", "SELL", signal_filter=sf) == "SELL"

    def test_hold_passes_through_untouched(self, db):
        sf = SignalFilter(db_path=db)
        assert self._graph()._evidence_gate("NVDA", "HOLD", signal_filter=sf) == "HOLD"

    def test_ledger_error_fails_open(self):
        class ExplodingFilter:
            def should_trade(self, ticker, signal):
                raise sqlite3.OperationalError("no table: predictions")

        gate = self._graph()._evidence_gate
        assert gate("NVDA", "BUY", signal_filter=ExplodingFilter()) == "BUY"

    def test_real_production_ledger_resolves(self):
        """Default path: the production ledger opens and the gate returns a
        valid signal for a real ticker (fail-open if ledger missing)."""
        gate = self._graph()._evidence_gate
        out = gate("NVDA", "BUY")
        assert out in ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL")
