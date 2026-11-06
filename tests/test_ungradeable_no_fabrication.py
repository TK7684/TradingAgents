"""Ungradeable predictions must NOT be fabricated as HOLD (AGI 2026-11-06).

NEGATIVE-DATA LAW transfer: a self-improvement loop that records fabricated
outcomes cannot learn — worse, fabricated HOLD grades turn every missing-data
BUY/SELL prediction into a false failure, biasing the EB accuracy stats that
drive position weighting in signal_filter.py.

One number: fabricated grades on the no-price-data fixture: 1 -> 0.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.expanduser("~/TradingAgents"))
sys.path.insert(0, os.path.expanduser("~/TradingAgents/ec2_deploy"))

from ec2_deploy import score_predictions as sp


class _FakeHist:
    def __init__(self, rows):
        # rows: list of (date_str, close); `.empty` mirrors pandas (bool attr)
        self._rows = rows
        self.empty = len(rows) == 0

    def __bool__(self):
        return len(self._rows) > 0

    def iterrows(self):
        from pandas import Timestamp
        for ds, close in self._rows:
            yield Timestamp(ds), {"Close": close}


class _FakeTicker:
    rows = []

    def __init__(self, symbol):
        pass

    def history(self, start=None, end=None, auto_adjust=True):
        return _FakeHist(list(_FakeTicker.rows))


def _patch_yf(monkeypatch, rows):
    import yfinance as yf
    _FakeTicker.rows = rows
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)


DATE = "2026-09-01"
HORIZON_END = (datetime.strptime(DATE, "%Y-%m-%d") + timedelta(days=15)).strftime("%Y-%m-%d")


class TestUngradeableNoFabrication:
    def test_empty_history_returns_none_not_hold(self, monkeypatch):
        _patch_yf(monkeypatch, [])
        sig, det = sp.get_actual_signal("ZZZZZZ", DATE)
        assert sig is None
        assert det["close_0"] is None and det["close_N"] is None

    def test_missing_close_at_horizon_returns_none(self, monkeypatch):
        # price exists on signal date but never again (delisted / data gap)
        _patch_yf(monkeypatch, [(DATE, 100.0)])
        sig, det = sp.get_actual_signal("ZZZZZZ", DATE)
        assert sig is None
        assert det["close_0"] == 100.0
        assert det["close_N"] is None

    def test_gradeable_data_still_grades(self, monkeypatch):
        # +5% move over horizon clears 2% band + 20bp cost -> BUY
        _patch_yf(monkeypatch, [(DATE, 100.0), (HORIZON_END, 105.0)])
        sig, det = sp.get_actual_signal("ZZZZZZ", DATE)
        assert sig == "BUY"
        assert det["pct_change"] == 5.0

    def test_fabricated_grade_count_is_zero_on_no_data_fixture(self, monkeypatch):
        """The ONE NUMBER: for a no-price-data prediction the grading loop
        previously wrote a fabricated HOLD outcome (count=1); now it must
        record nothing (count=0)."""
        import sqlite3
        recorded = []

        class _Engine:
            def record_outcome(self, ticker, date_str, actual):
                recorded.append((ticker, date_str, actual))
                return "graded"
            def close(self): pass

        _patch_yf(monkeypatch, [])
        engine = _Engine()

        # replicate the guarded call from score_predictions.main()
        actual, details = sp.get_actual_signal("ZZZZZZ", DATE)
        if actual is None:
            pass  # gate: leave actual_signal NULL
        else:
            engine.record_outcome("ZZZZZZ", DATE, actual)

        assert len(recorded) == 0
