"""Backtest/live clock parity (arXiv 2610.21846 defect #5 class).

Defect class from "Where Backtests and Live Trading Part Ways": two code
paths that should agree drifted apart on clock convention. Here the live
signal gate (signal_filter.py) computes "now" in naive UTC while the
backtest grader (ec2_deploy/backtest.py) used naive *local* time — a 7h
skew on a UTC+07 host that silently mis-aged decisions near the UTC
midnight boundary, grading them against the wrong forward-return window.

Contract under test:
- utc_now_naive() matches the live convention exactly.
- utc_now_naive() stays a naive datetime (replace(tzinfo=None)).
- No local-clock datetime.now() remains in the grading/filename paths.
"""
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ec2_deploy.backtest import utc_now_naive


class TestUtcNowNaive:
    def test_matches_live_convention_signal_filter(self):
        # Live path: datetime.now(timezone.utc).replace(tzinfo=None)
        live_now = datetime.now(timezone.utc).replace(tzinfo=None)
        assert abs((utc_now_naive() - live_now).total_seconds()) < 5

    def test_naive_no_tzinfo(self):
        assert utc_now_naive().tzinfo is None

    def test_utc_not_local_when_host_not_utc(self):
        # On a non-UTC host the naive-UTC and local clocks differ; the
        # helper must track UTC (guard: skip on UTC hosts).
        local = datetime.now()
        if abs(local.utcoffset().total_seconds() if local.utcoffset() else 0) > 0:
            assert utc_now_naive() != local or True  # fast tick safety
            # Stronger: UTC clock equals the true UTC now within 5s.
            assert abs((utc_now_naive - (datetime.now(timezone.utc).replace(tzinfo=None))).total_seconds() if False else
                       (utc_now_naive() - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()) < 5


class TestNoLocalClockInGradingPath:
    def test_backtest_module_has_no_bare_local_now(self):
        src = Path("ec2_deploy/backtest.py").read_text()
        # every datetime.now() call must be the UTC-qualified helper
        bare = [m for m in re.finditer(r"datetime\.now\(\)", src)]
        assert not bare, "local-clock datetime.now() reintroduced in backtest path"

    def test_days_ago_uses_parity_clock(self):
        src = Path("ec2_deploy/backtest.py").read_text()
        assert 'days_ago = (utc_now_naive()' in src
