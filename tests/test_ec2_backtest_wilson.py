"""Tests for ec2_deploy/backtest.py Wilson-LB accuracy propagation (pattern #18).

Raw ``correct/total`` overrates small samples: a 1/1 ticker displays as 100%
green and flips the Discord/dashboard color gates. The proven Wilson score
lower bound (consensus.py) shrinks small samples toward 50% while leaving
large samples almost untouched. Verifies the helper contract, the shrinkage
behaviour at boundary sample sizes, and the shared-gate consistency between
this script and the consensus module.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ec2_deploy.backtest import wilson_lb_pct, score_decision
from tradingagents.graph.consensus import _wilson_lower_bound


class TestWilsonLbPct:
    def test_returns_percent_and_rounded(self):
        assert wilson_lb_pct(8, 10) == round(_wilson_lower_bound(8, 10) * 100, 1)

    def test_one_of_one_is_not_100(self):
        # The headline fix: 1/1 must not read as a proven 100% source.
        assert wilson_lb_pct(1, 1) < 50.0

    def test_zero_total_is_neutral_prior(self):
        assert wilson_lb_pct(0, 0) == 50.0

    def test_zero_correct_positive_total(self):
        assert wilson_lb_pct(0, 10) == 0.0

    def test_perfect_large_sample_stays_high(self):
        # 100/100 stays ~96.3% - shrinkage must not punish proven sources.
        assert 95.0 <= wilson_lb_pct(100, 100) <= 97.0

    def test_monotone_in_correct_for_fixed_n(self):
        vals = [wilson_lb_pct(c, 20) for c in range(21)]
        assert vals == sorted(vals)

    def test_shrinks_more_for_smaller_samples(self):
        # 50% at n=2 shrinks harder than 50% at n=100.
        assert wilson_lb_pct(1, 2) < wilson_lb_pct(50, 100)

    def test_matches_consensus_convention(self):
        # Same numbers must flow into weights and display (pattern #20).
        for c, n in [(1, 1), (3, 4), (8, 10), (47, 50), (0, 7)]:
            assert wilson_lb_pct(c, n) == round(_wilson_lower_bound(c, n) * 100, 1)


class TestScoreDecisionUnchanged:
    """Regression guard: the scoring rubric itself must not change."""

    def test_buy_up_is_correct(self):
        assert score_decision("BUY", {5: 2.0}) == "correct"

    def test_buy_down_is_wrong(self):
        assert score_decision("BUY", {5: -3.0}) == "wrong"

    def test_sell_down_is_correct(self):
        assert score_decision("SELL", {5: -2.0}) == "correct"

    def test_hold_flat_is_correct(self):
        assert score_decision("HOLD", {5: 0.5}) == "correct"

    def test_empty_returns_unknown(self):
        assert score_decision("BUY", {}) == "unknown"
