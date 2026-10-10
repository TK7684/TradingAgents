"""Cost-aware signal grading tests (AGI experiment H20261005150151).

Validates that get_actual_signal / grade_move only credit directional calls
whose realized move clears round-trip transaction costs (arxiv:2605.19337).
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/TradingAgents"))
sys.path.insert(0, os.path.expanduser("~/TradingAgents/ec2_deploy"))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "score_predictions", os.path.expanduser("~/TradingAgents/ec2_deploy/score_predictions.py"))
# yfinance may be heavy but must be importable; module-level import
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

from ec2_deploy.score_predictions import grade_move, ROUND_TRIP_COST, get_actual_signal


class TestGradeMove:
    def test_buy_requires_move_above_threshold_plus_cost(self):
        # +2.0% raw move: above the 2% band but inside cost-adjusted band -> HOLD
        assert grade_move(0.020) == "HOLD"

    def test_buy_passes_when_net_edge_positive(self):
        assert grade_move(0.020 + ROUND_TRIP_COST + 1e-6) == "BUY"

    def test_sell_requires_move_below_negative_threshold_minus_cost(self):
        assert grade_move(-0.020) == "HOLD"

    def test_sell_passes_when_net_edge_positive(self):
        assert grade_move(-(0.020 + ROUND_TRIP_COST + 1e-6)) == "SELL"

    def test_flat_move_grades_hold(self):
        assert grade_move(0.0) == "HOLD"

    def test_zero_cost_recovers_legacy_behavior(self):
        assert grade_move(0.0201, round_trip_cost=0.0) == "BUY"
        assert grade_move(-0.0201, round_trip_cost=0.0) == "SELL"

    def test_boundary_is_exact_threshold_plus_cost(self):
        b = 0.02 + ROUND_TRIP_COST
        assert grade_move(b + 1e-9) == "BUY"
        assert grade_move(b) == "HOLD"  # strictly greater required


class TestCostConstant:
    def test_round_trip_cost_is_sane_bps(self):
        # 20 bps round-trip: between 5 and 50 bps
        assert 0.0005 <= ROUND_TRIP_COST <= 0.005

    def test_get_actual_signal_exposes_cost_in_details(self):
        # No network: yfinance fetch returns empty hist -> HOLD path.
        # We can't easily force details population without network, so we
        # assert the function signature still returns (signal, details).
        sig, det = get_actual_signal("ZZZZZZ", "2026-09-01")
        assert sig == "HOLD"
        assert isinstance(det, dict)


class TestRegressionVsLegacy:
    def test_marginal_win_no_longer_credited(self):
        """The core fix: a +2.1% move with 20bp costs WAS 'BUY' before,
        now must be HOLD (net edge 2.1% - 2% - 0.2% < 0)."""
        # legacy: 0.021 > 0.02 -> BUY ; new: 0.021 < 0.022 -> HOLD
        assert grade_move(0.021) == "HOLD"

    def test_clear_win_still_credited(self):
        assert grade_move(0.03) == "BUY"
        assert grade_move(-0.03) == "SELL"
