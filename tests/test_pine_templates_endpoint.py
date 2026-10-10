"""Tests for the /pine-templates endpoints (dormant library activation).

tradingagents/tv_screener/pine_templates.py (5 Pine Script alert strategies,
297 lines) had ZERO importers before this change -- it was written to feed the
MT5 webhook receiver but never wired in. These tests pin the activation seam.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from tradingagents.mt5_bridge.webhook_receiver import create_webhook_app


class TestPineTemplatesEndpoint:
    def setup_method(self):
        self.client = create_webhook_app(queue=[]).test_client()

    def test_index_lists_all_five_templates(self):
        r = self.client.get("/pine-templates")
        assert r.status_code == 200
        body = r.get_json()
        assert body["count"] == 5
        assert set(body["templates"]) == {
            "ema_crossover", "rsi_bb_squeeze", "volume_breakout",
            "multi_confluence", "crypto_swing",
        }
        # every entry has a description and nonzero source length
        for meta in body["templates"].values():
            assert meta["description"]
            assert meta["length"] > 100

    def test_get_returns_full_pine_source(self):
        r = self.client.get("/pine-templates/ema_crossover")
        assert r.status_code == 200
        body = r.get_json()
        assert body["name"] == "ema_crossover"
        pine = body["pine"]
        assert pine.lstrip().startswith("//@version=5")
        # webhook contract: alert JSON must target this receiver's parser
        assert '"symbol"' in pine and '"side"' in pine

    def test_unknown_template_fails_closed_404(self):
        r = self.client.get("/pine-templates/does_not_exist")
        assert r.status_code == 404
        body = r.get_json()
        assert body["error"] == "unknown template"
        assert "ema_crossover" in body["available"]

    def test_existing_endpoints_untouched(self):
        assert self.client.get("/health").status_code == 200
        assert self.client.get("/signals").status_code == 200
