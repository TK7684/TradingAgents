"""Tests for MT5 Bridge components -- webhook receiver, signal parser, pipeline."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from threading import Thread

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


# ============================================================================
# Signal Validator Tests
# ============================================================================

class TestTradingViewSignalParser:
    """Test TradingView signal parsing and validation."""

    def setup_method(self):
        from tradingagents.mt5_bridge.signal_validator import TradingViewSignalParser
        self.parser = TradingViewSignalParser()

    def test_parse_json_buy_signal(self):
        signal = self.parser.parse_json({
            "symbol": "EURUSD", "side": "buy", "price": 1.0850,
            "strategy": "RSI_Divergence", "timeframe": "1H",
        })
        assert signal is not None
        assert signal.symbol == "EURUSD"
        assert signal.side == "BUY"
        assert signal.price == 1.0850
        assert signal.strategy == "RSI_Divergence"

    def test_parse_json_sell_signal(self):
        signal = self.parser.parse_json({
            "symbol": "XAUUSD", "action": "sell", "price": 2345.50,
            "sl": 2360.0, "tp": 2300.0, "lot": 0.1, "strategy": "MACD_Cross",
        })
        assert signal is not None
        assert signal.symbol == "XAUUSD"
        assert signal.side == "SELL"
        assert signal.sl == 2360.0
        assert signal.tp == 2300.0
        assert signal.lot == 0.1

    def test_parse_json_tradingview_template(self):
        signal = self.parser.parse_json({
            "ticker": "FX:EURUSD", "exchange": "FX",
            "action": "buy", "price": "1.0850", "interval": "60",
            "strategy": "MyStrategy", "volume": "12345",
        })
        assert signal is not None
        assert signal.symbol == "EURUSD"
        assert signal.side == "BUY"

    def test_parse_json_unknown_side(self):
        signal = self.parser.parse_json({
            "symbol": "EURUSD", "action": "hold", "price": 1.0850,
        })
        assert signal is None

    def test_parse_json_broker_prefixes(self):
        for prefix, expected in [
            ("NASDAQ:AAPL", "AAPL"),
            ("BINANCE:BTCUSDT", "BTCUSDT"),
            ("COINBASE:ETHUSD", "ETHUSD"),
        ]:
            signal = self.parser.parse_json({"ticker": prefix, "action": "buy"})
            assert signal is not None, f"Failed for {prefix}"
            assert signal.symbol == expected

    def test_parse_text_buy(self):
        signal = self.parser.parse_text("BUY EURUSD @ 1.0850 SL=30 TP=60")
        assert signal is not None
        assert signal.side == "BUY"
        assert signal.symbol == "EURUSD"

    def test_parse_text_long(self):
        signal = self.parser.parse_text("Long XAUUSD sl=2360 tp=2300 lot=0.1")
        assert signal is not None
        assert signal.side == "BUY"
        assert signal.symbol == "XAUUSD"

    def test_parse_text_sell(self):
        signal = self.parser.parse_text("SELL GBPUSD @ 1.2700")
        assert signal is not None
        assert signal.side == "SELL"
        assert signal.symbol == "GBPUSD"

    def test_parse_text_close(self):
        signal = self.parser.parse_text("EXIT EURUSD")
        assert signal is not None
        assert signal.side == "CLOSE"

    def test_parse_text_empty(self):
        assert self.parser.parse_text("") is None
        assert self.parser.parse_text("   ") is None

    def test_parse_text_no_action(self):
        assert self.parser.parse_text("EURUSD is at 1.0850") is None

    def test_validate_valid_signal(self):
        signal = self.parser.parse_json({
            "symbol": "EURUSD", "side": "buy", "price": 1.0850,
        })
        is_valid, reason = self.parser.validate(signal)
        assert is_valid
        assert reason == "valid"

    def test_validate_invalid_side(self):
        from tradingagents.mt5_bridge.signal_validator import TradingSignal
        signal = TradingSignal(
            symbol="EURUSD", side="UNKNOWN", price=1.0850,
            sl=None, tp=None, lot=None, strategy="test", timeframe="",
        )
        is_valid, reason = self.parser.validate(signal)
        assert not is_valid
        assert "invalid side" in reason


# ============================================================================
# Webhook Receiver Tests
# ============================================================================

class TestWebhookReceiver:
    """Test Flask webhook receiver endpoints."""

    def setup_method(self):
        self.queue = []
        from tradingagents.mt5_bridge.webhook_receiver import create_webhook_app
        self.app = create_webhook_app(queue=self.queue)
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "queue_size" in data

    def test_webhook_valid_json(self):
        resp = self.client.post(
            "/webhook/tradingview",
            json={"symbol": "EURUSD", "side": "buy", "price": 1.0850, "strategy": "test"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "queued"
        assert len(self.queue) == 1

    def test_webhook_empty_body(self):
        resp = self.client.post("/webhook/tradingview", data="", content_type="application/json")
        assert resp.status_code == 400

    def test_webhook_tradingview_template_format(self):
        resp = self.client.post("/webhook/tradingview", json={
            "ticker": "FX:EURUSD", "exchange": "FX",
            "action": "{{strategy.order.action}}",
            "price": "{{close}}", "interval": "{{interval}}",
            "volume": "{{volume}}",
        })
        assert resp.status_code == 200

    def test_webhook_with_secret_valid(self):
        from tradingagents.mt5_bridge.webhook_receiver import create_webhook_app
        import hashlib, hmac

        secret_queue = []
        app2 = create_webhook_app(queue=secret_queue, secret="test-secret")
        client2 = app2.test_client()

        body = json.dumps({"symbol": "EURUSD", "side": "buy"}).encode()
        sig = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()

        resp = client2.post(
            "/webhook/tradingview", data=body,
            content_type="application/json",
            headers={"X-Webhook-Signature": sig},
        )
        assert resp.status_code == 200
        assert len(secret_queue) == 1

    def test_webhook_with_secret_invalid(self):
        from tradingagents.mt5_bridge.webhook_receiver import create_webhook_app

        secret_queue = []
        app2 = create_webhook_app(queue=secret_queue, secret="test-secret")
        client2 = app2.test_client()

        body = json.dumps({"symbol": "EURUSD", "side": "buy"}).encode()
        resp = client2.post(
            "/webhook/tradingview", data=body,
            content_type="application/json",
            headers={"X-Webhook-Signature": "wrong-signature"},
        )
        assert resp.status_code == 403

    def test_list_signals(self):
        self.queue.append({"symbol": "TEST", "side": "BUY"})
        resp = self.client.get("/signals")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1

    def test_clear_signals(self):
        self.queue.append({"test": True})
        resp = self.client.post("/signals/clear")
        assert resp.status_code == 200
        assert len(self.queue) == 0

    def test_pop_signal(self):
        self.queue.append({"symbol": "EURUSD", "side": "buy"})
        resp = self.client.post("/signals/next")
        assert resp.status_code == 200
        assert len(self.queue) == 0

    def test_pop_empty_queue(self):
        resp = self.client.post("/signals/next")
        assert resp.status_code == 204

    def test_queue_max_size(self):
        from tradingagents.mt5_bridge.webhook_receiver import MAX_QUEUE_SIZE
        self.queue.extend([{"i": i} for i in range(MAX_QUEUE_SIZE)])
        resp = self.client.post("/webhook/tradingview", json={"symbol": "NEW", "side": "buy"})
        assert resp.status_code == 200
        assert len(self.queue) == MAX_QUEUE_SIZE
        assert self.queue[-1]["symbol"] == "NEW"


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestTradingPipeline:
    """Test trading pipeline (paper mode)."""

    def setup_method(self):
        from tradingagents.mt5_bridge.pipeline import TradingPipeline
        self.data_dir = tempfile.mkdtemp()
        self.pipeline = TradingPipeline(
            mode="paper", signal_queue=[], data_dir=self.data_dir,
        )

    def test_process_valid_buy_signal(self):
        self.pipeline._queue.append({
            "symbol": "EURUSD", "side": "buy", "price": 1.0850, "strategy": "test",
        })
        record = self.pipeline.process_next()
        assert record is not None
        assert record.decision == "BUY"
        assert record.mode == "paper"

    def test_process_sell_signal(self):
        self.pipeline._queue.append({
            "symbol": "XAUUSD", "action": "sell", "price": 2345.0, "strategy": "test",
        })
        record = self.pipeline.process_next()
        assert record is not None
        assert record.decision == "SELL"

    def test_process_invalid_signal(self):
        self.pipeline._queue.append({"garbage": True})
        record = self.pipeline.process_next()
        assert record is not None
        assert record.decision == "REJECTED"

    def test_process_empty_queue(self):
        assert self.pipeline.process_next() is None

    def test_trade_logging(self):
        self.pipeline._queue.append({
            "symbol": "EURUSD", "side": "buy", "price": 1.0850,
        })
        self.pipeline.process_next()

        log_file = Path(self.data_dir) / "bridge_trades.jsonl"
        assert log_file.exists()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["decision"] == "BUY"
        assert record["mode"] == "paper"
        assert "id" in record
        assert "timestamp" in record

    def test_stats_tracking(self):
        self.pipeline._queue.append({"symbol": "EURUSD", "side": "buy", "price": 1.0850})
        self.pipeline.process_next()
        self.pipeline._queue.append({"bad": "data"})
        self.pipeline.process_next()

        stats = self.pipeline.stats
        assert stats["total_processed"] == 2
        assert stats["total_executed"] == 1
        assert stats["total_rejected"] == 1

    def test_daily_trade_limit(self):
        self.pipeline._max_daily_trades = 2
        for _ in range(5):
            self.pipeline._queue.append({"symbol": "EURUSD", "side": "buy", "price": 1.0850})

        results = []
        for _ in range(5):
            r = self.pipeline.process_next()
            if r:
                results.append(r.decision)

        buy_count = sum(1 for d in results if d == "BUY")
        rejected_count = sum(1 for d in results if d == "REJECTED")
        assert buy_count <= 2
        assert rejected_count >= 3

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.data_dir, ignore_errors=True)


# ============================================================================
# MT5 HTTP Executor Tests
# ============================================================================

class TestMT5HttpExecutor:
    """Test MT5 HTTP executor with mocked HTTP requests."""

    def test_not_connected_by_default(self):
        from tradingagents.mt5_bridge.mt5_executor import MT5HttpExecutor
        executor = MT5HttpExecutor()
        assert not executor.is_connected

    def test_connect_success(self):
        from tradingagents.mt5_bridge.mt5_executor import MT5HttpExecutor
        executor = MT5HttpExecutor(bridge_url="http://127.0.0.1:5002")

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "success": True, "account": 12345, "balance": 10000.0,
            "currency": "USD", "server": "Broker-Demo",
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = executor.connect()
            assert result is True
            assert executor.is_connected

    def test_connect_failure(self):
        from tradingagents.mt5_bridge.mt5_executor import MT5HttpExecutor
        executor = MT5HttpExecutor(bridge_url="http://127.0.0.1:5002")

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "error": "connection refused",
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = executor.connect()
            assert result is False

    def test_market_buy_not_connected(self):
        from tradingagents.mt5_bridge.mt5_executor import MT5HttpExecutor
        executor = MT5HttpExecutor()
        result = executor.market_buy("EURUSD")
        assert not result.success
        assert "not connected" in result.comment

    def test_market_buy_success(self):
        from tradingagents.mt5_bridge.mt5_executor import MT5HttpExecutor
        executor = MT5HttpExecutor()
        executor._connected = True

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "success": True, "ticket": 123456, "volume": 0.01,
            "price": 1.0850, "comment": "",
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = executor.market_buy("EURUSD", lot=0.01)
            assert result.success
            assert result.ticket == 123456

    def test_market_sell_success(self):
        from tradingagents.mt5_bridge.mt5_executor import MT5HttpExecutor
        executor = MT5HttpExecutor()
        executor._connected = True

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "success": True, "ticket": 123457, "volume": 0.02,
            "price": 1.0840, "comment": "",
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = executor.market_sell("EURUSD", lot=0.02)
            assert result.success

    def test_health(self):
        from tradingagents.mt5_bridge.mt5_executor import MT5HttpExecutor
        executor = MT5HttpExecutor()

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok", "mt5_connected": True,
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = executor.health()
            assert result["status"] == "ok"


# ============================================================================
# Full Integration Test (paper mode)
# ============================================================================

class TestFullIntegration:
    """End-to-end integration test: webhook -> pipeline (paper)."""

    def test_full_pipeline_paper(self):
        from tradingagents.mt5_bridge.webhook_receiver import create_webhook_app
        from tradingagents.mt5_bridge.pipeline import TradingPipeline

        queue = []
        app = create_webhook_app(queue=queue)
        client = app.test_client()

        data_dir = tempfile.mkdtemp()
        try:
            pipeline = TradingPipeline(mode="paper", signal_queue=queue, data_dir=data_dir)

            # Health
            r = client.get("/health")
            assert r.status_code == 200

            # Send signal
            r = client.post("/webhook/tradingview", json={
                "symbol": "XAUUSD", "side": "buy", "price": 2345.50,
                "sl": 2320.0, "tp": 2380.0, "lot": 0.05,
                "strategy": "RSI_Divergence_M15", "timeframe": "15",
            })
            assert r.status_code == 200
            assert len(queue) == 1

            # Process
            record = pipeline.process_next()
            assert record is not None
            assert record.decision == "BUY"
            assert record.mode == "paper"
            assert pipeline.stats["total_executed"] == 1

            # Queue empty now
            assert pipeline.process_next() is None
        finally:
            import shutil
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
