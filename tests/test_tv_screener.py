"""Tests for TradingView Premium Screener integration."""

import json
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

import pandas as pd


class TestMoverResult(unittest.TestCase):
    """Test the MoverResult dataclass."""

    def setUp(self):
        from tradingagents.tv_screener.client import MoverResult
        self.mover = MoverResult(
            ticker="NASDAQ:NVDA",
            name="NVIDIA Corp",
            close=222.82,
            change_pct=3.5,
            volume=50_000_000,
            market_cap=5_000_000_000_000,
            rsi=65.2,
            recommendation=0.75,
        )

    def test_exchange_extraction(self):
        self.assertEqual(self.mover.exchange, "NASDAQ")

    def test_symbol_extraction(self):
        self.assertEqual(self.mover.symbol, "NVDA")

    def test_recommendation_labels(self):
        self.assertEqual(self.mover.recommendation_label, "BUY")

    def test_strong_buy(self):
        from tradingagents.tv_screener.client import MoverResult
        m = MoverResult("X:Y", "Y", 100, 5, 1e6, 1e9, 50, 0.85)
        self.assertEqual(m.recommendation_label, "STRONG_BUY")

    def test_hold(self):
        from tradingagents.tv_screener.client import MoverResult
        m = MoverResult("X:Y", "Y", 100, 0.5, 1e6, 1e9, 50, 0.0)
        self.assertEqual(m.recommendation_label, "HOLD")

    def test_sell(self):
        from tradingagents.tv_screener.client import MoverResult
        m = MoverResult("X:Y", "Y", 100, -3, 1e6, 1e9, 50, -0.5)
        self.assertEqual(m.recommendation_label, "SELL")

    def test_to_dict(self):
        d = self.mover.to_dict()
        self.assertEqual(d["symbol"], "NVDA")
        self.assertEqual(d["exchange"], "NASDAQ")
        self.assertIn("close", d)
        self.assertIn("recommendation_label", d)


class TestCapParsing(unittest.TestCase):
    """Test market cap string parsing."""

    def setUp(self):
        from tradingagents.tv_screener.client import TVScreener
        self.tv = TVScreener.__new__(TVScreener)

    def test_parse_billion(self):
        self.assertEqual(self.tv._parse_cap("10B"), 10_000_000_000)

    def test_parse_million(self):
        self.assertEqual(self.tv._parse_cap("500M"), 500_000_000)

    def test_parse_trillion(self):
        self.assertEqual(self.tv._parse_cap("1T"), 1_000_000_000_000)

    def test_parse_k(self):
        self.assertEqual(self.tv._parse_cap("100K"), 100_000)

    def test_parse_number(self):
        self.assertEqual(self.tv._parse_cap("12345"), 12345.0)

    def test_parse_invalid(self):
        self.assertEqual(self.tv._parse_cap("abc"), 1_000_000_000)  # default


class TestDFToMovers(unittest.TestCase):
    """Test DataFrame to MoverResult conversion."""

    def setUp(self):
        from tradingagents.tv_screener.client import TVScreener
        self.tv = TVScreener.__new__(TVScreener)

    def test_basic_conversion(self):
        df = pd.DataFrame([
            {
                "ticker": "NASDAQ:NVDA",
                "name": "NVIDIA",
                "close": 222.82,
                "change": 3.5,
                "volume": 50_000_000,
                "market_cap_basic": 5e12,
                "RSI": 65.2,
                "Recommend.All": 0.75,
            }
        ])
        movers = self.tv._df_to_movers(df)
        self.assertEqual(len(movers), 1)
        self.assertEqual(movers[0].symbol, "NVDA")

    def test_with_string_fields(self):
        df = pd.DataFrame([
            {
                "ticker": "NASDAQ:AAPL",
                "name": "Apple",
                "close": 315.20,
                "change": 2.9,
                "volume": 30_000_000,
                "market_cap_basic": 3e12,
                "RSI": 73.5,
                "Recommend.All": 0.6,
                "exchange": "NASDAQ",
                "submarket": "",
            }
        ])
        movers = self.tv._df_to_movers(df)
        self.assertEqual(len(movers), 1)
        self.assertEqual(movers[0].extra["exchange"], "NASDAQ")

    def test_nan_rsi(self):
        df = pd.DataFrame([
            {
                "ticker": "NASDAQ:XYZ",
                "name": "XYZ",
                "close": 100.0,
                "change": 1.0,
                "volume": 1_000_000,
                "market_cap_basic": 1e9,
                "RSI": float("nan"),
                "Recommend.All": float("nan"),
            }
        ])
        movers = self.tv._df_to_movers(df)
        self.assertEqual(len(movers), 1)
        self.assertEqual(movers[0].rsi, 0)
        self.assertEqual(movers[0].recommendation, 0)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["ticker", "name", "close", "change"])
        movers = self.tv._df_to_movers(df)
        self.assertEqual(len(movers), 0)


class TestDiscordFormatting(unittest.TestCase):
    """Test Discord briefing format."""

    def test_format_with_data(self):
        from tradingagents.tv_screener.briefing import format_discord_briefing

        briefing = {
            "market_breadth": {
                "total_scanned": 100,
                "bullish": 60,
                "bearish": 30,
                "neutral": 10,
                "bull_bear_ratio": 2.0,
                "avg_rsi": 55.0,
                "avg_change": 0.5,
                "gainers": 65,
                "losers": 35,
            },
            "top_gainers": [
                {
                    "symbol": "MRVL",
                    "name": "Marvell",
                    "close": 290.79,
                    "change_pct": 32.52,
                    "rsi": 85.88,
                    "recommendation_label": "BUY",
                }
            ],
            "top_losers": [],
            "oversold": [],
            "overbought": [],
            "volume_breakouts": [],
            "macd_bullish": [],
            "watchlist_data": {
                "NVDA": {"close": 222.82, "change": -0.69, "RSI": 58.79, "Recommend.All": 0.5, "volume": 50000000},
                "AAPL": {"close": 315.20, "change": 2.90, "RSI": 73.59, "Recommend.All": 0.65, "volume": 30000000},
            },
        }

        output = format_discord_briefing(briefing)
        self.assertIn("MRVL", output)
        self.assertIn("NVDA", output)
        self.assertIn("AAPL", output)
        self.assertIn("BULLISH", output)

    def test_empty_briefing(self):
        from tradingagents.tv_screener.briefing import format_discord_briefing
        output = format_discord_briefing({
            "market_breadth": {"total_scanned": 0, "bullish": 0, "bearish": 0,
                              "neutral": 0, "bull_bear_ratio": 0, "avg_rsi": 0,
                              "avg_change": 0, "gainers": 0, "losers": 0},
            "top_gainers": [],
            "top_losers": [],
            "oversold": [],
            "overbought": [],
            "volume_breakouts": [],
            "macd_bullish": [],
            "watchlist_data": {},
        })
        self.assertIn("Pre-Market Briefing", output)


if __name__ == "__main__":
    unittest.main()
