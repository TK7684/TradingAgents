import unittest

from cli.utils import normalize_ticker_symbol
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.dataflows.ticker_utils import normalize_yf_ticker


class TickerSymbolHandlingTests(unittest.TestCase):
    def test_normalize_ticker_symbol_preserves_exchange_suffix(self):
        self.assertEqual(normalize_ticker_symbol(" cnc.to "), "CNC.TO")

    def test_build_instrument_context_mentions_exact_symbol(self):
        context = build_instrument_context("7203.T")
        self.assertIn("7203.T", context)
        self.assertIn("exchange suffix", context)

    def test_build_instrument_context_mentions_crypto(self):
        context = build_instrument_context("BTC-USD")
        self.assertIn("BTC-USD", context)
        self.assertIn("crypto", context)

    def test_normalize_bare_crypto_appends_usd_pair(self):
        self.assertEqual(normalize_yf_ticker("btc"), "BTC-USD")
        self.assertEqual(normalize_yf_ticker(" ETH "), "ETH-USD")
        self.assertEqual(normalize_ticker_symbol("sol"), "SOL-USD")

    def test_normalize_preserves_existing_crypto_pair(self):
        self.assertEqual(normalize_yf_ticker("BTC-USD"), "BTC-USD")
        self.assertEqual(normalize_yf_ticker("eth-eur"), "ETH-EUR")

    def test_normalize_leaves_equities_untouched(self):
        self.assertEqual(normalize_yf_ticker("spy"), "SPY")
        self.assertEqual(normalize_yf_ticker("AAPL"), "AAPL")

    def test_normalize_preserves_exchange_suffix(self):
        self.assertEqual(normalize_yf_ticker("cnc.to"), "CNC.TO")
        self.assertEqual(normalize_yf_ticker("7203.t"), "7203.T")

    def test_normalize_handles_empty(self):
        self.assertEqual(normalize_yf_ticker(""), "")
        self.assertEqual(normalize_yf_ticker("   "), "")


if __name__ == "__main__":
    unittest.main()
