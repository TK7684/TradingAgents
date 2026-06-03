"""TradingView Premium Screener Integration.

Uses tradingview-screener Python package (official TradingView API) with
authenticated cookies from Chrome for real-time data. Feeds into:
- TradingAgents daily analysis (pre-market screener → enriched watchlist)
- TradingView webhook alerts → MT5 bridge (existing pipeline)
- Discord signal notifications
- Hyperliquid DEX crypto signals (future)

Key premium features leveraged:
- 400 price alerts + 400 technical alerts (vs 3/0 on free)
- 20K historical bars (vs 5K on free)
- Volume Profile, TPO, Footprint charts
- Pine Script backtesting with export
- Webhook alerts with custom JSON
- Multi-condition alerts
- Screener with 3000+ data fields
"""

__version__ = "1.0.0"
