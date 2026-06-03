"""TradingView Screener client — real-time market data from TradingView Premium.

Authenticates via Chrome cookies (rookiepy) to get streaming/real-time data.
Falls back to delayed data (15-min) without cookies — still useful for screeners.

Usage:
    from tradingagents.tv_screener.client import TVScreener

    tv = TVScreener()
    movers = tv.get_top_movers(direction="up", min_cap="10B", limit=15)
    oversold = tv.get_rsi_screener(rsi_max=30, min_cap="5B")
    watchlist_data = tv.get_watchlist_data(["NVDA", "AAPL", "TSLA"])
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import rookiepy
from tradingview_screener import Query, col, stocks, crypto, crypto_dex

logger = logging.getLogger(__name__)

# Default fields for screener queries
DEFAULT_FIELDS = [
    "name", "close", "change", "change_abs", "volume",
    "market_cap_basic", "Recommend.All", "RSI",
    "MACD.macd", "MACD.signal", "EMA20", "SMA50", "SMA200",
    "Perf.1M", "Perf.W", "Volatility.D",
]

# Fields for watchlist enrichment (adds to daily analysis)
ENRICHMENT_FIELDS = [
    "name", "close", "change", "change_abs", "volume",
    "market_cap_basic", "Recommend.All", "RSI", "RSI|1",
    "MACD.macd", "MACD.signal", "MACD.histogram",
    "EMA20", "EMA50", "SMA50", "SMA200",
    "BB.upper", "BB.lower", "ATR",
    "Perf.1W", "Perf.1M", "Perf.3M", "Perf.1Y",
    "Volatility.D", "Volatility.W", "Volatility.M",
    "ADX", "Stoch.K", "Stoch.D",
    "relative_volume_10d_calc",
    "High.All", "Low.All",
    "Pivot.M.High", "Pivot.M.Low",
]


@dataclass
class MoverResult:
    """A stock/crypto mover from the screener."""
    ticker: str
    name: str
    close: float
    change_pct: float
    volume: float
    market_cap: float
    rsi: float
    recommendation: float  # -1.0 (strong sell) to +1.0 (strong buy)
    extra: dict = field(default_factory=dict)

    @property
    def exchange(self) -> str:
        return self.ticker.split(":")[0] if ":" in self.ticker else "UNKNOWN"

    @property
    def symbol(self) -> str:
        return self.ticker.split(":")[-1] if ":" in self.ticker else self.ticker

    @property
    def recommendation_label(self) -> str:
        r = self.recommendation
        if r >= 0.8: return "STRONG_BUY"
        if r >= 0.4: return "BUY"
        if r >= -0.2: return "HOLD"
        if r >= -0.6: return "SELL"
        return "STRONG_SELL"

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "name": self.name,
            "close": self.close,
            "change_pct": round(self.change_pct, 2),
            "volume": self.volume,
            "market_cap": self.market_cap,
            "rsi": round(self.rsi, 2) if pd.notna(self.rsi) else None,
            "recommendation": round(self.recommendation, 4) if pd.notna(self.recommendation) else None,
            "recommendation_label": self.recommendation_label,
            **self.extra,
        }


class TVScreener:
    """TradingView Premium Screener client.

    Authenticates via Chrome cookies for real-time data.
    All methods return lists of MoverResult or dicts.
    """

    def __init__(self, cookies=None):
        """Initialize screener.

        Args:
            cookies: CookieJar for real-time data. If None, attempts Chrome auto-detect.
        """
        if cookies is None:
            try:
                self._cookies = rookiepy.to_cookiejar(
                    rookiepy.chrome([".tradingview.com"])
                )
                logger.info("TradingView cookies loaded from Chrome")
            except Exception as e:
                logger.warning("Could not load Chrome cookies, using delayed data: %s", e)
                self._cookies = None
        else:
            self._cookies = cookies

    @property
    def has_realtime(self) -> bool:
        return self._cookies is not None

    def _run_query(self, query: Query) -> tuple:
        """Execute a screener query."""
        kwargs = {}
        if self._cookies:
            kwargs["cookies"] = self._cookies
        return query.get_scanner_data(**kwargs)

    def _df_to_movers(self, df: pd.DataFrame) -> list[MoverResult]:
        """Convert DataFrame to list of MoverResult."""
        results = []
        for _, row in df.iterrows():
            results.append(MoverResult(
                ticker=str(row.get("ticker", "")),
                name=str(row.get("name", "")),
                close=float(row.get("close", 0) or 0),
                change_pct=float(row.get("change", 0) or 0),
                volume=float(row.get("volume", 0) or 0),
                market_cap=float(row.get("market_cap_basic", 0) or 0),
                rsi=float(row.get("RSI", 0) or 0) if pd.notna(row.get("RSI")) else 0,
                recommendation=float(row.get("Recommend.All", 0) or 0) if pd.notna(row.get("Recommend.All")) else 0,
            extra={
                k: float(v) if pd.notna(v) and isinstance(v, (int, float)) else v
                for k, v in row.items()
                if k not in ("ticker", "name", "close", "change", "volume",
                             "market_cap_basic", "RSI", "Recommend.All")
            },
            ))
        return results

    def get_top_movers(
        self,
        direction: str = "up",
        min_cap: str = "10B",
        min_volume: int = 1_000_000,
        limit: int = 15,
        exchange: str = "america",
        major_exchanges_only: bool = True,
    ) -> list[MoverResult]:
        """Get top gainers or losers by % change today.

        Args:
            direction: "up" for gainers, "down" for losers
            min_cap: Minimum market cap string (e.g., "10B", "1B", "100M")
            min_volume: Minimum daily volume
            limit: Max results
            exchange: Market to scan ("america", etc.)
            major_exchanges_only: Filter to NASDAQ/NYSE only (skip OTC/penny stocks)
        """
        cap_num = self._parse_cap(min_cap)
        ascending = direction == "down"
        api_limit = min(limit * 5, 500) if major_exchanges_only else limit

        q = (
            stocks(exchange)
            .select(*DEFAULT_FIELDS, "exchange", "submarket")
            .where(
                col("market_cap_basic") > cap_num,
                col("volume") > min_volume,
            )
            .order_by("change", ascending=ascending)
            .limit(api_limit)
        )

        if direction == "down":
            q = q.where(col("change") < -2)
        else:
            q = q.where(col("change") > 2)

        _, df = self._run_query(q)

        # Filter out OTC/penny stocks
        if major_exchanges_only and "exchange" in df.columns:
            df = df[df["exchange"].isin(["NASDAQ", "NYSE", "AMEX"])]
            df = df.head(limit)

        return self._df_to_movers(df)

    def get_rsi_screener(
        self,
        rsi_min: float = 0,
        rsi_max: float = 30,
        min_cap: str = "5B",
        limit: int = 20,
        exchange: str = "america",
        major_exchanges_only: bool = True,
    ) -> list[MoverResult]:
        cap_num = self._parse_cap(min_cap)
        api_limit = min(limit * 5, 500) if major_exchanges_only else limit

        q = (
            stocks(exchange)
            .select(*DEFAULT_FIELDS, "exchange", "submarket")
            .where(
                col("market_cap_basic") > cap_num,
                col("RSI") >= rsi_min,
                col("RSI") <= rsi_max,
            )
            .order_by("RSI", ascending=True)
            .limit(api_limit)
        )

        _, df = self._run_query(q)

        if major_exchanges_only and "exchange" in df.columns:
            df = df[df["exchange"].isin(["NASDAQ", "NYSE", "AMEX"])]
            df = df.head(limit)

        return self._df_to_movers(df)

    def get_volume_breakout(
        self,
        rv_mult: float = 2.0,
        min_cap: str = "5B",
        limit: int = 15,
        exchange: str = "america",
        major_exchanges_only: bool = True,
    ) -> list[MoverResult]:
        cap_num = self._parse_cap(min_cap)
        api_limit = min(limit * 5, 500) if major_exchanges_only else limit

        q = (
            stocks(exchange)
            .select(*DEFAULT_FIELDS, "relative_volume_10d_calc", "exchange", "submarket")
            .where(
                col("market_cap_basic") > cap_num,
                col("relative_volume_10d_calc") > rv_mult,
            )
            .order_by("relative_volume_10d_calc", ascending=False)
            .limit(api_limit)
        )

        _, df = self._run_query(q)

        if major_exchanges_only and "exchange" in df.columns:
            df = df[df["exchange"].isin(["NASDAQ", "NYSE", "AMEX"])]
            df = df.head(limit)

        return self._df_to_movers(df)

    def get_watchlist_data(
        self,
        tickers: list[str],
        fields: list[str] | None = None,
    ) -> dict[str, dict]:
        """Get real-time TradingView data for a list of tickers.

        Returns a dict mapping ticker symbol → data dict.

        Args:
            tickers: List of ticker symbols (e.g., ["NVDA", "AAPL", "TSLA"])
            fields: Fields to select (default: ENRICHMENT_FIELDS)
        """
        fields = fields or ENRICHMENT_FIELDS

        # TradingView screener uses exchange:SYMBOL format
        # We need to find each ticker. Simple approach: scan america market
        results = {}

        for batch_start in range(0, len(tickers), 50):
            batch = tickers[batch_start:batch_start + 50]

            q = (
                stocks("america")
                .select(*fields)
                .order_by("market_cap_basic", ascending=False)
                .limit(5000)
            )

            _, df = self._run_query(q)

            for _, row in df.iterrows():
                tv_ticker = str(row.get("ticker", ""))
                symbol = tv_ticker.split(":")[-1] if ":" in tv_ticker else tv_ticker
                if symbol in batch:
                    results[symbol] = {
                        k: float(v) if pd.notna(v) and isinstance(v, (int, float)) else v
                        for k, v in row.items()
                    }
                    results[symbol]["tv_ticker"] = tv_ticker

        return results

    def get_macd_crossover(
        self,
        min_cap: str = "5B",
        limit: int = 15,
        exchange: str = "america",
        major_exchanges_only: bool = True,
    ) -> list[MoverResult]:
        cap_num = self._parse_cap(min_cap)

        q = (
            stocks(exchange)
            .select(*DEFAULT_FIELDS, "MACD.histogram", "exchange", "submarket")
            .where(
                col("market_cap_basic") > cap_num,
                col("MACD.macd") > col("MACD.signal"),
                col("MACD.histogram") > 0,
            )
            .order_by("MACD.histogram", ascending=False)
            .limit(limit)
        )

        _, df = self._run_query(q)

        if major_exchanges_only and "exchange" in df.columns:
            df = df[df["exchange"].isin(["NASDAQ", "NYSE", "AMEX"])]

        return self._df_to_movers(df)

    def get_unusual_options_activity(
        self,
        min_volume: int = 10_000,
        limit: int = 10,
    ) -> list[MoverResult]:
        """Get options with unusually high volume."""
        from tradingview_screener import options

        q = (
            Query()
            .select("name", "close", "volume", "open_interest", "implied_volatility")
            .where(col("volume") > min_volume)
            .order_by("volume", ascending=False)
            .limit(limit)
        )

        _, df = self._run_query(q)
        return self._df_to_movers(df)

    def get_market_breadth(self) -> dict:
        """Get overall market breadth — bullish vs bearish ratio.

        Returns dict with:
        - total_scanned: number of stocks
        - bullish: count with positive recommendation
        - bearish: count with negative recommendation
        - neutral: count in between
        - avg_rsi: average RSI across market
        - new_highs: stocks near 52-week high
        - new_lows: stocks near 52-week low
        """
        q = (
            stocks("america")
            .select("close", "RSI", "Recommend.All", "High.All", "Low.All")
            .where(col("market_cap_basic") > 1_000_000_000)
            .order_by("volume", ascending=False)
            .limit(1000)
        )

        _, df = self._run_query(q)

        if df.empty:
            return {"error": "No data returned"}

        rec_col = df["Recommend.All"].dropna()
        bullish = int((rec_col > 0.2).sum())
        bearish = int((rec_col < -0.2).sum())
        neutral = len(rec_col) - bullish - bearish

        return {
            "total_scanned": len(df),
            "bullish": int(bullish),
            "bearish": int(bearish),
            "neutral": int(neutral),
            "bull_bear_ratio": round(bullish / max(bearish, 1), 2),
            "avg_rsi": round(float(df["RSI"].mean()), 2) if df["RSI"].notna().any() else None,
            "avg_change": round(float(df["change"].mean()), 2) if "change" in df.columns else None,
            "gainers": int((df.get("change", pd.Series(dtype=float)).fillna(0) > 0).sum()),
            "losers": int((df.get("change", pd.Series(dtype=float)).fillna(0) < 0).sum()),
        }

    def get_pre_market_watchlist(
        self,
        sectors: dict[str, list[str]] | None = None,
        top_movers: int = 10,
    ) -> dict:
        """Generate pre-market briefing for the daily pipeline.

        Combines:
        1. Top movers (gainers + losers)
        2. RSI extremes (oversold + overbought)
        3. Volume breakouts
        4. MACD crossovers
        5. Market breadth summary

        Returns a briefing dict ready for Discord or LLM context injection.
        """
        sectors = sectors or {
            "tech": ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "PLTR"],
            "etf": ["SPY", "QQQ", "IWM", "DIA", "ARKK"],
        }

        briefing = {
            "market_breadth": self.get_market_breadth(),
            "top_gainers": [m.to_dict() for m in self.get_top_movers("up", limit=top_movers)],
            "top_losers": [m.to_dict() for m in self.get_top_movers("down", limit=top_movers)],
            "oversold": [m.to_dict() for m in self.get_rsi_screener(rsi_max=30)],
            "overbought": [m.to_dict() for m in self.get_rsi_screener(rsi_min=70, rsi_max=100)],
            "volume_breakouts": [m.to_dict() for m in self.get_volume_breakout()],
            "macd_bullish": [m.to_dict() for m in self.get_macd_crossover()],
            "watchlist_data": self.get_watchlist_data(
                list({t for sector in sectors.values() for t in sector})
            ),
        }

        return briefing

    @staticmethod
    def _parse_cap(cap_str: str) -> float:
        """Parse market cap string to number.

        Examples: "10B" → 10_000_000_000, "500M" → 500_000_000
        """
        cap_str = cap_str.upper().strip()
        multipliers = {
            "T": 1_000_000_000_000,
            "B": 1_000_000_000,
            "M": 1_000_000,
            "K": 1_000,
        }
        for suffix, mult in multipliers.items():
            if cap_str.endswith(suffix):
                try:
                    return float(cap_str[:-1]) * mult
                except ValueError:
                    return 1_000_000_000  # default 1B
        try:
            return float(cap_str)
        except ValueError:
            return 1_000_000_000
