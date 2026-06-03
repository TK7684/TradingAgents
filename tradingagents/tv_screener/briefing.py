#!/usr/bin/env python3
"""Pre-market briefing from TradingView Premium screener.

Generates a comprehensive market briefing before the daily TradingAgents
analysis runs. Enriches the watchlist with real-time TradingView data.

Usage:
    python -m tradingagents.tv_screener.briefing          # Full briefing
    python -m tradingagents.tv_screener.briefing --format json  # JSON output
    python -m tradingagents.tv_screener.briefing --discord    # Format for Discord
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from tradingagents.tv_screener.client import TVScreener

logger = logging.getLogger(__name__)


def format_discord_briefing(briefing: dict) -> str:
    """Format briefing as Discord-compatible markdown."""
    lines = []

    # Header
    now = datetime.now(timezone.utc)
    lines.append(f"## 📊 Pre-Market Briefing — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # Market breadth
    b = briefing["market_breadth"]
    sentiment = "🟢 BULLISH" if b["bull_bear_ratio"] > 1.5 else ("🔴 BEARISH" if b["bull_bear_ratio"] < 0.8 else "🟡 NEUTRAL")
    lines.append(f"**Market Sentiment:** {sentiment} ({b['bull_bear_ratio']}x Bull/Bear)")
    lines.append(f"Avg RSI: {b['avg_rsi']} | Scanned: {b['total_scanned']} stocks")
    lines.append(f"Analysts Bullish: {b['bullish']} | Bearish: {b['bearish']} | Neutral: {b['neutral']}")
    lines.append("")

    # Top gainers
    if briefing["top_gainers"]:
        lines.append("### 🚀 Top Gainers (>$10B cap)")
        lines.append("```")
        lines.append(f"{'Ticker':<7} {'Price':>10} {'Chg%':>7} {'RSI':>6} {'Signal':>12}")
        lines.append("-" * 45)
        for m in briefing["top_gainers"][:10]:
            lines.append(f"{m['symbol']:<7} ${m['close']:>9,.2f} {m['change_pct']:>+6.2f}% {m['rsi']:>5.1f} {m['recommendation_label']:>12}")
        lines.append("```")
        lines.append("")

    # Top losers
    if briefing["top_losers"]:
        lines.append("### 📉 Top Losers (>$10B cap)")
        lines.append("```")
        lines.append(f"{'Ticker':<7} {'Price':>10} {'Chg%':>7} {'RSI':>6} {'Signal':>12}")
        lines.append("-" * 45)
        for m in briefing["top_losers"][:10]:
            lines.append(f"{m['symbol']:<7} ${m['close']:>9,.2f} {m['change_pct']:>+6.2f}% {m['rsi']:>5.1f} {m['recommendation_label']:>12}")
        lines.append("```")
        lines.append("")

    # Oversold
    if briefing["oversold"]:
        lines.append("### 🩸 Oversold (RSI < 30)")
        for m in briefing["oversold"][:5]:
            lines.append(f"  • `{m['symbol']}` ${m['close']:,.2f} RSI:{m['rsi']:.1f} — {m['recommendation_label']}")
        lines.append("")

    # Overbought
    if briefing["overbought"]:
        lines.append("### 🌕 Overbought (RSI > 70)")
        for m in briefing["overbought"][:5]:
            lines.append(f"  • `{m['symbol']}` ${m['close']:,.2f} RSI:{m['rsi']:.1f} — {m['recommendation_label']}")
        lines.append("")

    # Volume breakouts
    if briefing["volume_breakouts"]:
        lines.append("### 📢 Volume Breakouts (2x normal)")
        for m in briefing["volume_breakouts"][:5]:
            rv = m.get("relative_volume_10d_calc", 0)
            if rv:
                lines.append(f"  • `{m['symbol']}` {m['change_pct']:+.2f}% RV:{rv:.1f}x — {m.get('name', '')}")
        lines.append("")

    # Watchlist enrichment
    if briefing["watchlist_data"]:
        lines.append("### 📋 Watchlist Status")
        lines.append("```")
        lines.append(f"{'Ticker':<7} {'Price':>10} {'Chg%':>7} {'RSI':>6} {'Rec':>10} {'Vol':>12}")
        lines.append("-" * 55)
        for sym, d in sorted(briefing["watchlist_data"].items()):
            close = d.get("close", 0) or 0
            change = d.get("change", 0) or 0
            rsi = d.get("RSI", 0)
            rec = d.get("Recommend.All", 0)
            vol = d.get("volume", 0)
            rec_label = "STRONG BUY" if rec > 0.8 else ("BUY" if rec > 0.4 else ("HOLD" if rec > -0.2 else ("SELL" if rec > -0.6 else "STRONG SELL")))
            rsi_str = f"{rsi:.1f}" if rsi else "N/A"
            lines.append(f"{sym:<7} ${close:>9,.2f} {change:>+6.2f}% {rsi_str:>6} {rec_label:>10} {vol/1e6:>10.1f}M")
        lines.append("```")

    return "\n".join(lines)


def format_market_context_for_llm(briefing: dict, ticker_symbol: str = None) -> str:
    """Format TradingView briefing data into a compact text section for LLM prompt injection.

    Produces a concise market context block that gives the LLM real-time awareness
    of market conditions, top movers, and per-ticker TradingView data.

    Args:
        briefing: Dict from TVScreener.get_pre_market_watchlist()
        ticker_symbol: Optional ticker to highlight in context (e.g. "NVDA")

    Returns:
        Formatted string suitable for injection into LLM prompts.
        Returns empty string if briefing is empty or malformed.
    """
    if not briefing:
        return ""

    lines = []
    lines.append("=" * 60)
    lines.append("REAL-TIME MARKET CONTEXT (TradingView Screener)")
    lines.append("=" * 60)

    # Market breadth
    b = briefing.get("market_breadth", {})
    if b and "error" not in b:
        ratio = b.get("bull_bear_ratio", 0)
        sentiment = "BULLISH" if ratio > 1.5 else ("BEARISH" if ratio < 0.8 else "NEUTRAL")
        lines.append(f"\n📊 MARKET BREADTH: {sentiment} (Bull/Bear ratio: {ratio}x)")
        lines.append(f"   Stocks scanned: {b.get('total_scanned', 'N/A')}")
        lines.append(f"   Analysts — Bullish: {b.get('bullish', 'N/A')}, "
                      f"Bearish: {b.get('bearish', 'N/A')}, Neutral: {b.get('neutral', 'N/A')}")
        if b.get("avg_rsi") is not None:
            lines.append(f"   Average RSI: {b['avg_rsi']}")
        if b.get("avg_change") is not None:
            lines.append(f"   Average change: {b['avg_change']:+.2f}%")
        if b.get("gainers") is not None:
            lines.append(f"   Gainers: {b['gainers']}, Losers: {b['losers']}")

    # Top gainers
    gainers = briefing.get("top_gainers", [])
    if gainers:
        lines.append("\n🚀 TOP GAINERS (>$10B cap, >+2%):")
        for m in gainers[:8]:
            lines.append(f"   {m['symbol']:<6} ${m['close']:>9,.2f}  {m['change_pct']:+.2f}%  "
                         f"RSI:{m.get('rsi', 0):.1f}  {m['recommendation_label']}")

    # Top losers
    losers = briefing.get("top_losers", [])
    if losers:
        lines.append("\n📉 TOP LOSERS (>$10B cap, <-2%):")
        for m in losers[:8]:
            lines.append(f"   {m['symbol']:<6} ${m['close']:>9,.2f}  {m['change_pct']:+.2f}%  "
                         f"RSI:{m.get('rsi', 0):.1f}  {m['recommendation_label']}")

    # RSI extremes
    oversold = briefing.get("oversold", [])
    if oversold:
        lines.append("\n🩸 OVERSOLD (RSI < 30):")
        for m in oversold[:5]:
            lines.append(f"   {m['symbol']:<6} RSI:{m.get('rsi', 0):.1f}  "
                         f"${m['close']:,.2f}  {m['recommendation_label']}")

    overbought = briefing.get("overbought", [])
    if overbought:
        lines.append("\n🌕 OVERBOUGHT (RSI > 70):")
        for m in overbought[:5]:
            lines.append(f"   {m['symbol']:<6} RSI:{m.get('rsi', 0):.1f}  "
                         f"${m['close']:,.2f}  {m['recommendation_label']}")

    # Volume breakouts
    vol_breakouts = briefing.get("volume_breakouts", [])
    if vol_breakouts:
        lines.append("\n📢 VOLUME BREAKOUTS (>2x normal volume):")
        for m in vol_breakouts[:5]:
            rv = m.get("relative_volume_10d_calc", 0)
            rv_str = f"{rv:.1f}x" if rv else "N/A"
            lines.append(f"   {m['symbol']:<6} {m['change_pct']:+.2f}%  RelVol:{rv_str}  "
                         f"{m.get('name', '')}")

    # MACD crossovers
    macd_bullish = briefing.get("macd_bullish", [])
    if macd_bullish:
        lines.append("\n✨ MACD BULLISH CROSSOVERS:")
        for m in macd_bullish[:5]:
            hist = m.get("MACD.histogram", 0)
            lines.append(f"   {m['symbol']:<6} ${m['close']:,.2f}  "
                         f"Hist:{hist:.4f}  {m['recommendation_label']}")

    # Per-ticker enrichment data (highlight the analyzed ticker)
    watchlist = briefing.get("watchlist_data", {})
    if watchlist:
        lines.append("\n📋 WATCHLIST TRADINGVIEW DATA:")
        for sym, d in sorted(watchlist.items()):
            close = d.get("close", 0) or 0
            change = d.get("change", 0) or 0
            rsi = d.get("RSI")
            rec = d.get("Recommend.All", 0)
            vol = d.get("volume", 0)
            macd_hist = d.get("MACD.histogram")
            rec_label = ("STRONG BUY" if rec > 0.8 else
                         "BUY" if rec > 0.4 else
                         "HOLD" if rec > -0.2 else
                         "SELL" if rec > -0.6 else "STRONG SELL")
            rsi_str = f"{rsi:.1f}" if rsi else "N/A"
            macd_str = f"MACD_H:{macd_hist:.4f}" if macd_hist else ""
            marker = " ◀ ANALYZING" if ticker_symbol and sym == ticker_symbol else ""
            lines.append(f"   {sym:<7} ${close:>9,.2f} {change:>+6.2f}%  "
                         f"RSI:{rsi_str:>6}  {rec_label:<11}  "
                         f"Vol:{vol/1e6:.1f}M  {macd_str}{marker}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TradingView Pre-Market Briefing")
    parser.add_argument("--format", choices=["text", "json", "discord"], default="text")
    parser.add_argument("--output", help="Write output to file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Generating TradingView pre-market briefing...")
    tv = TVScreener()

    briefing = tv.get_pre_market_watchlist()

    if args.format == "json":
        output = json.dumps(briefing, indent=2, default=str)
    elif args.format == "discord":
        output = format_discord_briefing(briefing)
    else:
        output = format_discord_briefing(briefing)

    print(output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        logger.info("Written to %s", args.output)


if __name__ == "__main__":
    main()
