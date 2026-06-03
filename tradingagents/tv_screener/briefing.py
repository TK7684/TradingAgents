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
