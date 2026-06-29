#!/usr/bin/env python3
"""Send trading signals to Discord via bot token (no webhook needed)."""
import os
import sys
import json
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BOT_TOKEN = os.getenv("DISCORD_TOKEN", "")
CHANNEL_ID = os.getenv("DISCORD_SIGNAL_THREAD", "") or os.getenv("DISCORD_TRADE_CHANNEL", "")


def send_signal(embed_title, embed_description, color=0x5865F2):
    """Send an embed message to the trading-signals channel."""
    if not BOT_TOKEN or not CHANNEL_ID:
        print("WARN: DISCORD_TOKEN or DISCORD_TRADE_CHANNEL not set, skipping Discord")
        return False

    url = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
    payload = {
        "embeds": [{
            "title": embed_title,
            "description": embed_description[:4096],  # Discord limit
            "color": color,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }]
    }
    headers = {
        "Authorization": f"Bot {BOT_TOKEN}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code == 200:
        print(f"Discord signal sent: {embed_title}")
        return True
    else:
        print(f"Discord error {resp.status_code}: {resp.text}")
        return False


def send_trade_decision(ticker, decision, summary=""):
    """Send a trade decision to Discord."""
    color_map = {"BUY": 0x57F287, "SELL": 0xED4245, "HOLD": 0xFEE75C}
    color = color_map.get(decision, 0x5865F2)
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    emoji = emoji_map.get(decision, "⚪")

    title = f"{emoji} {decision}: {ticker}"
    desc = summary or f"TradingAgents analysis for {ticker}"
    return send_signal(title, desc, color)


def send_portfolio_update(portfolio_data):
    """Send portfolio status update."""
    total = portfolio_data.get("total_value", 0)
    cash = portfolio_data.get("cash", 0)
    positions = portfolio_data.get("positions", {})
    pnl = total - 100000
    pnl_pct = round(pnl / 100000 * 100, 2) if 100000 != 0 else 0

    emoji = "💰" if pnl >= 0 else "📉"
    sign = "+" if pnl >= 0 else ""

    pos_lines = []
    for ticker, pos in positions.items():
        pos_lines.append(f"• {ticker}: {pos['shares']} shares @ ${pos['avg_price']:.2f}")

    desc = (
        f"**Portfolio Value:** ${total:,.0f} ({sign}{pnl_pct}%)\n"
        f"**Cash:** ${cash:,.0f}\n"
        f"**Open Positions:** {len(positions)}\n"
    )
    if pos_lines:
        desc += "\n" + "\n".join(pos_lines)

    color = 0x57F287 if pnl >= 0 else 0xED4245
    return send_signal(f"{emoji} Portfolio Update", desc, color)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        send_trade_decision(sys.argv[1], sys.argv[2], " ".join(sys.argv[3:]))
    else:
        print("Usage: discord_signal.py <TICKER> <BUY|SELL|HOLD> [summary]")
