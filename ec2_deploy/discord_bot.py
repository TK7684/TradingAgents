#!/usr/bin/env python3
"""Discord command handler via webhook interactions.

Since we use webhooks (not a full bot), this provides command-like
functionality by polling a simple command file that can be triggered
from Discord or any HTTP client.

Usage:
    # Run as a service
    python3.12 discord_bot.py

    # Or trigger specific commands
    python3.12 discord_bot.py analyze AAPL
    python3.12 discord_bot.py status
    python3.12 discord_bot.py backtest
    python3.12 discord_bot.py portfolio
    python3.12 discord_bot.py watchlist
"""
import sys
import os
import subprocess
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import requests

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
BASE_DIR = os.path.expanduser("~/TradingAgents")


def send_discord(title, description, color=0x5865F2):
    if not WEBHOOK_URL:
        return
    payload = {"embeds": [{
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }]}
    requests.post(WEBHOOK_URL, json=payload)


def cmd_analyze(args):
    """Run analysis on a ticker."""
    ticker = args[0] if args else "NVDA"
    profile = "turbo"
    for i, a in enumerate(args):
        if a == "--profile" and i + 1 < len(args):
            profile = args[i + 1]

    send_discord("\U0001f50d Analyzing {}...".format(ticker),
                 "Profile: {} | Started: {}".format(profile, datetime.now().strftime("%H:%M UTC")),
                 0x5865F2)

    result = subprocess.run(
        ["python3.12", os.path.join(BASE_DIR, "analyze.py"), ticker, "--profile", profile],
        capture_output=True, text=True, cwd=BASE_DIR, timeout=600
    )
    if result.returncode != 0:
        send_discord("\u274c Analysis Failed: {}".format(ticker), result.stderr[:500], 0xED4245)
    # Note: analyze.py already sends Discord notification


def cmd_status():
    """Show system status."""
    import shutil
    disk = shutil.disk_usage("/")
    disk_pct = round(disk.used / disk.total * 100, 1)
    disk_free = round(disk.free / (1024**3), 1)

    import glob
    results = sorted(glob.glob(os.path.join(BASE_DIR, "results/daily/*.json")))
    last_run = "never"
    if results:
        with open(results[-1]) as f:
            data = json.load(f)
            last_run = data["date"]

    # Load decision history
    history_file = os.path.join(BASE_DIR, "results/decision_history.json")
    positions = 0
    if os.path.exists(os.path.join(BASE_DIR, "results/portfolio.json")):
        with open(os.path.join(BASE_DIR, "results/portfolio.json")) as f:
            portfolio = json.load(f)
            positions = len(portfolio.get("positions", {}))

    desc = (
        "**Last Analysis:** {}\n"
        "**Disk:** {}% used ({:.1f}G free)\n"
        "**Results Files:** {}\n"
        "**Open Positions:** {}\n"
    ).format(last_run, disk_pct, disk_free, len(results), positions)

    send_discord("\U0001f4ca System Status", desc, 0x5865F2)
    print(desc)


def cmd_portfolio():
    """Show paper portfolio."""
    subprocess.run(["python3.12", os.path.join(BASE_DIR, "paper_trader.py"), "--status"],
                   cwd=BASE_DIR)


def cmd_backtest():
    """Run backtest."""
    subprocess.run(["python3.12", os.path.join(BASE_DIR, "backtest.py")],
                   cwd=BASE_DIR, timeout=300)


def cmd_watchlist():
    """Show current watchlist."""
    sys.path.insert(0, BASE_DIR)
    from config import WATCHLIST
    # Load current signals
    history_file = os.path.join(BASE_DIR, "results/decision_history.json")
    signals = {}
    if os.path.exists(history_file):
        with open(history_file) as f:
            signals = json.load(f)

    lines = []
    emoji_map = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "HOLD": "\U0001f7e1"}
    for t in WATCHLIST:
        sig = signals.get(t, {})
        dec = sig.get("decision", "---")
        date = sig.get("date", "")
        emoji = emoji_map.get(dec, "\u26aa")
        lines.append("{} **{}** {} ({})".format(emoji, t, dec, date))

    send_discord("\U0001f4cb Watchlist", "\n".join(lines), 0x5865F2)
    print("\n".join(lines))


def cmd_help():
    """Show available commands."""
    desc = (
        "**Commands:**\n"
        "`analyze <TICKER> [--profile turbo|default|deep]`\n"
        "`status` — System health\n"
        "`portfolio` — Paper trading P&L\n"
        "`backtest` — Decision accuracy\n"
        "`watchlist` — Current signals\n"
        "`help` — This message\n"
    )
    send_discord("\U0001f916 TradingAgents Commands", desc, 0x5865F2)
    print(desc)


COMMANDS = {
    "analyze": cmd_analyze,
    "status": cmd_status,
    "portfolio": cmd_portfolio,
    "backtest": cmd_backtest,
    "watchlist": cmd_watchlist,
    "help": cmd_help,
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        cmd_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd in COMMANDS:
        if cmd == "analyze":
            COMMANDS[cmd](args)
        else:
            COMMANDS[cmd]()
    else:
        print("Unknown command: {}. Use 'help' for available commands.".format(cmd))
