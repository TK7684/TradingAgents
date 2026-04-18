#!/usr/bin/env python3
"""Minimal Discord webhook publisher for TradingAgents status reports."""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path

WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

if not WEBHOOK_URL:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DISCORD_WEBHOOK_URL="):
                WEBHOOK_URL = line.split("=", 1)[1].strip().strip('"').strip("'")
                break


def post_status(title, message, color=0x00ff00):
    """Post status embed to Discord webhook."""
    if not WEBHOOK_URL:
        print("ERROR: DISCORD_WEBHOOK_URL not set")
        return False

    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
        }]
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Luna-Trader/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 204
    except urllib.error.HTTPError as e:
        print(f"ERROR: Discord webhook failed ({e.code}): {e.reason}")
        return False
    except Exception as e:
        print(f"ERROR: Discord webhook error: {e}")
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        post_status("Test", "TradingAgents webhook is working", color=0x0099ff)
