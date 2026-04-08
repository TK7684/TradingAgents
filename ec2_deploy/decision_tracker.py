#!/usr/bin/env python3
"""Track decision changes and alert when a ticker flips signal.

Compares today's decisions with the most recent previous decisions.
Alerts on Discord when BUY->SELL, SELL->BUY, etc.

Runs automatically after daily batch analysis.
"""
import json
import glob
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import requests

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
RESULTS_DIR = os.path.expanduser("~/TradingAgents/results/daily")
TRACKER_FILE = os.path.expanduser("~/TradingAgents/results/decision_history.json")

FLIP_EMOJI = {
    ("SELL", "BUY"): "\U0001f7e2\u2b06\ufe0f",   # green up
    ("BUY", "SELL"): "\U0001f534\u2b07\ufe0f",    # red down
    ("HOLD", "BUY"): "\U0001f7e2\u2197\ufe0f",    # green diagonal
    ("HOLD", "SELL"): "\U0001f534\u2198\ufe0f",    # red diagonal
    ("BUY", "HOLD"): "\U0001f7e1\u27a1\ufe0f",    # yellow right
    ("SELL", "HOLD"): "\U0001f7e1\u27a1\ufe0f",   # yellow right
}


def load_history():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE) as f:
            return json.load(f)
    return {}


def save_history(history):
    with open(TRACKER_FILE, "w") as f:
        json.dump(history, f, indent=2)


def track_decisions(json_path=None):
    """Compare latest results with history, detect flips."""
    history = load_history()

    # Find latest result file
    if json_path:
        files = [json_path]
    else:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
        if not files:
            print("No results to track")
            return []
        files = [files[-1]]

    flips = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        date = data["date"]
        for r in data["results"]:
            ticker = r["ticker"]
            decision = r.get("decision")
            if not decision or r.get("status") != "ok":
                continue

            prev = history.get(ticker, {})
            prev_decision = prev.get("decision")
            prev_date = prev.get("date")

            # Update history
            history[ticker] = {"decision": decision, "date": date}

            # Detect flip
            if prev_decision and prev_decision != decision:
                flip = {
                    "ticker": ticker,
                    "from": prev_decision,
                    "to": decision,
                    "prev_date": prev_date,
                    "date": date,
                }
                flips.append(flip)
                emoji = FLIP_EMOJI.get((prev_decision, decision), "\U0001f504")
                print("{} {} flipped: {} -> {} (was {} on {})".format(
                    emoji, ticker, prev_decision, decision, prev_decision, prev_date))

    save_history(history)

    # Send Discord alert for flips
    if flips and WEBHOOK_URL:
        lines = []
        for flip in flips:
            emoji = FLIP_EMOJI.get((flip["from"], flip["to"]), "\U0001f504")
            lines.append("{} **{}**: {} -> **{}**".format(
                emoji, flip["ticker"], flip["from"], flip["to"]))

        payload = {"embeds": [{
            "title": "\U0001f504 Signal Changes Detected!",
            "description": "\n".join(lines),
            "color": 0xFF6B35,
            "footer": {"text": "TradingAgents Decision Tracker"},
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }]}
        resp = requests.post(WEBHOOK_URL, json=payload)
        if resp.status_code in (200, 204):
            print("Discord flip alert sent!")

    if not flips:
        print("No signal changes detected")

    return flips


def show_history():
    """Print current decision state for all tracked tickers."""
    history = load_history()
    if not history:
        print("No history yet")
        return
    print("{:<8} {:<8} {}".format("Ticker", "Signal", "Since"))
    print("-" * 30)
    for ticker, info in sorted(history.items()):
        print("{:<8} {:<8} {}".format(ticker, info["decision"], info["date"]))


if __name__ == "__main__":
    import sys
    if "--history" in sys.argv:
        show_history()
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        track_decisions(sys.argv[1])
    else:
        track_decisions()
