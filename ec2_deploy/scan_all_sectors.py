#!/usr/bin/env python3
"""Run all sector watchlists sequentially with Discord summary.

Usage:
    python3.12 scan_all_sectors.py                  # all sectors, turbo
    python3.12 scan_all_sectors.py --profile default # all sectors, default
    python3.12 scan_all_sectors.py --sector tech etf # specific sectors
"""
import sys
import os
import json
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import requests

from config import ALL_WATCHLISTS, get_config

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
RESULTS_DIR = os.path.expanduser("~/TradingAgents/results/daily")

# Import batch analysis function
from batch_analyze import analyze_ticker, get_latest_trading_date


def main():
    profile = "turbo"
    sectors = list(ALL_WATCHLISTS.keys())

    for i, a in enumerate(sys.argv):
        if a == "--profile" and i + 1 < len(sys.argv):
            profile = sys.argv[i + 1]
        if a == "--sector":
            sectors = sys.argv[i + 1:]
            break

    date = get_latest_trading_date()
    config = get_config(profile)
    all_results = {}
    total_start = time.time()

    print("\n" + "=" * 60)
    print("  FULL SECTOR SCAN")
    print("  Date: {} | Profile: {} | Sectors: {}".format(date, profile, len(sectors)))
    print("=" * 60)

    for sector in sectors:
        tickers = ALL_WATCHLISTS.get(sector, [])
        if not tickers:
            print("Unknown sector: {}".format(sector))
            continue

        print("\n--- {} ({} tickers) ---".format(sector.upper(), len(tickers)))
        sector_results = []

        for j, ticker in enumerate(tickers, 1):
            print("[{}/{}] {}...".format(j, len(tickers), ticker))
            start = time.time()
            result = analyze_ticker(ticker, date, config)
            elapsed = time.time() - start
            dec = result["decision"] or result["status"]
            icon = "OK" if result["status"] == "ok" else "FAIL"
            print("  {} {}: {} ({:.0f}s)".format(icon, ticker, dec, elapsed))
            sector_results.append(result)

            if j < len(tickers):
                time.sleep(5)

        all_results[sector] = sector_results

        # Save sector results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        outfile = os.path.join(RESULTS_DIR, "{}_{}_{}.json".format(date, sector, profile))
        with open(outfile, "w") as f:
            json.dump({"date": date, "profile": profile, "sector": sector,
                        "results": sector_results}, f, indent=2)

        # Pause between sectors
        time.sleep(10)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("  FULL SCAN COMPLETE ({:.0f}m {:.0f}s)".format(total_elapsed // 60, total_elapsed % 60))
    print("=" * 60)

    emoji_map = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "HOLD": "\U0001f7e1"}

    # Discord mega-summary
    if WEBHOOK_URL:
        fields = []
        for sector, results in all_results.items():
            lines = []
            for r in results:
                dec = r.get("decision", "ERR")
                emoji = emoji_map.get(dec, "\u26aa")
                lines.append("{} {}".format(emoji, r["ticker"]))
            fields.append({
                "name": sector.upper(),
                "value": "\n".join(lines),
                "inline": True,
            })

        total_buys = sum(1 for s in all_results.values() for r in s if r.get("decision") == "BUY")
        total_sells = sum(1 for s in all_results.values() for r in s if r.get("decision") == "SELL")
        total_holds = sum(1 for s in all_results.values() for r in s if r.get("decision") == "HOLD")
        total_tickers = sum(len(s) for s in all_results.values())

        payload = {"embeds": [{
            "title": "\U0001f30d Full Sector Scan Complete",
            "description": (
                "**Date:** {} | **Profile:** {} | **Time:** {:.0f}m\n\n"
                "\U0001f7e2 {} BUY | \U0001f534 {} SELL | \U0001f7e1 {} HOLD | "
                "Total: {} tickers across {} sectors"
            ).format(date, profile, total_elapsed / 60, total_buys, total_sells,
                     total_holds, total_tickers, len(all_results)),
            "fields": fields,
            "color": 0x5865F2,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }]}
        requests.post(WEBHOOK_URL, json=payload)
        print("Discord full scan summary sent!")

    # Print table
    for sector, results in all_results.items():
        print("\n  {}:".format(sector.upper()))
        for r in results:
            dec = r.get("decision", "N/A")
            print("    {:<8} {}".format(r["ticker"], dec))


if __name__ == "__main__":
    main()
