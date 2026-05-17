#!/usr/bin/env python3
"""TradingAgents persistent daemon.

Runs as a systemd service providing:
1. Scheduled daily analysis (market open)
2. Discord webhook notifications
3. Dashboard generation

Configuration via environment variables:
  DISCORD_WEBHOOK_URL  - Discord webhook for notifications
  ANALYSIS_SCHEDULE    - Cron-like hours (default: "9,12,16" = 9AM, noon, 4PM UTC)
  ANALYSIS_PROFILE     - Analysis profile: turbo|default|deep (default: turbo)
  ANALYSIS_TICKERS     - Space-separated tickers (default: from config WATCHLIST)
"""
import os
import sys
import time
import json
import signal
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("trading-daemon")

# Graceful shutdown
running = True
def shutdown(signum, frame):
    global running
    log.info("Received signal %s, shutting down...", signum)
    running = False

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

# Load env
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = BASE_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "daily"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add paths
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(BASE_DIR))

import requests

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
SCHEDULE_HOURS = [int(h) for h in os.getenv("ANALYSIS_SCHEDULE", "9,16").split(",")]
PROFILE = os.getenv("ANALYSIS_PROFILE", "turbo")
INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "15"))

log.info("TradingAgents Daemon starting")
log.info("  Webhook: %s", "configured" if WEBHOOK_URL else "NOT SET")
log.info("  Schedule: %s UTC", SCHEDULE_HOURS)
log.info("  Profile: %s", PROFILE)
log.info("  Check interval: %d min", INTERVAL_MINUTES)


def send_discord(title, description, color=0x5865F2):
    """Send Discord webhook notification."""
    if not WEBHOOK_URL:
        log.warning("No DISCORD_WEBHOOK_URL configured, skipping notification")
        return
    try:
        payload = {"embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }]}
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            log.info("Discord notification sent: %s", title)
        else:
            log.warning("Discord returned %s: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        log.error("Discord notification failed: %s", e)


def run_daily_analysis():
    """Run the daily analysis for all watchlist tickers."""
    from config import WATCHLIST, get_config
    
    config = get_config(PROFILE)
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    
    log.info("Starting daily analysis for %s (profile=%s)", date_str, PROFILE)
    send_discord(
        "🚀 Daily Analysis Starting",
        "Date: {} | Profile: {} | Tickers: {}".format(date_str, PROFILE, len(WATCHLIST)),
        0x5865F2
    )
    
    try:
        from batch_analyze import analyze_ticker
    except ImportError as e:
        log.error("Cannot import batch_analyze: %s", e)
        send_discord("❌ Import Error", str(e), 0xED4245)
        return
    
    all_results = []
    for i, ticker in enumerate(WATCHLIST, 1):
        if not running:
            log.info("Shutdown requested, stopping analysis")
            break
        log.info("[%d/%d] Analyzing %s...", i, len(WATCHLIST), ticker)
        try:
            result = analyze_ticker(ticker, date_str, config)
            dec = result.get("decision", "ERROR")
            log.info("  %s -> %s", ticker, dec)
            all_results.append(result)
        except Exception as e:
            log.error("  %s failed: %s", ticker, e)
            all_results.append({"ticker": ticker, "date": date_str, "decision": None, "status": "error", "error": str(e)})
        
        # Rate limit between tickers
        if i < len(WATCHLIST):
            time.sleep(5)
    
    # Save results
    result_file = RESULTS_DIR / "{}_daily_{}.json".format(date_str, PROFILE)
    with open(result_file, "w") as f:
        json.dump({
            "date": date_str,
            "profile": PROFILE,
            "results": all_results,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }, f, indent=2)
    log.info("Results saved to %s", result_file)
    
    # Run decision tracker if available
    try:
        from decision_tracker import track_decisions
        track_decisions(str(result_file))
    except ImportError:
        pass
    
    # Summary
    buys = sum(1 for r in all_results if r.get("decision") == "BUY")
    sells = sum(1 for r in all_results if r.get("decision") == "SELL")
    holds = sum(1 for r in all_results if r.get("decision") == "HOLD")
    errors = sum(1 for r in all_results if r.get("status") == "error")
    
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    lines = []
    for r in all_results:
        dec = r.get("decision", "ERR")
        emoji = emoji_map.get(dec, "⚪")
        lines.append("{} **{}**: {}".format(emoji, r["ticker"], dec))
    
    send_discord(
        "📊 Daily Analysis Complete",
        "🟢 {} BUY | 🔴 {} SELL | 🟡 {} HOLD | ❌ {} errors\n\n{}".format(
            buys, sells, holds, errors, "\n".join(lines)
        ),
        0x57F287 if errors == 0 else 0xFEE75C
    )
    
    # Generate dashboard
    try:
        from dashboard import generate_dashboard
        generate_dashboard()
        log.info("Dashboard generated")
    except Exception as e:
        log.warning("Dashboard generation failed: %s", e)
    
    # Run paper trader
    try:
        import subprocess
        subprocess.run([sys.executable, str(BASE_DIR / "paper_trader.py")],
                      cwd=str(BASE_DIR), timeout=60)
    except Exception as e:
        log.warning("Paper trader failed: %s", e)


def should_run_now():
    """Check if we should run analysis at current time."""
    now = datetime.now(tz=timezone.utc)
    current_hour = now.hour
    
    if current_hour not in SCHEDULE_HOURS:
        return False
    
    # Check if we already ran this hour today
    date_str = now.strftime("%Y-%m-%d")
    marker_file = RESULTS_DIR / ".last_run_{}_{}".format(date_str, current_hour)
    if marker_file.exists():
        return False
    
    return True


def mark_run():
    """Mark that analysis has run for this hour."""
    now = datetime.now(tz=timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    marker_file = RESULTS_DIR / ".last_run_{}_{}".format(date_str, now.hour)
    marker_file.write_text(now.isoformat())


def main():
    """Main daemon loop."""
    send_discord("🤖 TradingAgents Daemon Online",
                 "Schedule: {} UTC | Profile: {}".format(SCHEDULE_HOURS, PROFILE),
                 0x5865F2)
    
    last_health_check = 0
    
    while running:
        try:
            now = time.time()
            
            if should_run_now():
                log.info("Scheduled analysis time reached, running...")
                run_daily_analysis()
                mark_run()
            
            # Health check / heartbeat every 6 hours
            if now - last_health_check > 6 * 3600:
                last_health_check = now
                import shutil
                disk = shutil.disk_usage("/")
                disk_pct = round(disk.used / disk.total * 100, 1)
                log.info("Health check: disk=%.1f%% used", disk_pct)
            
        except Exception as e:
            log.error("Main loop error: %s\n%s", e, traceback.format_exc())
            send_discord("⚠️ Daemon Error", str(e)[:500], 0xED4245)
        
        # Sleep in small increments for responsive shutdown
        for _ in range(INTERVAL_MINUTES * 60 // 10):
            if not running:
                break
            time.sleep(10)
    
    send_discord("🔴 TradingAgents Daemon Stopping", "Shutdown at {}".format(
        datetime.now(tz=timezone.utc).isoformat()), 0xED4245)
    log.info("Daemon stopped")


if __name__ == "__main__":
    main()
