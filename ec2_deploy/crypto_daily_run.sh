#!/usr/bin/env bash
# crypto_daily_run.sh — Daily crypto watchlist analysis via TradingAgents
#
# Usage:
#   ./crypto_daily_run.sh              # turbo profile (fast)
#   ./crypto_daily_run.sh default      # default profile (balanced)
#   ./crypto_daily_run.sh deep         # deep profile (thorough)
#
# Cron example (daily at 7:30 UTC):
#   30 7 * * * /home/tk578/TradingAgents/ec2_deploy/crypto_daily_run.sh >> /home/tk578/TradingAgents/logs/crypto_daily.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-turbo}"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "=========================================="
echo "  Crypto Daily Run — ${TIMESTAMP}"
echo "  Profile: ${PROFILE}"
echo "=========================================="

cd "${SCRIPT_DIR}"

python3 daily_run.py --sector crypto --profile "${PROFILE}"

echo ""
echo "[${TIMESTAMP}] Crypto daily run complete (profile: ${PROFILE})"
