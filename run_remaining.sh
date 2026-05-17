#!/bin/bash
source /home/tk578/TradingAgents/.venv/bin/activate
export PYTHONUNBUFFERED=1
LOG=/home/tk578/TradingAgents/results/daily_run_remaining.log

for ticker in TSLA MSFT GOOGL AMZN META AMD NFLX; do
  echo "=== Starting $ticker at $(date) ===" >> "$LOG"
  python -u /home/tk578/TradingAgents/ec2_deploy/daily_run.py --ticker "$ticker" >> "$LOG" 2>&1
  echo "=== Finished $ticker at $(date) ===" >> "$LOG"
done
echo "=== ALL DONE at $(date) ===" >> "$LOG"
