#!/usr/bin/env python3
"""Test paper trader with a single BUY."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ec2_deploy"))
from dotenv import load_dotenv
load_dotenv()
from paper_trader import load_portfolio, execute_trade, show_status

portfolio = load_portfolio()
trade = execute_trade(portfolio, "NVDA", "BUY", "2026-04-17")
if trade:
    print("Trade executed:", trade)
else:
    print("No trade (already in position or no price data)")
show_status(portfolio)
