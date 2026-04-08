#!/usr/bin/env python3
"""Paper trading simulator based on TradingAgents decisions.

Simulates a portfolio following BUY/SELL/HOLD signals.
Tracks positions, P&L, and performance over time.

Usage:
    python3.12 paper_trader.py                     # process latest decisions
    python3.12 paper_trader.py --status             # show portfolio status
    python3.12 paper_trader.py --reset              # reset portfolio
    python3.12 paper_trader.py --performance        # show performance history
"""
import json
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
import requests

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
PORTFOLIO_FILE = os.path.expanduser("~/TradingAgents/results/portfolio.json")
TRADES_FILE = os.path.expanduser("~/TradingAgents/results/trades.json")

INITIAL_CASH = 100000.0  # $100k paper money
POSITION_SIZE = 0.10     # 10% of portfolio per position


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "cash": INITIAL_CASH,
        "positions": {},
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "total_trades": 0,
    }


def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def save_trades(trades):
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def get_current_price(ticker):
    try:
        t = yf.Ticker(ticker)
        price = t.info.get("currentPrice") or t.info.get("regularMarketPrice")
        if price:
            return float(price)
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def execute_trade(portfolio, ticker, decision, date):
    """Execute a paper trade based on the decision."""
    price = get_current_price(ticker)
    if not price:
        return None

    trades = load_trades()
    trade = None
    position = portfolio["positions"].get(ticker)
    total_value = get_portfolio_value(portfolio)

    if decision == "BUY" and not position:
        # Open new position — 10% of portfolio
        amount = total_value * POSITION_SIZE
        shares = int(amount / price)
        if shares > 0 and portfolio["cash"] >= shares * price:
            cost = shares * price
            portfolio["cash"] -= cost
            portfolio["positions"][ticker] = {
                "shares": shares, "avg_price": price,
                "cost": cost, "date": date
            }
            trade = {"action": "BUY", "ticker": ticker, "shares": shares,
                     "price": price, "cost": cost, "date": date}
            portfolio["total_trades"] += 1

    elif decision == "SELL" and position:
        # Close position
        shares = position["shares"]
        revenue = shares * price
        pnl = revenue - position["cost"]
        pnl_pct = round(pnl / position["cost"] * 100, 2)
        portfolio["cash"] += revenue
        del portfolio["positions"][ticker]
        trade = {"action": "SELL", "ticker": ticker, "shares": shares,
                 "price": price, "revenue": revenue, "pnl": pnl,
                 "pnl_pct": pnl_pct, "date": date}
        portfolio["total_trades"] += 1

    if trade:
        trades.append(trade)
        save_trades(trades)
        save_portfolio(portfolio)

    return trade


def get_portfolio_value(portfolio):
    """Calculate total portfolio value including positions."""
    value = portfolio["cash"]
    for ticker, pos in portfolio["positions"].items():
        price = get_current_price(ticker)
        if price:
            value += pos["shares"] * price
        else:
            value += pos["cost"]  # fallback to cost basis
    return value


def process_decisions(json_path=None):
    """Process latest decisions and execute paper trades."""
    import glob
    results_dir = os.path.expanduser("~/TradingAgents/results/daily")

    if json_path:
        files = [json_path]
    else:
        files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
        if not files:
            print("No results to process")
            return
        files = [files[-1]]

    portfolio = load_portfolio()
    executed = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        date = data["date"]

        for r in data["results"]:
            ticker = r["ticker"]
            decision = r.get("decision")
            if not decision or r.get("status") != "ok":
                continue

            trade = execute_trade(portfolio, ticker, decision, date)
            if trade:
                executed.append(trade)
                if trade["action"] == "BUY":
                    print("  BUY  {} x{} @ ${:.2f} (${:.0f})".format(
                        ticker, trade["shares"], trade["price"], trade["cost"]))
                else:
                    emoji = "+" if trade["pnl"] > 0 else ""
                    print("  SELL {} x{} @ ${:.2f} (P&L: {}${:.0f}, {}%)".format(
                        ticker, trade["shares"], trade["price"],
                        emoji, trade["pnl"], trade["pnl_pct"]))

    if not executed:
        print("No trades executed (positions already aligned or HOLD)")

    # Show status
    show_status(portfolio)

    # Discord update
    if executed and WEBHOOK_URL:
        total_value = get_portfolio_value(portfolio)
        pnl = total_value - INITIAL_CASH
        pnl_pct = round(pnl / INITIAL_CASH * 100, 2)
        emoji = "\U0001f4b0" if pnl > 0 else "\U0001f4c9"

        trade_lines = []
        for t in executed:
            if t["action"] == "BUY":
                trade_lines.append("\U0001f7e2 BUY {} x{} @ ${:.2f}".format(
                    t["ticker"], t["shares"], t["price"]))
            else:
                e = "\U0001f7e2" if t["pnl"] > 0 else "\U0001f534"
                trade_lines.append("{} SELL {} P&L: ${:.0f} ({}%)".format(
                    e, t["ticker"], t["pnl"], t["pnl_pct"]))

        sign = "+" if pnl > 0 else ""
        payload = {"embeds": [{
            "title": "{} Paper Portfolio: ${:,.0f} ({}{}%)".format(emoji, total_value, sign, pnl_pct),
            "description": "**Trades:**\n{}\n\n**Cash:** ${:,.0f}\n**Positions:** {}".format(
                "\n".join(trade_lines), portfolio["cash"], len(portfolio["positions"])),
            "color": 0x57F287 if pnl > 0 else 0xED4245,
        }]}
        requests.post(WEBHOOK_URL, json=payload)


def show_status(portfolio=None):
    if portfolio is None:
        portfolio = load_portfolio()

    total_value = get_portfolio_value(portfolio)
    pnl = total_value - INITIAL_CASH
    pnl_pct = round(pnl / INITIAL_CASH * 100, 2)

    print("\n" + "=" * 50)
    print("PAPER PORTFOLIO")
    print("=" * 50)
    sign = "+" if pnl > 0 else ""
    print("Total Value: ${:,.0f} ({}{}%)".format(total_value, sign, pnl_pct))
    print("Cash:        ${:,.0f}".format(portfolio["cash"]))
    print("Positions:   {}".format(len(portfolio["positions"])))
    print("Total Trades: {}".format(portfolio["total_trades"]))

    if portfolio["positions"]:
        print("\n{:<8} {:<8} {:<10} {:<10} {}".format(
            "Ticker", "Shares", "Avg Price", "Current", "P&L"))
        print("-" * 50)
        for ticker, pos in sorted(portfolio["positions"].items()):
            price = get_current_price(ticker)
            if price:
                pos_pnl = (price - pos["avg_price"]) * pos["shares"]
                pos_pct = round((price - pos["avg_price"]) / pos["avg_price"] * 100, 1)
                sign = "+" if pos_pnl > 0 else ""
                print("{:<8} {:<8} ${:<9.2f} ${:<9.2f} {}${:.0f} ({}%)".format(
                    ticker, pos["shares"], pos["avg_price"], price, sign, pos_pnl, pos_pct))
    print()


def reset_portfolio():
    portfolio = {
        "cash": INITIAL_CASH,
        "positions": {},
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "total_trades": 0,
    }
    save_portfolio(portfolio)
    if os.path.exists(TRADES_FILE):
        os.remove(TRADES_FILE)
    print("Portfolio reset to ${:,.0f}".format(INITIAL_CASH))


if __name__ == "__main__":
    if "--status" in sys.argv:
        show_status()
    elif "--reset" in sys.argv:
        reset_portfolio()
    elif "--performance" in sys.argv:
        trades = load_trades()
        for t in trades:
            print(t)
    else:
        process_decisions()
