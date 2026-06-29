#!/usr/bin/env python3
"""Paper trading simulator based on TradingAgents decisions.

Simulates a portfolio following BUY/SELL/HOLD signals.
Tracks positions, P&L, and performance over time.

Risk management features:
- Stop-loss: -5% (hard exit)
- Take-profit: +15% (lock in gains)
- Trailing stop: 10% from peak (protect profits on runners)
- Max drawdown: -10% portfolio-level circuit breaker

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
POSITION_SIZE = 0.20     # 20% of portfolio per position (max 5)
MAX_POSITIONS = 5        # Maximum concurrent positions
STOP_LOSS_PCT = -0.05    # -5% stop-loss per position
TAKE_PROFIT_PCT = 0.15   # +15% take-profit per position
TRAILING_STOP_PCT = 0.10 # 10% trailing stop from peak
MAX_DRAWDOWN_PCT = -0.10 # -10% portfolio-level circuit breaker
MIN_HOLD_DAYS = 1        # Don't sell within 1 day of buying (avoid churn)


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "cash": INITIAL_CASH,
        "positions": {},
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "total_trades": 0,
        "peak_value": INITIAL_CASH,
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

    # Map OVERWEIGHT -> BUY, UNDERWEIGHT -> SELL
    if decision == "OVERWEIGHT":
        decision = "BUY"
    elif decision == "UNDERWEIGHT":
        decision = "SELL"

    if decision == "BUY" and not position:
        # ── Signal accuracy filter ────────────────────────────────────
        try:
            sys.path.insert(0, os.path.expanduser("~/TradingAgents"))
            from signal_filter import SignalFilter
            sf = SignalFilter()
            allowed, reason = sf.should_trade(ticker, "BUY")
            if not allowed:
                print(f"  BLOCKED {ticker}: {reason}")
                return None
            if reason.startswith("WARN"):
                print(f"  ⚠️  {reason}")
        except Exception as e:
            print(f"  ⚠️  Signal filter error (continuing): {e}")

        # ── Market regime check ──────────────────────────────────────
        try:
            from signal_filter import check_market_regime
            regime = check_market_regime()
            if not regime["allow_buys"]:
                print(f"  BLOCKED {ticker}: market regime = {regime['regime']} — {regime['reason']}")
                return None
        except Exception as e:
            print(f"  ⚠️  Market regime check error (continuing): {e}")

        # ── Position weight from signal accuracy ──────────────────────
        position_weight = 1.0
        try:
            from signal_filter import SignalFilter
            sf = SignalFilter()
            position_weight = sf.get_position_weight(ticker)
            if position_weight < 1.0:
                print(f"  📉 Position weight: {position_weight:.2f} (reduced due to accuracy)")
        except Exception:
            pass

        # Enforce max positions limit
        if len(portfolio["positions"]) >= MAX_POSITIONS:
            print(f"  SKIP {ticker}: max positions ({MAX_POSITIONS}) reached")
            return None

        # Enforce position size cap (adjusted by accuracy weight)
        effective_size = POSITION_SIZE * position_weight
        amount = total_value * effective_size
        shares = int(amount / price)
        if shares <= 0:
            print(f"  SKIP {ticker}: cannot afford even 1 share")
            return None
        cost = shares * price
        if cost > total_value * effective_size * 1.05:  # 5% tolerance for rounding
            shares = int(total_value * effective_size / price)
            cost = shares * price
        if portfolio["cash"] < cost:
            print(f"  SKIP {ticker}: insufficient cash (${portfolio['cash']:.0f} < ${cost:.0f})")
            return None

        portfolio["cash"] -= cost
        portfolio["positions"][ticker] = {
            "shares": shares, "avg_price": price,
            "cost": cost, "date": date,
            "peak_price": price,  # Track peak for trailing stop
        }
        trade = {"action": "BUY", "ticker": ticker, "shares": shares,
                 "price": price, "cost": cost, "date": date}
        portfolio["total_trades"] += 1

    elif decision == "SELL" and position:
        trade = _close_position(portfolio, trades, ticker, price, date, reason="SIGNAL")

    if trade:
        trades.append(trade)
        save_trades(trades)
        # Update peak value tracking
        new_value = get_portfolio_value(portfolio)
        if new_value > portfolio.get("peak_value", INITIAL_CASH):
            portfolio["peak_value"] = new_value
        save_portfolio(portfolio)

    return trade


def _close_position(portfolio, trades, ticker, price, date, reason="SIGNAL"):
    """Close a position and record the trade. Returns the trade dict."""
    position = portfolio["positions"][ticker]
    shares = position["shares"]
    revenue = shares * price
    pnl = revenue - position["cost"]
    pnl_pct = round(pnl / position["cost"] * 100, 2)

    portfolio["cash"] += revenue
    del portfolio["positions"][ticker]
    portfolio["total_trades"] += 1

    trade = {
        "action": "SELL", "ticker": ticker, "shares": shares,
        "price": price, "revenue": revenue, "pnl": pnl,
        "pnl_pct": pnl_pct, "date": date, "reason": reason,
    }
    trades.append(trade)

    emoji = "⛔" if reason == "STOP_LOSS" else "🎯" if reason == "TAKE_PROFIT" else "📉" if reason == "TRAILING_STOP" else "💰"
    print(f"  {emoji} {reason} {ticker}: SELL x{shares} @ ${price:.2f} (P&L: ${pnl:.0f}, {pnl_pct}%)")

    return trade


def check_risk_rules(portfolio, date=None):
    """Check all risk management rules on existing positions.

    Returns list of triggered trades (stop-loss, take-profit, trailing stop).
    Also checks portfolio-level max drawdown circuit breaker.
    """
    if date is None:
        date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    trades = load_trades()
    triggered = []

    for ticker, pos in list(portfolio["positions"].items()):
        price = get_current_price(ticker)
        if not price:
            continue

        # Update peak price for trailing stop
        if price > pos.get("peak_price", pos["avg_price"]):
            pos["peak_price"] = price

        # Calculate unrealized P&L percentage
        pnl_pct = (price - pos["avg_price"]) / pos["avg_price"]

        # 1. Hard stop-loss
        if pnl_pct <= STOP_LOSS_PCT:
            trade = _close_position(portfolio, trades, ticker, price, date, reason="STOP_LOSS")
            triggered.append(trade)
            continue

        # 2. Take-profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            trade = _close_position(portfolio, trades, ticker, price, date, reason="TAKE_PROFIT")
            triggered.append(trade)
            continue

        # 3. Trailing stop (from peak, not entry)
        peak = pos.get("peak_price", pos["avg_price"])
        drawdown_from_peak = (price - peak) / peak
        if drawdown_from_peak <= -TRAILING_STOP_PCT:
            trade = _close_position(portfolio, trades, ticker, price, date, reason="TRAILING_STOP")
            triggered.append(trade)
            continue

    # 4. Portfolio-level max drawdown circuit breaker
    total_value = get_portfolio_value(portfolio)
    peak = portfolio.get("peak_value", INITIAL_CASH)
    if peak > 0:
        portfolio_drawdown = (total_value - peak) / peak
        if portfolio_drawdown <= MAX_DRAWDOWN_PCT:
            print(f"  🚨 PORTFOLIO DRAWDOWN {portfolio_drawdown:.1%} — closing all positions!")
            for ticker in list(portfolio["positions"].keys()):
                price = get_current_price(ticker)
                if price:
                    trade = _close_position(portfolio, trades, ticker, price, date, reason="MAX_DRAWDOWN")
                    triggered.append(trade)

    if triggered:
        save_trades(trades)
        new_value = get_portfolio_value(portfolio)
        if new_value > portfolio.get("peak_value", INITIAL_CASH):
            portfolio["peak_value"] = new_value
        save_portfolio(portfolio)

    return triggered


def check_stop_losses(portfolio, date=None):
    """Backward-compatible alias for check_risk_rules."""
    return check_risk_rules(portfolio, date)


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

    # Step 1: Check risk rules (stop-loss, take-profit, trailing stop) before processing new decisions
    print("Checking risk management rules...")
    risk_trades = check_risk_rules(portfolio)
    if risk_trades:
        print(f"  {len(risk_trades)} risk rule(s) triggered")
        executed.extend(risk_trades)

    # Step 2: Process analyst decisions
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
                    reason = f" ({trade.get('reason', '')})" if trade.get("reason") else ""
                    print("  SELL {} x{} @ ${:.2f} (P&L: {}${:.0f}, {}%){}".format(
                        ticker, trade["shares"], trade["price"],
                        emoji, trade["pnl"], trade["pnl_pct"], reason))

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
                reason = f" ({t.get('reason', '')})" if t.get("reason") else ""
                trade_lines.append("{} SELL {} P&L: ${:.0f} ({}%){}".format(
                    e, t["ticker"], t["pnl"], t["pnl_pct"], reason))

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
        print("\n{:<8} {:<8} {:<10} {:<10} {:<6} {}".format(
            "Ticker", "Shares", "Avg Price", "Current", "P&L%", "Trailing"))
        print("-" * 60)
        for ticker, pos in sorted(portfolio["positions"].items()):
            price = get_current_price(ticker)
            if price:
                pos_pnl = (price - pos["avg_price"]) * pos["shares"]
                pos_pct = round((price - pos["avg_price"]) / pos["avg_price"] * 100, 1)
                peak = pos.get("peak_price", pos["avg_price"])
                trail = round((price - peak) / peak * 100, 1)
                sign = "+" if pos_pnl > 0 else ""
                print("{:<8} {:<8} ${:<9.2f} ${:<9.2f} {}{:<5}% {}".format(
                    ticker, pos["shares"], pos["avg_price"], price, sign, pos_pct,
                    f"trail:{trail}%"))
    print()


def reset_portfolio():
    portfolio = {
        "cash": INITIAL_CASH,
        "positions": {},
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "total_trades": 0,
        "peak_value": INITIAL_CASH,
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
