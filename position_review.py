#!/usr/bin/env python3
"""Position review script — automated portfolio health checks.

Reads current positions from results/portfolio.json, calculates P&L,
and prints a risk report with recommendations.

Usage:
    python3 position_review.py
    python3 position_review.py --json    # output as JSON
"""
import json
import os
import sys

# ── Sector mapping ─────────────────────────────────────────────────────────
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "AMD": "Technology", "META": "Technology",
    "AMZN": "Consumer Cyclical", "NFLX": "Communication Services",
    "TSLA": "Consumer Cyclical", "PLTR": "Technology",
    "BTC": "Crypto", "ETH": "Crypto", "SOL": "Crypto",
}

PORTFOLIO_FILE = os.path.expanduser("~/TradingAgents/results/portfolio.json")
TRADES_FILE = os.path.expanduser("~/TradingAgents/results/trades.json")
INITIAL_CASH = 100000.0


def get_current_price(ticker):
    """Fetch current price via yfinance."""
    try:
        import yfinance as yf
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


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"cash": INITIAL_CASH, "positions": {}, "total_trades": 0, "peak_value": INITIAL_CASH}


def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def review_portfolio():
    """Run portfolio review and return report dict."""
    portfolio = load_portfolio()
    trades = load_trades()
    positions = portfolio.get("positions", {})
    cash = portfolio.get("cash", 0)

    report = {
        "cash": cash,
        "peak_value": portfolio.get("peak_value", INITIAL_CASH),
        "total_trades": portfolio.get("total_trades", 0),
        "positions": [],
        "at_risk": [],
        "hold": [],
        "sector_exposure": {},
        "closed_trades_summary": _summarize_closed_trades(trades),
    }

    total_position_value = 0

    for ticker, pos in sorted(positions.items()):
        current = get_current_price(ticker)
        entry = pos["avg_price"]
        shares = pos["shares"]
        cost = pos["cost"]
        peak = pos.get("peak_price", entry)

        if current:
            pnl_pct = (current - entry) / entry * 100
            position_value = shares * current
            unrealized_pnl = position_value - cost
        else:
            pnl_pct = 0
            position_value = cost
            unrealized_pnl = 0

        total_position_value += position_value

        sector = SECTOR_MAP.get(ticker, "Unknown")
        report["sector_exposure"][sector] = report["sector_exposure"].get(sector, 0) + position_value

        pos_info = {
            "ticker": ticker,
            "shares": shares,
            "entry_price": entry,
            "current_price": current,
            "peak_price": peak,
            "cost": cost,
            "market_value": position_value,
            "pnl_pct": round(pnl_pct, 1),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "sector": sector,
            "date": pos.get("date", "unknown"),
        }
        report["positions"].append(pos_info)

        # Risk classification
        if pnl_pct < -8:
            pos_info["risk_level"] = "CRITICAL"
            pos_info["recommendation"] = (
                f"⛔ {ticker} down {pnl_pct:.1f}% — near stop-loss, "
                "flag for immediate review or exit"
            )
            report["at_risk"].append(pos_info)
        elif pnl_pct < -5:
            pos_info["risk_level"] = "WARNING"
            pos_info["recommendation"] = (
                f"⚠️ {ticker} down {pnl_pct:.1f}% — approaching stop-loss, monitor closely"
            )
            report["at_risk"].append(pos_info)
        else:
            pos_info["risk_level"] = "OK"
            pos_info["recommendation"] = f"✅ {ticker} within tolerance ({pnl_pct:+.1f}%)"
            report["hold"].append(pos_info)

    # Portfolio totals
    total_value = cash + total_position_value
    report["total_value"] = round(total_value, 2)
    report["total_pnl"] = round(total_value - INITIAL_CASH, 2)
    report["total_pnl_pct"] = round((total_value - INITIAL_CASH) / INITIAL_CASH * 100, 1)
    report["cash_pct"] = round(cash / total_value * 100, 1) if total_value > 0 else 100
    report["positions_value"] = round(total_position_value, 2)

    # Sector concentration warnings
    report["sector_warnings"] = []
    for sector, value in report["sector_exposure"].items():
        pct = value / total_value * 100 if total_value > 0 else 0
        report["sector_exposure"][sector] = {"value": round(value, 2), "pct": round(pct, 1)}
        if pct > 40:
            report["sector_warnings"].append(
                f"🔴 HIGH CONCENTRATION: {sector} at {pct:.1f}% of portfolio"
            )
        elif pct > 25:
            report["sector_warnings"].append(
                f"🟡 MODERATE: {sector} at {pct:.1f}% of portfolio"
            )

    return report


def _summarize_closed_trades(trades):
    sells = [t for t in trades if t["action"] == "SELL"]
    if not sells:
        return {"count": 0}
    wins = [t for t in sells if t.get("pnl", 0) > 0]
    losses = [t for t in sells if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in sells)
    return {
        "count": len(sells),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0,
    }


def print_report(report):
    """Print a human-readable report."""
    print("=" * 60)
    print("  PORTFOLIO HEALTH REVIEW")
    print("=" * 60)

    # Header
    sign = "+" if report["total_pnl"] >= 0 else ""
    print(f"\n  Total Value: ${report['total_value']:,.2f}")
    print(f"  P&L:         {sign}${report['total_pnl']:,.2f} ({sign}{report['total_pnl_pct']}%)")
    print(f"  Cash:        ${report['cash']:,.2f} ({report['cash_pct']}%)")
    print(f"  Positions:   ${report['positions_value']:,.2f} ({100 - report['cash_pct']:.1f}%)")
    print(f"  Peak:        ${report['peak_value']:,.2f}")

    # Positions at risk
    print(f"\n{'─' * 60}")
    print(f"  ⚠️  POSITIONS AT RISK ({len(report['at_risk'])})")
    print(f"{'─' * 60}")
    if report["at_risk"]:
        for p in report["at_risk"]:
            print(f"  {p['recommendation']}")
            cur_str = f"${p['current_price']:.2f}" if p['current_price'] else 'N/A'
            print(f"     Entry: ${p['entry_price']:.2f} → {cur_str} | "
                  f"P&L: {p['pnl_pct']:+.1f}% (${p['unrealized_pnl']:+,.0f})")
    else:
        print("  No positions at risk 🎉")

    # Positions to hold
    print(f"\n{'─' * 60}")
    print(f"  ✅ POSITIONS TO HOLD ({len(report['hold'])})")
    print(f"{'─' * 60}")
    if report["hold"]:
        for p in report["hold"]:
            print(f"  {p['recommendation']}")
            if p["current_price"]:
                print(f"     ${p['entry_price']:.2f} → ${p['current_price']:.2f}")
    else:
        print("  No positions in hold zone")

    # Cash available
    print(f"\n{'─' * 60}")
    print("  💰 CASH AVAILABLE FOR NEW OPPORTUNITIES")
    print(f"{'─' * 60}")
    new_position_size = report["cash"] * 0.20  # 20% position sizing
    max_new_positions = min(5 - len(report["positions"]), int(report["cash"] / (INITIAL_CASH * 0.20)))
    print(f"  Cash: ${report['cash']:,.2f}")
    print(f"  Max position size (20%): ${new_position_size:,.2f}")
    print(f"  Available slots: {max_new_positions}")

    # Sector concentration
    print(f"\n{'─' * 60}")
    print("  📊 SECTOR EXPOSURE")
    print(f"{'─' * 60}")
    for sector, info in sorted(report["sector_exposure"].items()):
        print(f"  {sector}: ${info['value']:,.0f} ({info['pct']:.1f}%)")
    if report["sector_warnings"]:
        print()
        for w in report["sector_warnings"]:
            print(f"  {w}")

    # Closed trade summary
    cs = report["closed_trades_summary"]
    if cs["count"] > 0:
        print(f"\n{'─' * 60}")
        print("  📈 CLOSED TRADE SUMMARY")
        print(f"{'─' * 60}")
        print(f"  Closed trades: {cs['count']}")
        print(f"  Win rate: {cs['win_rate']}% ({cs['wins']}W / {cs['losses']}L)")
        print(f"  Total P&L: ${cs['total_pnl']:+,.0f}")
        if cs["avg_win"]:
            print(f"  Avg win:  ${cs['avg_win']:+,.0f}")
        if cs["avg_loss"]:
            print(f"  Avg loss: ${cs['avg_loss']:+,.0f}")

    print(f"\n{'=' * 60}\n")


def main():
    if "--json" in sys.argv:
        report = review_portfolio()
        print(json.dumps(report, indent=2))
    else:
        report = review_portfolio()
        print_report(report)


if __name__ == "__main__":
    main()
