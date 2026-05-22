"""Portfolio context builder — provides position data to trading agents.

Agents currently analyze tickers in isolation with no awareness of existing
positions. This module generates a portfolio context string that gets injected
into the Trader and Portfolio Manager prompts, enabling informed SELL/HOLD
decisions based on unrealized P&L and position age.
"""
import json
import os
from datetime import datetime, timezone

PORTFOLIO_FILE = os.path.expanduser("~/TradingAgents/results/portfolio.json")


def get_portfolio_context(ticker: str) -> str:
    """Build a portfolio context string for the given ticker.

    Returns a multi-line string describing:
    - Whether we hold this ticker, entry price, unrealized P&L
    - How long we've held it
    - Overall portfolio health (cash %, total positions)

    If no position exists, returns a brief "no position" note.
    """
    if not os.path.exists(PORTFOLIO_FILE):
        return "No portfolio data available."

    try:
        with open(PORTFOLIO_FILE) as f:
            portfolio = json.load(f)
    except (json.JSONDecodeError, IOError):
        return "No portfolio data available."

    positions = portfolio.get("positions", {})
    cash = portfolio.get("cash", 0)
    position = positions.get(ticker)

    # Calculate total portfolio value approximation
    total_position_cost = sum(p.get("cost", 0) for p in positions.values())
    total_value = cash + total_position_cost
    cash_pct = round(cash / total_value * 100, 1) if total_value > 0 else 0

    lines = [
        f"**Portfolio Status**: ${total_value:,.0f} total | ${cash:,.0f} cash ({cash_pct}%) | {len(positions)} positions",
    ]

    if position:
        entry_price = position.get("avg_price", 0)
        cost = position.get("cost", 0)
        shares = position.get("shares", 0)
        entry_date = position.get("date", "")

        # Calculate holding period
        try:
            entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
            days_held = (datetime.now(tz=timezone.utc) - entry_dt).days
        except (ValueError, TypeError):
            days_held = "?"

        lines.append(
            f"**CURRENT POSITION in {ticker}**: {shares} shares @ ${entry_price:.2f} avg, "
            f"cost basis ${cost:,.0f}, held {days_held} days."
        )
        lines.append(
            f"→ You OWN this stock. Evaluate whether to HOLD or SELL based on current "
            f"analysis vs entry price ${entry_price:.2f}. Consider taking profits if up >10%, "
            f"or cutting losses if down >5%."
        )
    else:
        lines.append(
            f"**No current position in {ticker}**. Evaluate whether to BUY at current prices "
            f"or skip."
        )

    # Also list other positions for context
    if positions and ticker in positions:
        other_lines = []
        for t, p in positions.items():
            if t == ticker:
                continue
            pct = round((p.get("avg_price", 0) / p.get("cost", 1)) * 100, 1) if p.get("cost") else 0
            other_lines.append(f"  - {t}: {p.get('shares', 0)} shares, ${p.get('cost', 0):,.0f} cost")
        if other_lines:
            lines.append(f"**Other positions** ({len(other_lines)}):")
            lines.extend(other_lines)

    return "\n".join(lines)
