#!/usr/bin/env python3
"""Lightweight backtesting script for TradingAgents.

Simulates the TradingAgents strategy on historical data using technical
indicators (RSI, MACD, SMA, Bollinger Bands) as a proxy for LLM decisions.

Usage:
    python backtest.py --tickers NVDA AAPL TSLA --start 2024-01-01 --end 2025-01-01
    python backtest.py --sector tech --start 2024-06-01 --end 2025-06-01
    python backtest.py --ticker SPY --start 2023-01-01 --end 2025-01-01 --plot
    python backtest.py --sector etf --start 2024-01-01 --end 2025-01-01 --plot
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants (mirrors ec2_deploy/paper_trader.py)
# ---------------------------------------------------------------------------
INITIAL_CASH = 100_000.0
POSITION_SIZE_PCT = 0.10  # 10% of portfolio per position

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "backtest_results")

# Sector watchlists (mirrors ec2_deploy/config.py)
WATCHLISTS = {
    "tech": ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "PLTR"],
    "etf": ["SPY", "QQQ", "IWM", "DIA", "ARKK"],
    "finance": ["JPM", "GS", "V", "MA", "BRK-B"],
    "health": ["UNH", "JNJ", "LLY", "PFE", "ABBV"],
    "energy": ["XOM", "CVX", "NEE", "ENPH", "FSLR"],
    "crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
        "DOT-USD", "MATIC-USD", "ADA-USD", "XRP-USD", "DOGE-USD",
    ],
}

# ============================================================================
# Technical Indicator Computation (pure pandas / numpy — no TA-Lib needed)
# ============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = compute_sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


# ============================================================================
# Signal Generation — proxy for TradingAgents LLM decisions
# ============================================================================

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and BUY/SELL/HOLD signals to OHLCV DataFrame.

    Decision rules (simplified proxy for the multi-agent LLM debate):
      BUY  — RSI oversold (<35) AND price above SMA200  [buy-the-dip in uptrend]
           OR MACD bullish crossover (histogram crosses above zero)
           OR price touches lower Bollinger Band while above SMA50
      SELL — RSI overbought (>70) AND price below SMA200
           OR MACD bearish crossover (histogram crosses below zero)
           OR price touches upper Bollinger Band while below SMA50
      HOLD — everything else
    """
    close = df["Close"]

    # Indicators
    df["RSI"] = compute_rsi(close)
    df["SMA_20"] = compute_sma(close, 20)
    df["SMA_50"] = compute_sma(close, 50)
    df["SMA_200"] = compute_sma(close, 200)
    macd_line, signal_line, histogram = compute_macd(close)
    df["MACD_hist"] = histogram
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close)
    df["BB_upper"] = bb_upper
    df["BB_lower"] = bb_lower

    # Previous histogram for crossover detection
    hist_prev = histogram.shift(1)

    # Signal conditions
    buy_cond = (
        # RSI oversold in long-term uptrend
        ((df["RSI"] < 35) & (close > df["SMA_200"]))
        # MACD bullish crossover
        | ((histogram > 0) & (hist_prev <= 0))
        # Price at lower BB while still above SMA50
        | ((close <= df["BB_lower"]) & (close > df["SMA_50"]))
    )

    sell_cond = (
        # RSI overbought in long-term downtrend
        ((df["RSI"] > 70) & (close < df["SMA_200"]))
        # MACD bearish crossover
        | ((histogram < 0) & (hist_prev >= 0))
        # Price at upper BB while below SMA50
        | ((close >= df["BB_upper"]) & (close < df["SMA_50"]))
    )

    df["signal"] = "HOLD"
    df.loc[buy_cond, "signal"] = "BUY"
    df.loc[sell_cond, "signal"] = "SELL"

    # Avoid conflicting signals on the same bar — SELL wins over BUY
    df.loc[buy_cond & sell_cond, "signal"] = "HOLD"

    return df


# ============================================================================
# Backtest Engine
# ============================================================================

def run_backtest(ticker: str, df: pd.DataFrame) -> dict:
    """Run backtest for a single ticker. Returns results dict."""
    cash = INITIAL_CASH
    position = None  # {"shares": int, "avg_price": float, "cost": float}
    trades = []
    portfolio_values = []
    dates = []

    for date, row in df.iterrows():
        price = row["Close"]
        signal = row["signal"]

        # --- Execute signal ---
        if signal == "BUY" and position is None:
            total_value = cash + (position["shares"] * position["avg_price"] if position else 0)
            alloc = total_value * POSITION_SIZE_PCT
            shares = int(alloc / price)
            if shares > 0 and cash >= shares * price:
                cost = shares * price
                cash -= cost
                position = {"shares": shares, "avg_price": price, "cost": cost}
                trades.append({
                    "date": str(date.date()) if hasattr(date, "date") else str(date),
                    "action": "BUY", "ticker": ticker,
                    "shares": shares, "price": round(price, 2), "cost": round(cost, 2),
                })

        elif signal == "SELL" and position is not None:
            revenue = position["shares"] * price
            pnl = revenue - position["cost"]
            pnl_pct = round(pnl / position["cost"] * 100, 2) if position["cost"] else 0
            cash += revenue
            trades.append({
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "action": "SELL", "ticker": ticker,
                "shares": position["shares"], "price": round(price, 2),
                "revenue": round(revenue, 2), "pnl": round(pnl, 2), "pnl_pct": pnl_pct,
            })
            position = None

        # --- Track portfolio value ---
        pos_value = position["shares"] * price if position else 0
        portfolio_values.append(cash + pos_value)
        dates.append(date)

    # Final metrics
    final_value = portfolio_values[-1] if portfolio_values else INITIAL_CASH
    total_return_pct = round((final_value - INITIAL_CASH) / INITIAL_CASH * 100, 2)

    # Buy-and-hold benchmark (invest same POSITION_SIZE_PCT at start)
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    bh_shares = int((INITIAL_CASH * POSITION_SIZE_PCT) / start_price)
    bh_cost = bh_shares * start_price
    bh_final = (INITIAL_CASH - bh_cost) + bh_shares * end_price
    bh_return_pct = round((bh_final - INITIAL_CASH) / INITIAL_CASH * 100, 2)

    # Win rate on closed trades
    closed = [t for t in trades if t["action"] == "SELL"]
    wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
    win_rate = round(wins / len(closed) * 100, 1) if closed else 0

    return {
        "ticker": ticker,
        "start_date": str(df.index[0].date()) if hasattr(df.index[0], "date") else str(df.index[0]),
        "end_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
        "initial_cash": INITIAL_CASH,
        "final_value": round(final_value, 2),
        "total_return_pct": total_return_pct,
        "buyhold_return_pct": bh_return_pct,
        "alpha_pct": round(total_return_pct - bh_return_pct, 2),
        "total_trades": len(trades),
        "buys": sum(1 for t in trades if t["action"] == "BUY"),
        "sells": len(closed),
        "win_rate": win_rate,
        "trades": trades,
        "portfolio_curve": {
            str(d.date() if hasattr(d, "date") else d): round(v, 2)
            for d, v in zip(dates, portfolio_values)
        },
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_results(results: list[dict], out_dir: str):
    """Generate equity curve + buy-and-hold comparison chart."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # --- Equity curves ---
    ax = axes[0]
    for r in results:
        curve = r["portfolio_curve"]
        dates = [pd.Timestamp(d) for d in curve.keys()]
        values = list(curve.values())
        ax.plot(dates, values, label=f"{r['ticker']} Strategy ({r['total_return_pct']:+.1f}%)", linewidth=1.5)

    ax.axhline(y=INITIAL_CASH, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax.set_title("TradingAgents Backtest — Strategy Equity Curves")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Summary table ---
    ax2 = axes[1]
    ax2.axis("off")
    table_data = []
    for r in results:
        table_data.append([
            r["ticker"],
            f"{r['total_return_pct']:+.1f}%",
            f"{r['buyhold_return_pct']:+.1f}%",
            f"{r['alpha_pct']:+.1f}%",
            str(r["total_trades"]),
            f"{r['win_rate']:.0f}%",
        ])
    col_labels = ["Ticker", "Strategy", "Buy&Hold", "Alpha", "Trades", "Win Rate"]
    table = ax2.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "backtest_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="TradingAgents lightweight backtester")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", nargs="+", help="List of tickers, e.g. NVDA AAPL")
    group.add_argument("--ticker", help="Single ticker")
    group.add_argument("--sector", choices=list(WATCHLISTS.keys()), help="Predefined sector watchlist")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--plot", action="store_true", help="Generate equity curve chart")
    p.add_argument("--output", default=None, help="Output directory (default: backtest_results/)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve tickers
    if args.tickers:
        tickers = args.tickers
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = WATCHLISTS[args.sector]

    out_dir = args.output or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    start_date = args.start
    end_date = args.end

    print("=" * 65)
    print("  TradingAgents Backtester")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Period:  {start_date} -> {end_date}")
    print(f"  Capital: ${INITIAL_CASH:,.0f} | Position Size: {POSITION_SIZE_PCT*100:.0f}%")
    print("=" * 65)

    all_results = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")

        # Download data (add 300 days buffer for SMA 200 warm-up)
        print("  Downloading data...")
        try:
            buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=400)
            df = yf.download(
                ticker,
                start=buffer_start.strftime("%Y-%m-%d"),
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")
            continue

        if df.empty or len(df) < 250:
            print(f"  SKIP — insufficient data ({len(df)} rows)")
            continue

        # Handle MultiIndex columns (yfinance >= 0.2.36)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Generate signals
        print("  Computing indicators & signals...")
        df = generate_signals(df)

        # Filter to requested date range (discard warm-up buffer)
        df = df.loc[start_date:end_date]

        if df.empty:
            print(f"  SKIP — no data in requested range")
            continue

        buys = (df["signal"] == "BUY").sum()
        sells = (df["signal"] == "SELL").sum()
        holds = (df["signal"] == "HOLD").sum()
        print(f"  Signals: {buys} BUY | {sells} SELL | {holds} HOLD")

        # Run backtest
        result = run_backtest(ticker, df)
        all_results.append(result)

        sign = "+" if result["total_return_pct"] >= 0 else ""
        print(f"  Result:  ${result['final_value']:,.0f} ({sign}{result['total_return_pct']}%)"
              f"  |  Buy&Hold: {result['buyhold_return_pct']:+}%"
              f"  |  Alpha: {result['alpha_pct']:+}%"
              f"  |  Trades: {result['total_trades']}"
              f"  |  Win: {result['win_rate']}%")

    # --- Summary ---
    if not all_results:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Ticker':<10} {'Strategy':>10} {'B&H':>10} {'Alpha':>10} {'Trades':>8} {'WinRate':>10}")
    print("  " + "-" * 58)

    avg_return = 0
    avg_alpha = 0
    for r in all_results:
        print(f"  {r['ticker']:<10} {r['total_return_pct']:>+9.1f}% {r['buyhold_return_pct']:>+9.1f}%"
              f" {r['alpha_pct']:>+9.1f}% {r['total_trades']:>8} {r['win_rate']:>9.0f}%")
        avg_return += r["total_return_pct"]
        avg_alpha += r["alpha_pct"]

    n = len(all_results)
    print("  " + "-" * 58)
    print(f"  {'AVERAGE':<10} {avg_return/n:>+9.1f}% {'':>10} {avg_alpha/n:>+9.1f}%")
    print()

    # Save JSON results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"backtest_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({"run_date": ts, "start": start_date, "end": end_date,
                    "initial_cash": INITIAL_CASH, "results": all_results}, f, indent=2)
    print(f"  Results saved: {json_path}")

    # Plot
    if args.plot:
        chart_path = plot_results(all_results, out_dir)
        print(f"  Chart saved:   {chart_path}")

    print("=" * 65)


if __name__ == "__main__":
    main()
