#!/usr/bin/env python3
"""Walk-forward validation backtest engine for TradingAgents.

Implements the arXiv 2605.19337 recommendation: LLM trading systems need
walk-forward validation with transaction costs for reproducibility.

This module backtests historical LLM-generated signals against real price
data, applying the same risk rules as paper_trader.py (stop-loss, take-profit,
trailing stop) plus realistic transaction costs (commission + slippage + spread).

Walk-forward approach: the validator steps through signal dates in rolling
windows, loading prices as-of each signal date so there is no look-ahead bias.

Usage:
    python ec2_deploy/backtest_engine.py --start 2026-05-15 --end 2026-05-28
    python ec2_deploy/backtest_engine.py --start 2026-05-15 --end 2026-05-28 --benchmark SPY
    python ec2_deploy/backtest_engine.py --latest 14
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRANSACTION_COST_PCT: float = 0.0005   # 0.05% per trade (~IBKR $0.005/share)
SPREAD_PCT: float = 0.0001              # 0.01% bid-ask spread (liquid large caps)
WINDOW_SIZE: int = 5                    # days of signals per walk-forward window
STEP_SIZE: int = 1                      # days to step forward between windows
INITIAL_CASH: float = 100_000.0
POSITION_SIZE: float = 0.10             # 10% of portfolio per position
MAX_POSITIONS: int = 8
STOP_LOSS_PCT: float = -0.05            # -5% hard stop-loss
TAKE_PROFIT_PCT: float = 0.15           # +15% take-profit
TRAILING_STOP_PCT: float = 0.10         # 10% trailing stop from peak
RISK_FREE_RATE: float = 0.045           # annualized risk-free rate for Sharpe
MAX_HOLD_DAYS: int = 10                 # forced exit after N trading days
MIN_CONFIDENCE: float = 0.5             # minimum confidence to trade (if available)

RESULTS_DIR: str = os.path.expanduser("~/TradingAgents/results/daily")
OUTPUT_DIR: str = os.path.expanduser("~/TradingAgents/results/backtest")

logger = logging.getLogger("backtest_engine")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """Open position in the backtest portfolio."""

    ticker: str
    shares: int
    avg_price: float
    cost: float
    entry_date: str
    peak_price: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_price == 0.0:
            self.peak_price = self.avg_price


@dataclass
class TradeRecord:
    """Completed trade."""

    action: str        # BUY or SELL
    ticker: str
    shares: int
    price: float
    amount: float      # cost for BUY, revenue for SELL
    date: str
    reason: str = "SIGNAL"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    txn_cost: float = 0.0


@dataclass
class WindowResult:
    """Result of one walk-forward window."""

    window_start: str
    window_end: str
    trades: list[TradeRecord] = field(default_factory=list)
    start_equity: float = 0.0
    end_equity: float = 0.0
    signal_count: int = 0


# ---------------------------------------------------------------------------
# WalkForwardValidator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Walk-forward backtest validator with transaction costs.

    Loads historical daily signal JSONs from ``RESULTS_DIR``, fetches price
    data via yfinance, and simulates paper trading with the same risk rules
    as ``paper_trader.py`` plus realistic friction costs.
    """

    def __init__(
        self,
        transaction_cost_pct: float = TRANSACTION_COST_PCT,
        spread_pct: float = SPREAD_PCT,
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        initial_cash: float = INITIAL_CASH,
        position_size: float = POSITION_SIZE,
        max_positions: int = MAX_POSITIONS,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        trailing_stop_pct: float = TRAILING_STOP_PCT,
        risk_free_rate: float = RISK_FREE_RATE,
        max_hold_days: int = MAX_HOLD_DAYS,
        min_confidence: float = MIN_CONFIDENCE,
        results_dir: str = RESULTS_DIR,
    ) -> None:
        self.transaction_cost_pct = transaction_cost_pct
        self.spread_pct = spread_pct
        self.window_size = window_size
        self.step_size = step_size
        self.initial_cash = initial_cash
        self.position_size = position_size
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.risk_free_rate = risk_free_rate
        self.max_hold_days = max_hold_days
        self.min_confidence = min_confidence
        self.results_dir = results_dir

        # State populated by run_backtest()
        self.cash: float = initial_cash
        self.positions: dict[str, Position] = {}
        self.peak_equity: float = initial_cash
        self.equity_curve: list[tuple[str, float]] = []  # (date, equity)
        self.all_trades: list[TradeRecord] = []
        self.window_results: list[WindowResult] = []
        self.daily_returns: list[tuple[str, float]] = []
        self.benchmark_curve: list[tuple[str, float]] = []
        self.signal_sources: dict[str, list[TradeRecord]] = {}  # source -> trades
        self._metrics: dict[str, Any] = {}
        self._trading_days: list[str] = []  # ordered list of trading days seen

        # Price cache: ticker -> {date: price}
        self._price_cache: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Signal loading
    # ------------------------------------------------------------------

    def _load_daily_files(self) -> dict[str, dict]:
        """Load all daily result JSONs indexed by date string.

        Returns:
            {date_str: {date, profile, results: [...]}}
        """
        files = sorted(Path(self.results_dir).glob("*.json"))
        loaded: dict[str, dict] = {}
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                date_str = data.get("date")
                if date_str:
                    loaded[date_str] = data
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping corrupt file %s: %s", f, exc)
        return loaded

    def _get_signal_dates(
        self, daily_data: dict[str, dict], start_date: str, end_date: str
    ) -> list[str]:
        """Return sorted list of signal dates within [start_date, end_date]."""
        return sorted(
            d for d in daily_data if start_date <= d <= end_date
        )

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def _fetch_prices(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> dict[str, pd.Series]:
        """Fetch adjusted close prices for all tickers over date range.

        Returns:
            {ticker: pd.Series indexed by date}
        """
        if not tickers:
            return {}
        unique = sorted(set(tickers))
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=5)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=10)

        prices: dict[str, pd.Series] = {}
        for ticker in unique:
            try:
                hist: pd.DataFrame = yf.download(  # type: ignore[assignment]
                    ticker,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if hist is None or hist.empty:
                    logger.warning("No price data for %s", ticker)
                    continue
                close_col: Any = hist["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                close: pd.Series = pd.Series(close_col, dtype=float)  # type: ignore[assignment]
                prices[ticker] = close
                for idx, val in close.items():
                    ds = str(idx)[:10]
                    if ticker not in self._price_cache:
                        self._price_cache[ticker] = {}
                    self._price_cache[ticker][ds] = float(val)
            except Exception as exc:
                logger.warning("Error fetching prices for %s: %s", ticker, exc)

        return prices

    def _get_price(self, ticker: str, date: str) -> Optional[float]:
        """Get cached price for ticker on date, or None."""
        return self._price_cache.get(ticker, {}).get(date)

    def _get_next_trading_day_price(
        self, ticker: str, date: str, price_series: pd.Series
    ) -> Optional[float]:
        """Find the next trading day's close price after *date*."""
        for idx, val in price_series.items():
            ds = idx.strftime("%Y-%m-%d")
            if ds > date:
                return float(val)
        return None

    def _get_price_on_or_after(
        self, ticker: str, date: str, price_series: pd.Series
    ) -> Optional[float]:
        """Get the close price on date or the next available trading day."""
        for idx, val in price_series.items():
            ds = idx.strftime("%Y-%m-%d")
            if ds >= date:
                return float(val)
        return None

    # ------------------------------------------------------------------
    # Trade simulation helpers
    # ------------------------------------------------------------------

    def _equity(self, date: str) -> float:
        """Calculate total portfolio equity as of *date* using cached prices."""
        value = self.cash
        for ticker, pos in self.positions.items():
            price = self._get_price(ticker, date)
            if price is not None:
                value += pos.shares * price
            else:
                value += pos.cost  # fallback to cost basis
        return value

    def _apply_transaction_cost(self, amount: float) -> float:
        """Deduct transaction cost + spread from trade amount."""
        cost = amount * self.transaction_cost_pct + amount * self.spread_pct
        return cost

    def _normalize_decision(self, decision: str) -> Optional[str]:
        """Map decision string to BUY / SELL / None.

        Handles new consensus recommendations (STRONG_BUY, CAUTIOUS_BUY, etc.)
        and legacy raw decisions (BUY, SELL, HOLD, OVERWEIGHT, UNDERWEIGHT).
        """
        d = (decision or "").upper().strip()
        if d in ("BUY", "OVERWEIGHT", "STRONG_BUY"):
            return "BUY"
        if d in ("SELL", "UNDERWEIGHT", "STRONG_SELL"):
            return "SELL"
        # CAUTIOUS_BUY, HOLD, STRONG_HOLD → no trade
        return None

    def _consensus_to_decision(self, recommendation: str) -> str:
        """Map consensus recommendation to a backtest decision.

        CAUTIOUS_BUY → HOLD (too uncertain to trade)
        STRONG_HOLD/HOLD → no trade
        STRONG_BUY/BUY → BUY
        STRONG_SELL/SELL → SELL
        """
        if not recommendation:
            return "HOLD"
        rec = recommendation.upper()
        if "BUY" in rec and "CAUTIOUS" not in rec:
            return "BUY"
        if "SELL" in rec:
            return "SELL"
        return "HOLD"

    # ------------------------------------------------------------------
    # Window simulation
    # ------------------------------------------------------------------

    def _simulate_window(
        self,
        signals: list[dict],
        price_data: dict[str, pd.Series],
        window_end: str,
    ) -> WindowResult:
        """Execute trades for one walk-forward window.

        For each signal date within the window:
        1. Check risk rules on existing positions (stop-loss, take-profit,
           trailing stop) using that date's prices.
        2. Process new BUY/SELL signals.
        3. Record daily equity.

        Args:
            signals: list of signal dicts with date, ticker, decision, etc.
            price_data: {ticker: pd.Series of close prices}
            window_end: end date of this window (for equity recording).

        Returns:
            WindowResult with trades and equity snapshot.
        """
        result = WindowResult(
            window_start=signals[0]["date"] if signals else window_end,
            window_end=window_end,
            start_equity=self._equity(window_end),
        )

        # Group signals by date
        by_date: dict[str, list[dict]] = {}
        for sig in signals:
            by_date.setdefault(sig["date"], []).append(sig)

        sorted_dates = sorted(by_date.keys())
        all_tickers = list({s["ticker"] for s in signals})

        # Fetch any missing tickers
        self._fetch_prices(all_tickers, sorted_dates[0], window_end)

        for date_str in sorted_dates:
            day_signals = by_date[date_str]

            # Track trading days for hold-period calculation
            if date_str not in self._trading_days:
                self._trading_days.append(date_str)

            # Step 1: risk management on existing positions (incl. forced exit)
            self._check_risk_rules(date_str, price_data, result)

            # Step 2: sort signals by confidence (highest first) if available
            day_signals = sorted(
                day_signals,
                key=lambda s: s.get("confidence", 0.0),
                reverse=True,
            )

            # Step 3: process new signals
            for sig in day_signals:
                ticker = sig["ticker"]
                decision = self._normalize_decision(sig.get("decision", ""))
                if decision is None:
                    logger.debug(
                        "SKIP %s on %s: decision='%s' → normalized to None",
                        ticker, date_str, sig.get("decision"),
                    )
                    continue
                if sig.get("status") != "ok":
                    logger.debug(
                        "SKIP %s on %s: status='%s' (not ok)",
                        ticker, date_str, sig.get("status"),
                    )
                    continue

                # Confidence filtering: only trade if confidence >= threshold
                # (confidence is optional — not all daily JSONs include it)
                confidence = sig.get("confidence")
                if confidence is not None and confidence < self.min_confidence:
                    logger.info(
                        "SKIP %s on %s: confidence=%.2f < min=%.2f",
                        ticker, date_str, confidence, self.min_confidence,
                    )
                    continue

                price = self._get_price(ticker, date_str)
                if price is None:
                    price = self._get_price_on_or_after(ticker, date_str, price_data.get(ticker, pd.Series(dtype=float)))
                if price is None:
                    logger.warning("No price for %s on %s — skipping signal", ticker, date_str)
                    continue

                result.signal_count += 1

                if decision == "BUY":
                    self._execute_buy(ticker, price, date_str, result, sig)
                elif decision == "SELL":
                    self._execute_sell(ticker, price, date_str, result, sig)

            # Record equity after processing this day
            eq = self._equity(date_str)
            self.equity_curve.append((date_str, eq))
            if eq > self.peak_equity:
                self.peak_equity = eq

        result.end_equity = self._equity(window_end)
        return result

    def _check_risk_rules(
        self,
        date_str: str,
        price_data: dict[str, pd.Series],
        result: WindowResult,
    ) -> None:
        """Apply stop-loss, take-profit, trailing stop, and forced hold-period exit."""
        # Determine how many trading days have elapsed since entry for each position
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            price = self._get_price(ticker, date_str)
            if price is None:
                price = self._get_price_on_or_after(ticker, date_str, price_data.get(ticker, pd.Series(dtype=float)))
            if price is None:
                continue

            # Update peak
            if price > pos.peak_price:
                pos.peak_price = price

            pnl_pct = (price - pos.avg_price) / pos.avg_price

            reason = None
            if pnl_pct <= self.stop_loss_pct:
                reason = "STOP_LOSS"
            elif pnl_pct >= self.take_profit_pct:
                reason = "TAKE_PROFIT"
            else:
                drawdown_from_peak = (price - pos.peak_price) / pos.peak_price
                if drawdown_from_peak <= -self.trailing_stop_pct:
                    reason = "TRAILING_STOP"

            # Forced exit: if held for more than max_hold_days trading days, close
            if reason is None and self.max_hold_days > 0:
                days_held = 0
                for td in self._trading_days:
                    if td >= pos.entry_date:
                        days_held += 1
                    if td == date_str:
                        break
                if days_held > self.max_hold_days:
                    reason = "MAX_HOLD_DAYS"
                    logger.info(
                        "FORCE EXIT %s on %s: held %d trading days (max=%d), pnl=%.1f%%",
                        ticker, date_str, days_held, self.max_hold_days, pnl_pct * 100,
                    )

            if reason:
                self._close_position(ticker, price, date_str, reason, result)

    def _execute_buy(
        self,
        ticker: str,
        price: float,
        date_str: str,
        result: WindowResult,
        sig: dict | None = None,
    ) -> None:
        """Open a long position respecting position limits and costs."""
        if ticker in self.positions:
            logger.debug(
                "SKIP BUY %s on %s: already held (entry %s)",
                ticker, date_str, self.positions[ticker].entry_date,
            )
            return  # already held
        if len(self.positions) >= self.max_positions:
            logger.info(
                "SKIP BUY %s on %s: max positions (%d/%d) reached",
                ticker, date_str, len(self.positions), self.max_positions,
            )
            return

        total_value = self.cash + sum(
            p.shares * (self._get_price(t, date_str) or p.cost)
            for t, p in self.positions.items()
        )
        target_amount = total_value * self.position_size
        txn_cost = self._apply_transaction_cost(target_amount)
        available = self.cash - txn_cost

        if available < target_amount * 0.1:
            logger.info(
                "SKIP BUY %s on %s: insufficient cash ($%.0f available < $%.0f needed * 10%%)",
                ticker, date_str, available, target_amount,
            )
            return

        effective_price = price * (1.0 + self.spread_pct / 2)
        # Cap purchase at target amount, not all available cash
        invest_amount = min(available, target_amount)
        shares = int(invest_amount / effective_price)
        if shares <= 0:
            logger.info(
                "SKIP BUY %s on %s: zero shares (price=$%.2f, available=$%.0f)",
                ticker, date_str, effective_price, available,
            )
            return

        cost = shares * effective_price
        if cost + txn_cost > self.cash:
            shares = max(1, int((self.cash - txn_cost) / effective_price))
            cost = shares * effective_price
            txn_cost = self._apply_transaction_cost(cost)

        self.cash -= (cost + txn_cost)
        self.positions[ticker] = Position(
            ticker=ticker,
            shares=shares,
            avg_price=effective_price,
            cost=cost,
            entry_date=date_str,
        )

        logger.info(
            "BUY %s on %s: %d shares @ $%.2f (cost=$%.0f, txn=$%.2f, cash=$%.0f)",
            ticker, date_str, shares, effective_price, cost, txn_cost, self.cash,
        )

        trade = TradeRecord(
            action="BUY",
            ticker=ticker,
            shares=shares,
            price=effective_price,
            amount=cost,
            date=date_str,
            txn_cost=txn_cost,
        )
        self.all_trades.append(trade)
        result.trades.append(trade)
        self._track_signal_source(sig, trade)

    def _execute_sell(
        self,
        ticker: str,
        price: float,
        date_str: str,
        result: WindowResult,
        sig: dict | None = None,
    ) -> None:
        """Close position on SELL signal."""
        if ticker not in self.positions:
            logger.debug("SKIP SELL %s on %s: no position held", ticker, date_str)
            return
        self._close_position(ticker, price, date_str, "SIGNAL", result, sig)

    def _close_position(
        self,
        ticker: str,
        price: float,
        date_str: str,
        reason: str,
        result: WindowResult,
        sig: dict | None = None,
    ) -> None:
        """Close an open position and record the trade."""
        pos = self.positions[ticker]

        # Apply spread on sell side
        effective_price = price * (1.0 - self.spread_pct / 2)
        revenue = pos.shares * effective_price
        txn_cost = self._apply_transaction_cost(revenue)
        net_revenue = revenue - txn_cost

        pnl = net_revenue - pos.cost
        pnl_pct = (pnl / pos.cost * 100) if pos.cost > 0 else 0.0

        logger.info(
            "SELL %s on %s: %d shares @ $%.2f, pnl=$%.0f (%.1f%%) [%s], cash=$%.0f",
            ticker, date_str, pos.shares, effective_price, pnl, pnl_pct, reason, self.cash + net_revenue,
        )

        self.cash += net_revenue
        del self.positions[ticker]

        trade = TradeRecord(
            action="SELL",
            ticker=ticker,
            shares=pos.shares,
            price=effective_price,
            amount=net_revenue,
            date=date_str,
            reason=reason,
            pnl=pnl,
            pnl_pct=round(pnl_pct, 2),
            txn_cost=txn_cost,
        )
        self.all_trades.append(trade)
        result.trades.append(trade)
        self._track_signal_source(sig, trade)

    def _track_signal_source(self, sig: dict | None, trade: TradeRecord) -> None:
        """If consensus source data is available, categorize the trade."""
        if sig is None:
            return
        consensus = sig.get("consensus")
        if not consensus:
            return
        source = consensus.get("recommendation", "unknown")
        self.signal_sources.setdefault(source, []).append(trade)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def _fetch_benchmark(
        self, benchmark: str, start_date: str, end_date: str
    ) -> None:
        """Fetch buy-and-hold benchmark equity curve."""
        try:
            hist: pd.DataFrame = yf.download(  # type: ignore[assignment]
                benchmark,
                start=start_date,
                end=(datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if hist is None or hist.empty:
                logger.warning("No benchmark data for %s", benchmark)
                return
            close_col: Any = hist["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            close: pd.Series = pd.Series(close_col, dtype=float)  # type: ignore[assignment]
            base_price = float(close.iloc[0])
            benchmark_shares = self.initial_cash / base_price
            for idx, val in close.items():
                self.benchmark_curve.append(
                    (str(idx)[:10], benchmark_shares * float(val))
                )
        except Exception as exc:
            logger.warning("Error fetching benchmark %s: %s", benchmark, exc)

    # ------------------------------------------------------------------
    # Main backtest loop
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        benchmark: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run walk-forward validation from *start_date* to *end_date*.

        Args:
            start_date: first signal date (YYYY-MM-DD).
            end_date: last signal date (YYYY-MM-DD).
            benchmark: ticker for buy-and-hold benchmark (e.g. "SPY").

        Returns:
            Dictionary of metrics (also stored in ``self._metrics``).
        """
        # Reset state
        self.cash = self.initial_cash
        self.positions.clear()
        self.peak_equity = self.initial_cash
        self.equity_curve.clear()
        self.all_trades.clear()
        self.window_results.clear()
        self.daily_returns.clear()
        self.benchmark_curve.clear()
        self.signal_sources.clear()
        self._trading_days.clear()
        self._price_cache.clear()
        self._metrics.clear()

        daily_data = self._load_daily_files()
        signal_dates = self._get_signal_dates(daily_data, start_date, end_date)

        if not signal_dates:
            logger.error("No signal data found between %s and %s", start_date, end_date)
            return {}

        logger.info(
            "Walk-forward backtest: %d signal dates from %s to %s (window=%d, step=%d)",
            len(signal_dates), signal_dates[0], signal_dates[-1],
            self.window_size, self.step_size,
        )

        # Fetch all prices up front for the full date range + margin
        all_tickers = set()
        for d in signal_dates:
            for r in daily_data[d].get("results", []):
                if r.get("status") == "ok":
                    all_tickers.add(r["ticker"])

        logger.info("Fetching prices for %d tickers...", len(all_tickers))
        price_data = self._fetch_prices(
            list(all_tickers),
            start_date,
            (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d"),
        )

        # Fetch benchmark
        if benchmark:
            self._fetch_benchmark(benchmark, start_date, end_date)

        # Walk-forward windows
        window_idx = 0
        while window_idx < len(signal_dates):
            window_dates = signal_dates[window_idx : window_idx + self.window_size]
            if not window_dates:
                break

            window_signals: list[dict] = []
            for d in window_dates:
                for r in daily_data[d].get("results", []):
                    r["date"] = r.get("date", d)
                    # Use consensus recommendation if available, else fall back to decision
                    consensus = r.get("consensus")
                    if consensus and isinstance(consensus, dict):
                        rec = consensus.get("recommendation", "")
                        r["decision"] = self._consensus_to_decision(rec)
                        r["confidence"] = consensus.get("confidence", r.get("confidence"))
                    window_signals.append(r)

            window_end = window_dates[-1]
            logger.info(
                "  Window %d: %s -> %s (%d signals)",
                len(self.window_results) + 1,
                window_dates[0], window_end, len(window_signals),
            )

            wr = self._simulate_window(window_signals, price_data, window_end)
            self.window_results.append(wr)

            window_idx += self.step_size

        # Compute daily returns from equity curve
        prev_eq = self.initial_cash
        for date_str, eq in self.equity_curve:
            if prev_eq > 0:
                ret = (eq - prev_eq) / prev_eq
                self.daily_returns.append((date_str, ret))
            prev_eq = eq

        # Ensure final equity is recorded and deduplicate by date
        seen_dates: dict[str, float] = {}
        deduped: list[tuple[str, float]] = []
        for date_str, eq in self.equity_curve:
            if date_str not in seen_dates:
                seen_dates[date_str] = eq
                deduped.append((date_str, eq))
            else:
                # Keep the latest equity value for this date
                deduped = [(d, eq) if d == date_str else (d, e) for d, e in deduped]
        self.equity_curve = deduped

        # Recompute daily returns from deduped curve
        self.daily_returns.clear()
        prev_eq = self.initial_cash
        for date_str, eq in self.equity_curve:
            if prev_eq > 0:
                ret = (eq - prev_eq) / prev_eq
                self.daily_returns.append((date_str, ret))
            prev_eq = eq

        # Compute metrics
        self._metrics = self._calculate_metrics()
        logger.info("Backtest complete. Final equity: $%.2f", self._metrics.get("final_equity", 0))

        return self._metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_metrics(self) -> dict[str, Any]:
        """Compute performance metrics from the equity curve."""
        metrics: dict[str, Any] = {}

        if not self.equity_curve:
            return metrics

        equity_values = [e for _, e in self.equity_curve]
        final_equity = equity_values[-1]
        total_return = (final_equity - self.initial_cash) / self.initial_cash

        metrics["initial_cash"] = self.initial_cash
        metrics["final_equity"] = round(final_equity, 2)
        metrics["total_return"] = round(total_return, 4)
        metrics["total_return_pct"] = round(total_return * 100, 2)
        metrics["total_trades"] = len(self.all_trades)
        metrics["windows_evaluated"] = len(self.window_results)
        metrics["total_signals"] = sum(w.signal_count for w in self.window_results)

        # Daily returns
        daily_rets = np.array([r for _, r in self.daily_returns]) if self.daily_returns else np.array([])
        if len(daily_rets) > 0:
            # Sharpe ratio (annualized)
            mean_ret = np.mean(daily_rets)
            std_ret = np.std(daily_rets, ddof=1) if len(daily_rets) > 1 else 0.0
            daily_rf = self.risk_free_rate / 252
            sharpe = (mean_ret - daily_rf) / std_ret if std_ret > 0 else 0.0
            metrics["sharpe_ratio"] = round(sharpe * math.sqrt(252), 2)

            # Sortino ratio (annualized)
            downside = daily_rets[daily_rets < 0]
            downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
            sortino = (mean_ret - daily_rf) / downside_std if downside_std > 0 else 0.0
            metrics["sortino_ratio"] = round(sortino * math.sqrt(252), 2)

            # Max drawdown
            peak = np.maximum.accumulate(equity_values)
            drawdowns = (equity_values - peak) / peak
            max_dd = float(np.min(drawdowns))
            metrics["max_drawdown"] = round(max_dd, 4)
            metrics["max_drawdown_pct"] = round(max_dd * 100, 2)

            # Calmar ratio (annualized return / |max drawdown|)
            annualized_return = (1 + total_return) ** (252 / max(len(daily_rets), 1)) - 1
            calmar = annualized_return / abs(max_dd) if abs(max_dd) > 0 else 0.0
            metrics["calmar_ratio"] = round(calmar, 2)

            # Volatility (annualized)
            metrics["annualized_volatility"] = round(std_ret * math.sqrt(252), 4)
            metrics["annualized_volatility_pct"] = round(std_ret * math.sqrt(252) * 100, 2)
        else:
            metrics["sharpe_ratio"] = 0.0
            metrics["sortino_ratio"] = 0.0
            metrics["max_drawdown"] = 0.0
            metrics["max_drawdown_pct"] = 0.0
            metrics["calmar_ratio"] = 0.0
            metrics["annualized_volatility"] = 0.0
            metrics["annualized_volatility_pct"] = 0.0

        # Trade-level stats
        sells = [t for t in self.all_trades if t.action == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        losses = [t for t in sells if t.pnl <= 0]
        metrics["winning_trades"] = len(wins)
        metrics["losing_trades"] = len(losses)
        metrics["win_rate"] = round(len(wins) / len(sells) * 100, 1) if sells else 0.0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        metrics["gross_profit"] = round(gross_profit, 2)
        metrics["gross_loss"] = round(gross_loss, 2)
        metrics["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
        metrics["avg_win_pct"] = round(avg_win, 2)
        metrics["avg_loss_pct"] = round(avg_loss, 2)
        metrics["payoff_ratio"] = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else float("inf")

        # Total transaction costs
        total_txn = sum(t.txn_cost for t in self.all_trades)
        metrics["total_transaction_costs"] = round(total_txn, 2)
        metrics["transaction_cost_pct_of_initial"] = round(total_txn / self.initial_cash * 100, 4)

        # Benchmark comparison
        if self.benchmark_curve:
            bench_values = [e for _, e in self.benchmark_curve]
            bench_return = (bench_values[-1] - self.initial_cash) / self.initial_cash
            metrics["benchmark_return"] = round(bench_return, 4)
            metrics["benchmark_return_pct"] = round(bench_return * 100, 2)
            metrics["alpha_vs_benchmark"] = round(
                (total_return - bench_return) * 100, 2
            )
        else:
            metrics["benchmark_return"] = None
            metrics["benchmark_return_pct"] = None
            metrics["alpha_vs_benchmark"] = None

        return metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Generate a markdown backtest report.

        Returns:
            Markdown string with full report.
        """
        if not self._metrics:
            return "# No backtest results\n\nRun `run_backtest()` first.\n"

        m = self._metrics
        lines: list[str] = []

        lines.append("# Walk-Forward Validation Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Parameters
        lines.append("## Parameters")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|---|---|")
        lines.append(f"| Transaction Cost | {self.transaction_cost_pct*100:.2f}% |")
        lines.append(f"| Spread | {self.spread_pct*100:.2f}% |")
        lines.append(f"| Position Size | {self.position_size*100:.0f}% |")
        lines.append(f"| Max Positions | {self.max_positions} |")
        lines.append(f"| Stop-Loss | {self.stop_loss_pct*100:.1f}% |")
        lines.append(f"| Take-Profit | {self.take_profit_pct*100:.1f}% |")
        lines.append(f"| Trailing Stop | {self.trailing_stop_pct*100:.1f}% |")
        lines.append(f"| Max Hold Days | {self.max_hold_days} |")
        lines.append(f"| Min Confidence | {self.min_confidence} |")
        lines.append(f"| Window Size | {self.window_size} days |")
        lines.append(f"| Step Size | {self.step_size} days |")
        lines.append("")

        # Overall metrics
        lines.append("## Overall Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Initial Cash | ${m['initial_cash']:,.2f} |")
        lines.append(f"| Final Equity | ${m['final_equity']:,.2f} |")
        lines.append(f"| Total Return | {m['total_return_pct']:+.2f}% |")
        lines.append(f"| Sharpe Ratio | {m['sharpe_ratio']:.2f} |")
        lines.append(f"| Sortino Ratio | {m['sortino_ratio']:.2f} |")
        lines.append(f"| Max Drawdown | {m['max_drawdown_pct']:.2f}% |")
        lines.append(f"| Calmar Ratio | {m['calmar_ratio']:.2f} |")
        lines.append(f"| Annualized Volatility | {m.get('annualized_volatility_pct', 0):.2f}% |")
        lines.append(f"| Win Rate | {m['win_rate']:.1f}% |")
        lines.append(f"| Profit Factor | {m['profit_factor']} |")
        lines.append(f"| Total Trades | {m['total_trades']} |")
        lines.append(f"| Signals Processed | {m['total_signals']} |")
        lines.append(f"| Windows Evaluated | {m['windows_evaluated']} |")
        lines.append(f"| Total Transaction Costs | ${m['total_transaction_costs']:,.2f} |")
        lines.append("")

        # Trade breakdown
        lines.append("## Trade Breakdown")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Winning Trades | {m['winning_trades']} |")
        lines.append(f"| Losing Trades | {m['losing_trades']} |")
        lines.append(f"| Avg Win | {m['avg_win_pct']:+.2f}% |")
        lines.append(f"| Avg Loss | {m['avg_loss_pct']:+.2f}% |")
        lines.append(f"| Payoff Ratio | {m['payoff_ratio']} |")
        lines.append(f"| Gross Profit | ${m['gross_profit']:,.2f} |")
        lines.append(f"| Gross Loss | ${m['gross_loss']:,.2f} |")
        lines.append("")

        # Benchmark comparison
        if m.get("benchmark_return") is not None:
            lines.append("## Benchmark Comparison")
            lines.append("")
            lines.append("| Metric | Strategy | Benchmark | Delta |")
            lines.append("|---|---|---|---|")
            lines.append(
                f"| Total Return | {m['total_return_pct']:+.2f}% "
                f"| {m['benchmark_return_pct']:+.2f}% "
                f"| {m['alpha_vs_benchmark']:+.2f}% |"
            )
            lines.append("")

        # Per-signal-source breakdown
        if self.signal_sources:
            lines.append("## Per-Source Breakdown")
            lines.append("")
            lines.append("| Source | Trades | Wins | Win Rate | Avg P&L% |")
            lines.append("|---|---|---|---|---|")
            for source, trades in sorted(self.signal_sources.items()):
                sells_src = [t for t in trades if t.action == "SELL"]
                wins_src = [t for t in sells_src if t.pnl > 0]
                n = len(sells_src)
                wr = len(wins_src) / n * 100 if n > 0 else 0
                avg_pnl = np.mean([t.pnl_pct for t in sells_src]) if sells_src else 0
                lines.append(f"| {source} | {n} | {len(wins_src)} | {wr:.1f}% | {avg_pnl:+.2f}% |")
            lines.append("")

        # Exit reason breakdown
        sells_all = [t for t in self.all_trades if t.action == "SELL"]
        if sells_all:
            lines.append("## Exit Reason Breakdown")
            lines.append("")
            lines.append("| Reason | Count | Avg P&L% |")
            lines.append("|---|---|---|")
            from collections import Counter
            reasons: dict[str, list[TradeRecord]] = {}
            for t in sells_all:
                reasons.setdefault(t.reason, []).append(t)
            for reason, trades_r in sorted(reasons.items()):
                avg = np.mean([t.pnl_pct for t in trades_r])
                lines.append(f"| {reason} | {len(trades_r)} | {avg:+.2f}% |")
            lines.append("")

        # Monthly returns
        lines.append("## Monthly Returns")
        lines.append("")
        lines.append("| Month | Return |")
        lines.append("|---|---|")
        monthly: dict[str, list[float]] = {}
        for date_str, eq in self.equity_curve:
            month_key = date_str[:7]  # YYYY-MM
            monthly.setdefault(month_key, []).append(eq)
        months_sorted = sorted(monthly.keys())
        for i, month in enumerate(months_sorted):
            vals = monthly[month]
            if i == 0:
                prev = self.initial_cash
            else:
                prev_vals = monthly[months_sorted[i - 1]]
                prev = prev_vals[-1]
            ret = (vals[-1] - prev) / prev * 100 if prev > 0 else 0
            lines.append(f"| {month} | {ret:+.2f}% |")
        lines.append("")

        # Equity curve data
        lines.append("## Equity Curve")
        lines.append("")
        lines.append("```")
        lines.append(f"{'Date':<12} {'Equity':>12} {'Return':>10}")
        lines.append("-" * 36)
        prev_eq = self.initial_cash
        for date_str, eq in self.equity_curve:
            ret = (eq - prev_eq) / prev_eq * 100 if prev_eq > 0 else 0
            lines.append(f"{date_str:<12} ${eq:>11,.2f} {ret:>+9.2f}%")
            prev_eq = eq
        lines.append("```")
        lines.append("")

        # Trade log
        if self.all_trades:
            lines.append("## Trade Log")
            lines.append("")
            lines.append("| Date | Action | Ticker | Shares | Price | P&L | Reason |")
            lines.append("|---|---|---|---|---|---|---|")
            for t in self.all_trades:
                pnl_str = f"${t.pnl:,.0f} ({t.pnl_pct:+.1f}%)" if t.action == "SELL" else "—"
                lines.append(
                    f"| {t.date} | {t.action} | {t.ticker} "
                    f"| {t.shares} | ${t.price:.2f} | {pnl_str} | {t.reason} |"
                )
            lines.append("")

        return "\n".join(lines)

    def save_report(self, path: Optional[str] = None) -> str:
        """Save the report to a markdown file.

        Args:
            path: file path. Defaults to OUTPUT_DIR/backtest_YYYY-MM-DD.md.

        Returns:
            Path to the saved file.
        """
        if path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(
                OUTPUT_DIR,
                f"backtest_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.md",
            )
        report = self.generate_report()
        with open(path, "w") as f:
            f.write(report)
        logger.info("Report saved to %s", path)
        return path

    def get_metrics(self) -> dict[str, Any]:
        """Return computed metrics dict for programmatic use."""
        return dict(self._metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _latest_n_days(n: int) -> tuple[str, str]:
    """Compute start and end dates for the last *n* calendar days."""
    end = datetime.now()
    start = end - timedelta(days=n)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward validation backtest for TradingAgents signals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ec2_deploy/backtest_engine.py --start 2026-05-15 --end 2026-05-28\n"
            "  python ec2_deploy/backtest_engine.py --start 2026-05-15 --end 2026-05-28 --benchmark SPY\n"
            "  python ec2_deploy/backtest_engine.py --latest 14\n"
        ),
    )
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--latest", type=int, metavar="N", help="Backtest last N days")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark ticker (default: SPY)")
    parser.add_argument("--output", type=str, default=None, help="Output report path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.latest:
        start_date, end_date = _latest_n_days(args.latest)
        logger.info("Backtesting last %d days: %s to %s", args.latest, start_date, end_date)
    elif args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        parser.error("Provide --start/--end or --latest N")

    validator = WalkForwardValidator()
    metrics = validator.run_backtest(
        start_date=start_date,
        end_date=end_date,
        benchmark=args.benchmark,
    )

    if not metrics:
        logger.error("No metrics produced — check signal data and dates.")
        sys.exit(1)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Final Equity:       ${metrics['final_equity']:,.2f}")
    print(f"  Total Return:       {metrics['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.2f}")
    print(f"  Win Rate:           {metrics['win_rate']:.1f}%")
    print(f"  Profit Factor:      {metrics['profit_factor']}")
    print(f"  Total Trades:       {metrics['total_trades']}")
    print(f"  Transaction Costs:  ${metrics['total_transaction_costs']:,.2f}")
    if metrics.get("benchmark_return") is not None:
        print(f"  Benchmark Return:   {metrics['benchmark_return_pct']:+.2f}%")
        print(f"  Alpha vs Benchmark: {metrics['alpha_vs_benchmark']:+.2f}%")
    print("=" * 60)

    # Save report
    report_path = validator.save_report(args.output)
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
