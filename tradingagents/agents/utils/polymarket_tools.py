"""Polymarket prediction market tools for LangChain agents."""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.polymarket import (
    get_polymarket_odds as _get_odds,
    get_polymarket_sentiment as _get_sentiment,
)


@tool
def get_polymarket_odds(
    ticker: Annotated[str, "Asset ticker symbol (e.g., BTC, ETH, NVDA)"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days of context"] = 7,
) -> str:
    """
    Retrieve prediction market odds from Polymarket for an asset.
    Returns crowd-sourced probability estimates for price targets and events.
    Outcome prices represent implied probabilities (e.g., $0.65 = 65% probability).

    Args:
        ticker (str): Asset ticker symbol
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Days of context (default 7)
    Returns:
        str: Formatted table of prediction market data with probabilities and volumes
    """
    return _get_odds(ticker, curr_date, look_back_days)


@tool
def get_polymarket_sentiment(
    ticker: Annotated[str, "Asset ticker symbol (e.g., BTC, ETH, NVDA)"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Get aggregated prediction market sentiment for an asset from Polymarket.
    Analyzes active prediction markets to determine bullish/bearish crowd consensus.
    Returns sentiment score, signal breakdown, and high-volume market highlights.

    Args:
        ticker (str): Asset ticker symbol
        curr_date (str): Current date in yyyy-mm-dd format
    Returns:
        str: Formatted sentiment summary with bullish/bearish score
    """
    return _get_sentiment(ticker, curr_date)
