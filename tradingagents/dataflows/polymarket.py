"""Polymarket prediction market data vendor.

Fetches crowd-sourced probability estimates from Polymarket's Gamma API.
No authentication required for read-only market data.

Data strategy:
- /events?tag_slug=crypto for crypto assets (returns events with nested markets)
- /markets for individual market queries with client-side filtering
"""

import json
import requests


GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Maps asset tickers to search terms and related keywords
TICKER_MAP = {
    "BTC": {"name": "Bitcoin", "keywords": ["bitcoin", "btc"]},
    "ETH": {"name": "Ethereum", "keywords": ["ethereum", "eth"]},
    "SOL": {"name": "Solana", "keywords": ["solana", "sol"]},
    "XRP": {"name": "XRP", "keywords": ["xrp", "ripple"]},
    "DOGE": {"name": "Dogecoin", "keywords": ["dogecoin", "doge"]},
}


def _fetch_crypto_events(limit: int = 50):
    """Fetch crypto events with nested markets from Gamma API."""
    try:
        resp = requests.get(
            f"{GAMMA_API_BASE}/events",
            params={"closed": "false", "limit": limit, "tag_slug": "crypto"},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return f"Error fetching Polymarket events: {e}"


def _fetch_markets_raw(limit: int = 200):
    """Fetch raw markets sorted by volume."""
    try:
        resp = requests.get(
            f"{GAMMA_API_BASE}/markets",
            params={"closed": "false", "limit": limit},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return f"Error fetching Polymarket markets: {e}"


def _filter_by_keywords(items, keywords, field="question"):
    """Filter a list of dicts by keyword match in a given field.

    Uses word-boundary matching to avoid false positives
    (e.g., 'eth' should not match 'Netherlands').
    """
    import re
    patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in keywords]
    return [
        item for item in items
        if isinstance(item, dict)
        and any(p.search(item.get(field, "")) for p in patterns)
    ]


def _format_price(prices, outcomes):
    """Format outcome prices with labels."""
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except (json.JSONDecodeError, TypeError):
            return str(prices)

    if isinstance(prices, list) and isinstance(outcomes, list):
        parts = []
        for o, p in zip(outcomes, prices):
            try:
                parts.append(f"{o}: {float(p):.0%}")
            except (ValueError, TypeError):
                parts.append(f"{o}: N/A")
        return " / ".join(parts)
    return str(prices)


def _format_volume(vol):
    """Format volume as dollar string."""
    try:
        v = float(vol) if vol else 0
        return f"${v:,.0f}" if v else "N/A"
    except (ValueError, TypeError):
        return "N/A"


def get_polymarket_odds(ticker: str, curr_date: str, look_back_days: int = 7) -> str:
    """Get Polymarket prediction market odds related to an asset.

    Searches both crypto events and general markets for relevant predictions.

    Args:
        ticker: Asset ticker (e.g., "BTC", "ETH", "NVDA")
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Days of context (for prompt context)

    Returns:
        Formatted string with prediction market data
    """
    ticker_info = TICKER_MAP.get(ticker.upper())
    search_name = ticker_info["name"] if ticker_info else ticker
    keywords = ticker_info["keywords"] if ticker_info else [ticker.lower()]

    all_markets = []

    # Strategy 1: Crypto events endpoint (richest source for crypto)
    if ticker_info:
        events = _fetch_crypto_events(limit=50)
        if isinstance(events, list):
            relevant_events = _filter_by_keywords(events, keywords, field="title")
            for event in relevant_events:
                for m in event.get("markets", []):
                    m["_event_title"] = event.get("title", "")
                    all_markets.append(m)

    # Strategy 2: Raw markets endpoint (catches anything missed)
    raw_markets = _fetch_markets_raw(limit=200)
    if isinstance(raw_markets, list):
        relevant_raw = _filter_by_keywords(raw_markets, keywords)
        for m in relevant_raw:
            # Dedupe by conditionId
            existing_ids = {x.get("conditionId") for x in all_markets}
            if m.get("conditionId") not in existing_ids:
                all_markets.append(m)

    if not all_markets:
        return (
            f"No Polymarket prediction markets found for '{search_name}' as of {curr_date}.\n"
            f"Note: Polymarket may not have active markets for this asset. "
            f"The platform focuses on event outcomes, not continuous price feeds."
        )

    # Sort by volume (most active first)
    def sort_vol(m):
        try:
            return float(m.get("volume24hr", 0) or 0)
        except (ValueError, TypeError):
            return 0

    all_markets.sort(key=sort_vol, reverse=True)

    lines = [
        f"## Polymarket Prediction Market Data for {search_name}",
        f"**Date**: {curr_date}",
        f"**Markets found**: {len(all_markets)}",
        "",
        "| Market Question | Implied Probability | 24h Volume | Status |",
        "|----------------|-------------------|-----------|--------|",
    ]

    for m in all_markets[:15]:
        question = m.get("question", "N/A")
        if len(question) > 65:
            question = question[:62] + "..."

        prices = m.get("outcomePrices", [])
        outcomes = m.get("outcomes", [])
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except (json.JSONDecodeError, TypeError):
                outcomes = ["Yes", "No"]
        if not outcomes:
            outcomes = ["Yes", "No"]
        price_str = _format_price(prices, outcomes)
        vol_24h = _format_volume(m.get("volume24hr", m.get("volume", 0)))
        active = m.get("active", True)
        closed = m.get("closed", False)
        status = "Closed" if closed else ("Active" if active else "Unknown")

        lines.append(f"| {question} | {price_str} | {vol_24h} | {status} |")

    lines.extend([
        "",
        "**Reading the table**: Outcome prices = implied probabilities.",
        "Yes: 65% means the crowd estimates a 65% chance of that outcome.",
        "High volume = more traders with real money backing the estimate.",
        f"**Source**: Polymarket Gamma API ({GAMMA_API_BASE})",
    ])

    return "\n".join(lines)


def get_polymarket_sentiment(ticker: str, curr_date: str) -> str:
    """Get aggregated prediction market sentiment for an asset.

    Analyzes active prediction markets to determine bullish/bearish consensus.

    Args:
        ticker: Asset ticker
        curr_date: Current date in yyyy-mm-dd format

    Returns:
        Formatted sentiment summary
    """
    ticker_info = TICKER_MAP.get(ticker.upper())
    search_name = ticker_info["name"] if ticker_info else ticker
    keywords = ticker_info["keywords"] if ticker_info else [ticker.lower()]

    all_markets = []

    # Fetch from both endpoints
    if ticker_info:
        events = _fetch_crypto_events(limit=50)
        if isinstance(events, list):
            for event in _filter_by_keywords(events, keywords, field="title"):
                all_markets.extend(event.get("markets", []))

    raw = _fetch_markets_raw(limit=200)
    if isinstance(raw, list):
        existing_ids = {m.get("conditionId") for m in all_markets}
        for m in _filter_by_keywords(raw, keywords):
            if m.get("conditionId") not in existing_ids:
                all_markets.append(m)

    if not all_markets:
        return f"No prediction market sentiment available for {search_name}."

    bullish_signals = 0
    bearish_signals = 0
    total_volume = 0
    notable_markets = []

    bullish_kw = ["above", "hit", "reach", "all time high", "ath", "up", "exceed"]
    bearish_kw = ["below", "drop", "crash", "down", "fall", "sell"]

    for m in all_markets:
        question = m.get("question", "").lower()
        prices = m.get("outcomePrices", [])

        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except (json.JSONDecodeError, TypeError):
                continue

        if not isinstance(prices, list) or len(prices) < 1:
            continue

        try:
            yes_price = float(prices[0])
        except (ValueError, TypeError):
            continue

        try:
            vol = float(m.get("volume24hr", 0) or 0)
        except (ValueError, TypeError):
            vol = 0

        total_volume += vol

        is_bullish_q = any(kw in question for kw in bullish_kw)
        is_bearish_q = any(kw in question for kw in bearish_kw)

        if is_bullish_q and yes_price > 0.5:
            bullish_signals += 1
        elif is_bullish_q and yes_price < 0.5:
            bearish_signals += 1
        elif is_bearish_q and yes_price > 0.5:
            bearish_signals += 1
        elif is_bearish_q and yes_price < 0.5:
            bullish_signals += 1

        if vol > 1000:
            notable_markets.append({
                "question": m.get("question", "N/A"),
                "yes_price": yes_price,
                "volume": vol,
            })

    total = bullish_signals + bearish_signals
    if total == 0:
        sentiment = "Neutral (insufficient directional data)"
        score = 0.5
    else:
        score = bullish_signals / total
        if score > 0.65:
            sentiment = "Bullish"
        elif score < 0.35:
            sentiment = "Bearish"
        else:
            sentiment = "Mixed"

    lines = [
        f"## Polymarket Sentiment: {search_name}",
        f"**Date**: {curr_date}",
        f"**Overall Sentiment**: {sentiment} (score: {score:.0%} bullish)",
        f"**Markets Analyzed**: {len(all_markets)}",
        f"**Bullish Signals**: {bullish_signals} | **Bearish Signals**: {bearish_signals}",
        f"**Total Volume**: ${total_volume:,.0f}",
        "",
    ]

    if notable_markets:
        lines.append("### Notable Markets (>$1k volume)")
        for nm in sorted(notable_markets, key=lambda x: x["volume"], reverse=True)[:8]:
            q = nm["question"]
            if len(q) > 60:
                q = q[:57] + "..."
            lines.append(f"- {q}: Yes={nm['yes_price']:.0%}, Vol=${nm['volume']:,.0f}")

    lines.extend([
        "",
        "**Note**: Sentiment is derived from prediction market pricing.",
        "Use as a supplementary signal alongside technical and fundamental analysis.",
        "Polymarket captures crowd conviction backed by real money.",
    ])

    return "\n".join(lines)
