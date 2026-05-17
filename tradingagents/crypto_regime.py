"""Crypto Regime Reader — reads shared regime signal from hyperliquid-dex.

Usage:
    from tradingagents.crypto_regime import get_regime, get_bias

    regime = get_regime()
    print(regime["regime"])           # 'strong_uptrend', 'strong_downtrend', 'ranging'
    print(regime["suggested_bias"])   # 'bullish', 'bearish', 'neutral'

    # Quick shortcut
    bias = get_bias()  # returns one of 'bullish', 'bearish', 'neutral'
"""

import json
import time
import logging
import fcntl
from pathlib import Path

log = logging.getLogger(__name__)

REGIME_PATH = Path("/home/tk578/trading-shared/regime.json")
MAX_AGE_SECONDS = 7200  # 2 hours — if regime file is older, treat as stale


def get_regime() -> dict:
    """Read the current crypto regime from the shared JSON file.

    Returns a dict with keys:
        regime           — 'strong_uptrend', 'strong_downtrend', 'ranging', or 'unknown'
        btc_price        — float or None
        btc_ema200       — float or None
        adx              — float or None
        timestamp        — ISO 8601 UTC string or None
        suggested_bias   — 'bullish', 'bearish', or 'neutral'
        stale            — bool, True if the signal is older than MAX_AGE_SECONDS
    """
    default = {
        "regime": "unknown",
        "btc_price": None,
        "btc_ema200": None,
        "adx": None,
        "timestamp": None,
        "suggested_bias": "neutral",
        "stale": True,
    }

    if not REGIME_PATH.exists():
        log.warning("crypto_regime: %s does not exist", REGIME_PATH)
        return default

    try:
        with open(REGIME_PATH, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            raw = f.read()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("crypto_regime: failed to read %s: %s", REGIME_PATH, e)
        return default

    # Check staleness
    ts = data.get("timestamp_epoch")
    if ts is not None:
        data["stale"] = (time.time() - ts) > MAX_AGE_SECONDS
    else:
        data["stale"] = True

    # Fill defaults for missing keys
    for key in default:
        data.setdefault(key, default[key])

    return data


def get_bias() -> str:
    """Quick shortcut: returns the suggested_bias string ('bullish', 'bearish', 'neutral')."""
    return get_regime().get("suggested_bias", "neutral")


def get_regime_name() -> str:
    """Quick shortcut: returns the regime name string."""
    return get_regime().get("regime", "unknown")


def get_regime_context() -> str:
    """Get formatted regime context string for injection into analyst prompts.

    Returns empty string if regime is unknown or not available.
    Otherwise returns a formatted block with market regime info.
    """
    r = get_regime()

    if r["regime"] == "unknown":
        return ""

    btc_vs_ema = "above" if (r["btc_price"] is not None and r.get("btc_ema200") is not None and r["btc_price"] > r["btc_ema200"]) else "below"
    btc_str = f"${r['btc_price']:,.0f}" if r['btc_price'] is not None else "N/A"
    adx_str = f"{r['adx']:.1f}" if r['adx'] is not None else "N/A"

    return (
        f"\n\nCRYPTO MARKET REGIME (live from scanner):\n"
        f"- Regime: {r['regime']}\n"
        f"- Suggested Bias: {r['suggested_bias']}\n"
        f"- BTC: {btc_str} ({btc_vs_ema} EMA200)\n"
        f"- ADX: {adx_str}\n"
        f"- Data freshness: {'FRESH' if not r['stale'] else 'STALE (>2h)'}\n"
        f"Use this context to align your analysis with the current market regime.\n"
    )
