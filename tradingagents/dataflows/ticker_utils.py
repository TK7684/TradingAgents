"""Ticker symbol normalization for yfinance compatibility."""

# Common crypto symbols that yfinance expects as ``<SYM>-USD`` pairs.
# Extend this set as new assets are added to the supported universe.
KNOWN_CRYPTO_SYMBOLS = frozenset(
    {
        "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX",
        "MATIC", "LINK", "LTC", "BCH", "XLM", "ATOM", "UNI", "ETC",
        "FIL", "APT", "ARB", "OP", "NEAR", "ICP", "INJ", "SUI",
        "TON", "TRX", "HBAR", "VET", "ALGO", "AAVE", "MKR", "SHIB",
        "PEPE", "WIF", "BONK",
    }
)


def normalize_yf_ticker(symbol: str) -> str:
    """Return a yfinance-compatible ticker.

    yfinance represents crypto as ``BTC-USD`` / ``ETH-USD``. Bare symbols
    like ``BTC`` silently return empty data, which breaks downstream agents.
    This helper maps bare known-crypto symbols to their USD pair while
    preserving everything else (equities, ETFs, exchange-suffixed tickers
    such as ``CNC.TO`` or ``7203.T``) untouched.
    """
    if not symbol:
        return symbol

    cleaned = symbol.strip().upper()
    if not cleaned:
        return cleaned

    # Already a yfinance pair (``BTC-USD``) or exchange-suffixed (``CNC.TO``).
    if "-" in cleaned or "." in cleaned:
        return cleaned

    if cleaned in KNOWN_CRYPTO_SYMBOLS:
        return f"{cleaned}-USD"

    return cleaned
