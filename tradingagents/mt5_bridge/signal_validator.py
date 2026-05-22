"""Signal validator — parses and validates TradingView webhook signals.

Supports TradingView Pine Script JSON format and custom alert messages.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .webhook_receiver import _normalize_signal


@dataclass(frozen=True)
class TradingSignal:
    """Validated, normalized trading signal."""

    symbol: str
    side: str  # BUY, SELL, CLOSE
    price: float | None
    sl: float | None
    tp: float | None
    lot: float | None
    strategy: str
    timeframe: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


class TradingViewSignalParser:
    """Parse and validate TradingView alert payloads.

    Supports:
    1. Custom Pine Script JSON: {"symbol": "...", "side": "buy", ...}
    2. TradingView template variables: {"ticker": "{{ticker}}", ...}
    3. Plain text alerts: "BUY EURUSD @ 1.0850 SL=30 TP=60"
    4. TradingView default format with {{strategy.order.action}}
    """

    # Valid forex pairs, crypto, stocks (MT5 symbols)
    KNOWN_FOREX = {
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCAD", "EURAUD", "GBPCHF",
        "XAUUSD", "XAGUSD",  # gold/silver
    }
    KNOWN_CRYPTO = {
        "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOTUSD",
        "LINKUSD", "AVAXUSD", "DOGEUSD", "MATICUSD",
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
    }

    def parse_json(self, data: dict[str, Any]) -> TradingSignal | None:
        """Parse a JSON payload into a TradingSignal."""
        normalized = _normalize_signal(data)

        if normalized["side"] == "UNKNOWN":
            return None

        if normalized["symbol"] == "UNKNOWN":
            return None

        return TradingSignal(
            symbol=normalized["symbol"],
            side=normalized["side"],
            price=normalized["price"],
            sl=normalized["sl"],
            tp=normalized["tp"],
            lot=normalized["lot"],
            strategy=normalized["strategy"],
            timeframe=normalized["timeframe"],
            raw=data,
        )

    def parse_text(self, text: str) -> TradingSignal | None:
        """Parse a plain text TradingView alert into a TradingSignal.

        Examples:
            "BUY EURUSD @ 1.0850 SL=30 TP=60"
            "SELL XAUUSD"
            "Long BTCUSD sl=65000 tp=75000 lot=0.01"
        """
        text = text.strip()
        if not text:
            return None

        # Extract side
        side_match = re.search(
            r"\b(BUY|SELL|LONG|SHORT|CLOSE|EXIT)\b", text, re.IGNORECASE
        )
        if not side_match:
            return None
        side = side_match.group(1).upper()
        if side in ("LONG", "BUY"):
            side = "BUY"
        elif side in ("SHORT", "SELL"):
            side = "SELL"
        elif side in ("CLOSE", "EXIT"):
            side = "CLOSE"

        # Extract symbol — try forex/crypto patterns first (6+ chars with currency)
        symbol = None
        # Pattern: 6+ uppercase letters ending with currency pair
        symbol_match = re.search(
            r"\b([A-Z]{4,8}(?:USDT?|JPY|EUR|GBP|CHF|CAD|AUD|NZD))\b", text.upper()
        )
        if not symbol_match:
            # Try gold/silver
            symbol_match = re.search(r"\b(XAUUSD|XAGUSD)\b", text.upper())
        if symbol_match:
            symbol = symbol_match.group(1)
        else:
            # Generic ticker — look for 2-6 uppercase letters after the action word
            remaining = text[side_match.end():].strip()
            ticker_match = re.match(r"^([A-Z]{2,6})", remaining.strip())
            if ticker_match:
                candidate = ticker_match.group(1)
                skip_words = {
                    "SL", "TP", "LOT", "PRICE", "AT", "THE", "AND",
                    "WITH", "FOR", "SET", "STOP", "TAKE", "LIMIT",
                }
                if candidate not in skip_words:
                    symbol = candidate

        if not symbol:
            return None

        # Extract price
        price = None
        price_match = re.search(r"@?\s*(\d+\.?\d*)", text)
        if price_match:
            try:
                price = float(price_match.group(1))
            except ValueError:
                pass

        # Extract SL/TP (in pips or price level)
        sl = None
        tp = None
        sl_match = re.search(r"[Ss][Ll]\s*[=:]\s*(\d+\.?\d*)", text)
        if sl_match:
            try:
                sl = float(sl_match.group(1))
            except ValueError:
                pass
        tp_match = re.search(r"[Tt][Pp]\s*[=:]\s*(\d+\.?\d*)", text)
        if tp_match:
            try:
                tp = float(tp_match.group(1))
            except ValueError:
                pass

        # Extract lot size
        lot = None
        lot_match = re.search(r"[Ll][Oo][Tt]\s*[=:]\s*(\d+\.?\d*)", text)
        if lot_match:
            try:
                lot = float(lot_match.group(1))
            except ValueError:
                pass

        return TradingSignal(
            symbol=symbol,
            side=side,
            price=price,
            sl=sl,
            tp=tp,
            lot=lot,
            strategy="text_alert",
            timeframe="",
            raw={"text": text},
        )

    def validate(self, signal: TradingSignal) -> tuple[bool, str]:
        """Validate a signal against known rules.

        Returns (is_valid, reason).
        """
        # Symbol must be recognized
        all_known = self.KNOWN_FOREX | self.KNOWN_CRYPTO
        # Also allow any ticker that looks valid (2-10 uppercase chars)
        if not re.match(r"^[A-Z0-9]{2,12}$", signal.symbol):
            return False, f"invalid symbol: {signal.symbol}"

        # Side must be actionable
        if signal.side not in ("BUY", "SELL", "CLOSE"):
            return False, f"invalid side: {signal.side}"

        # If SL/TP provided as pips (< 10000), convert to price if price is known
        # (Conversion happens in the executor, not here)

        return True, "valid"
