"""MT5 HTTP Executor — connects to the Windows-side MT5 bridge via HTTP.

The MetaTrader5 Python package is Windows-only, so we run a small HTTP bridge
server on Windows that exposes MT5 functions. This executor talks to that server.

    mt5_windows_bridge.py on Windows (port 5002)
    WSL executor: this file -- sends HTTP requests to Windows bridge

Usage::
    executor = MT5HttpExecutor()
    if executor.connect():
        result = executor.market_buy("EURUSD", lot=0.01, sl=1.0830, tp=1.0900)
"""

import logging
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an MT5 order execution."""
    success: bool
    order_id: int | None = None
    ticket: int | None = None
    volume: float | None = None
    price: float | None = None
    comment: str = ""
    error_code: int = 0


@dataclass
class AccountInfo:
    """MT5 account information."""
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    leverage: int
    currency: str
    server: str


@dataclass
class PositionInfo:
    """Open position in MT5."""
    ticket: int
    symbol: str
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    comment: str
    type: int


class MT5HttpExecutor:
    """Execute trades on MT5 via the Windows-side HTTP bridge.

    The bridge server runs on Windows at http://127.0.0.1:5002.
    """

    def __init__(self, bridge_url: str = "http://127.0.0.1:5002"):
        """Initialize HTTP executor.

        Args:
            bridge_url: URL of the Windows-side MT5 bridge server.
        """
        self._url = bridge_url.rstrip("/")
        self._connected = False
        self._account_info: AccountInfo | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _request(self, method: str, path: str, data: dict | None = None) -> dict:
        """Send HTTP request to the bridge server."""
        url = f"{self._url}{path}"
        body = json.dumps(data).encode("utf-8") if data else None

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode("utf-8"))
                return {**body, "status_code": e.code}
            except Exception:
                return {"error": str(e), "status_code": e.code}
        except Exception as e:
            return {"error": str(e)}

    def connect(self) -> bool:
        """Connect to the MT5 bridge server."""
        result = self._request("POST", "/connect")
        if result.get("success"):
            self._connected = True
            self._account_info = AccountInfo(
                login=result.get("account", 0),
                balance=result.get("balance", 0),
                equity=0,
                margin=0,
                free_margin=0,
                leverage=0,
                currency=result.get("currency", ""),
                server=result.get("server", ""),
            )
            log.info(
                "MT5 bridge connected: account=%d, balance=%.2f, server=%s",
                result.get("account"), result.get("balance"), result.get("server"),
            )
            return True
        log.error("MT5 bridge connection failed: %s", result.get("error", "unknown"))
        return False

    def shutdown(self):
        """Disconnect from bridge."""
        if self._connected:
            self._request("POST", "/disconnect")
            self._connected = False
            log.info("MT5 bridge disconnected")

    def health(self) -> dict:
        """Check bridge health."""
        return self._request("GET", "/health")

    def get_account_info(self) -> AccountInfo | None:
        """Get account info from bridge."""
        if not self._connected:
            return None
        result = self._request("GET", "/account")
        if "error" in result:
            return None
        self._account_info = AccountInfo(
            login=result.get("login", 0),
            balance=result.get("balance", 0),
            equity=result.get("equity", 0),
            margin=result.get("margin", 0),
            free_margin=result.get("free_margin", 0),
            leverage=result.get("leverage", 0),
            currency=result.get("currency", ""),
            server=result.get("server", ""),
        )
        return self._account_info

    def get_positions(self, symbol: str | None = None) -> list[PositionInfo]:
        """Get open positions."""
        path = "/positions"
        if symbol:
            path += f"?symbol={symbol}"
        result = self._request("GET", path)
        if isinstance(result, list):
            return [
                PositionInfo(
                    ticket=p["ticket"],
                    symbol=p["symbol"],
                    volume=p["volume"],
                    price_open=p["price_open"],
                    price_current=p["price_current"],
                    sl=p["sl"],
                    tp=p["tp"],
                    profit=p["profit"],
                    comment=p.get("comment", ""),
                    type=p["type"],
                )
                for p in result
            ]
        return []

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """Get symbol specification."""
        result = self._request("GET", f"/symbol/{symbol}")
        if "error" in result:
            return None
        return result

    def get_symbols(self) -> list[str]:
        """Get available symbols."""
        result = self._request("GET", "/symbols")
        if isinstance(result, list):
            return result
        return []

    def market_buy(self, symbol: str, lot: float = 0.01,
                   sl: float | None = None, tp: float | None = None,
                   comment: str = "TradingAgents") -> OrderResult:
        """Execute a market buy order."""
        return self._execute_trade(symbol, "buy", lot, sl, tp, comment)

    def market_sell(self, symbol: str, lot: float = 0.01,
                    sl: float | None = None, tp: float | None = None,
                    comment: str = "TradingAgents") -> OrderResult:
        """Execute a market sell order."""
        return self._execute_trade(symbol, "sell", lot, sl, tp, comment)

    def close_position(self, ticket: int, comment: str = "TradingAgents") -> OrderResult:
        """Close a position by ticket."""
        result = self._request("POST", "/trade", {
            "action": "close",
            "ticket": ticket,
            "comment": comment,
        })
        return self._parse_trade_result(result)

    def close_all_positions(self, symbol: str | None = None,
                            comment: str = "TradingAgents") -> list[OrderResult]:
        """Close all positions."""
        result = self._request("POST", "/trade", {
            "action": "close_all",
            "comment": comment,
        })
        results_list = result.get("results", [])
        return [
            OrderResult(success=r.get("success", False), ticket=r.get("ticket"))
            for r in results_list
        ]

    def _execute_trade(self, symbol: str, action: str, lot: float,
                       sl: float | None, tp: float | None,
                       comment: str) -> OrderResult:
        """Execute a trade via the bridge."""
        if not self._connected:
            return OrderResult(success=False, comment="not connected")

        payload: dict[str, Any] = {
            "symbol": symbol,
            "action": action,
            "lot": lot,
            "comment": comment,
        }
        if sl is not None:
            payload["sl"] = sl
        if tp is not None:
            payload["tp"] = tp

        log.info("MT5 order: %s %s lot=%.2f sl=%s tp=%s", action.upper(), symbol, lot, sl, tp)
        result = self._request("POST", "/trade", payload)
        return self._parse_trade_result(result)

    def _parse_trade_result(self, result: dict) -> OrderResult:
        """Parse bridge response into OrderResult."""
        if result.get("success"):
            log.info(
                "Order executed: ticket=%d, vol=%.2f, price=%.5f",
                result.get("ticket", 0),
                result.get("volume", 0),
                result.get("price", 0),
            )
            return OrderResult(
                success=True,
                order_id=result.get("ticket"),
                ticket=result.get("ticket"),
                volume=result.get("volume"),
                price=result.get("price"),
                comment=result.get("comment", ""),
            )
        else:
            log.warning("Order failed: %s", result.get("error", "unknown"))
            return OrderResult(
                success=False,
                comment=result.get("error", "unknown"),
                error_code=result.get("status_code", result.get("retcode", -1)),
            )

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        acct = f"#{self._account_info.login}" if self._account_info else ""
        return f"MT5HttpExecutor({status}{acct})"
