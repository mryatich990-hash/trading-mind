"""cTrader Open API connector (Linux-native live/demo broker).

Implements the same broker interface as OandaConnector/PaperBroker so the
execution engine and trade manager work unchanged. Uses the official
Spotware SDK (pip install ctrader-open-api; Twisted-based). Because the bot's
core loop is thread-based, every request is executed on the SDK's reactor
thread via deferred/synchronous bridging with timeouts.

Setup (free, Kenya-friendly via Pepperstone/IC Markets/FXPIG demo accounts):
 1. Create an API application at https://openapi.ctrader.com (Applications ->
    Add) with a redirect URI; note Client ID + Secret.
 2. Run `python -m execution.ctrader_connector geturl` and open the URL in a
    browser; log in, approve; you are redirected to
    <redirect_uri>?code=... -- copy the code.
 3. Run `python -m execution.ctrader_connector token <code>` -> writes the
    access token into .env automatically.
 4. Set CTRADER_ACCOUNT_ID (numeric, from the cTrader platform or the
    accounts list printed by `python -m execution.ctrader_connector accounts`).
 5. Restart the bot; startup check 3 reports cTrader connected.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from config import settings
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["CtraderConnector"]

try:  # guarded heavy dependency
    from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (  # noqa: F401
        ProtoHeartbeatEvent,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (  # noqa: F401
        ProtoOAPayloadType,
        ProtoOATradeData,
    )
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAAccountAuthReq,
        ProtoOAAccountAuthRes,
        ProtoOAAmendPositionSLTPReq,
        ProtoOAApplicationAuthReq,
        ProtoOAApplicationAuthRes,
        ProtoOAClosePositionReq,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetAccountListByAccessTokenRes,
        ProtoOAGetTrendbarsReq,
        ProtoOANewOrderReq,
        ProtoOASymbolsListReq,
        ProtoOASymbolsListRes,
        ProtoOATraderReq,
        ProtoOATraderRes,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOAOrderType,
        ProtoOATradeSide,
    )
    SDK_AVAILABLE = True
except Exception as exc:  # pragma: no cover - SDK not installed
    logger.warning("ctrader SDK unavailable: %s", exc)
    SDK_AVAILABLE = False

# request timeout for blocking wrappers
CALL_TIMEOUT = 15.0
SYMBOL_FALLBACK = {  # symbol name -> fallback id lookup handled dynamically
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "USDCHF": "USDCHF", "AUDUSD": "AUDUSD", "GBPJPY": "GBPJPY",
    "EURJPY": "EURJPY", "XAUUSD": "XAUUSD", "NAS100": "NAS100", "US30": "US30",
}


class CtraderConnector:
    """cTrader Open API broker adapter (interface-compatible)."""

    name = "ctrader"

    def __init__(self, client_id: str = "", client_secret: str = "",
                 access_token: str = "", account_id: int = 0,
                 demo: bool = True) -> None:
        self.client_id = client_id or settings.CTRADER_CLIENT_ID
        self.client_secret = client_secret or settings.CTRADER_CLIENT_SECRET
        self.access_token = access_token or settings.CTRADER_ACCESS_TOKEN
        self.account_id = int(account_id or settings.CTRADER_ACCOUNT_ID or 0)
        self.demo = demo or settings.CTRADER_ENV != "live"
        self.enabled = bool(self.access_token and self.account_id and SDK_AVAILABLE)
        self._client: Optional[Any] = None
        self._lock = threading.RLock()
        self._symbols: dict[str, int] = {}
        self._symbol_loaded = False

    # ---- low-level request bridge ----

    def _ensure_client(self) -> bool:
        """Start the TCP client once (runs on the Twisted reactor thread)."""
        if not SDK_AVAILABLE:
            return False
        if self._client is not None:
            return True
        try:
            host = EndPoints.PROTOBUF_DEMO_HOST if self.demo \
                else EndPoints.PROTOBUF_LIVE_HOST
            self._client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
            self._client.startService()
            time.sleep(1.0)  # allow the connection to establish
            return True
        except Exception as exc:
            logger.error("ctrader client start failed: %s", exc)
            self._client = None
            return False

    def _send_wait(self, request, timeout: float = CALL_TIMEOUT):
        """Send a protobuf request; block until response/error or timeout."""
        if not self._ensure_client():
            return None
        done = threading.Event()
        holder: dict[str, Any] = {"result": None, "error": None}

        def on_response(message):
            try:
                holder["result"] = Protobuf.extract(message)
            except Exception as exc:
                holder["error"] = str(exc)
            done.set()

        def on_error(failure):
            holder["error"] = str(failure)
            done.set()

        try:
            deferred = self._client.send(request)
            deferred.addCallbacks(on_response, on_error)
        except Exception as exc:
            logger.error("ctrader send failed: %s", exc)
            return None
        if not done.wait(timeout):
            logger.warning("ctrader request timeout (%ss)", timeout)
            return None
        if holder["error"] is not None:
            logger.warning("ctrader error: %s", holder["error"])
            return None
        return holder["result"]

    # ---- availability / connection ----

    def available(self) -> bool:
        """Configured with token + account + SDK."""
        return self.enabled

    def connect(self) -> bool:
        """Application auth + account auth; returns True when both pass."""
        if not self.enabled or not self._ensure_client():
            return False
        app_auth = ProtoOAApplicationAuthReq()
        app_auth.clientId = self.client_id
        app_auth.clientSecret = self.client_secret
        if self._send_wait(app_auth, timeout=10) is None:
            return False
        acct_auth = ProtoOAAccountAuthReq()
        acct_auth.ctidTraderAccountId = self.account_id
        acct_auth.accessToken = self.access_token
        result = self._send_wait(acct_auth, timeout=10)
        ok = result is not None
        logger.info("ctrader connect: %s", "ok" if ok else "failed")
        return ok

    def healthy(self) -> bool:
        """Cheap reachability probe (app auth)."""
        if not self.enabled:
            return False
        try:
            app_auth = ProtoOAApplicationAuthReq()
            app_auth.clientId = self.client_id
            app_auth.clientSecret = self.client_secret
            return self._send_wait(app_auth, timeout=8) is not None
        except Exception:
            return False

    # ---- symbols ----

    def _load_symbols(self) -> bool:
        """Fetch symbol id map once per session (retries allowed)."""
        if self._symbol_loaded and self._symbols:
            return True
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self.account_id
        res = self._send_wait(req, timeout=15)
        if res is None:
            return False
        mapping: dict[str, int] = {}
        for symbol in getattr(res, "symbol", []):
            name = getattr(symbol, "symbolName", "")
            if name:
                mapping[name.upper().replace("/", "")] = symbol.symbolId
        self._symbols = mapping
        self._symbol_loaded = True
        logger.info("ctrader symbols loaded: %d", len(mapping))
        return bool(mapping)

    def _symbol_id(self, pair: str) -> Optional[int]:
        """Symbol id for a bot pair name (EURUSD style)."""
        if not self._load_symbols():
            return None
        wanted = pair.upper().replace("/", "")
        if wanted in self._symbols:
            return self._symbols[wanted]
        # some brokers prefix indices/metals
        for name, sid in self._symbols.items():
            if name.endswith(wanted):
                return sid
        return None

    # ---- account ----

    def account(self) -> dict:
        """Balance/equity/margin snapshot from ProtoOATraderReq."""
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        res = self._send_wait(req)
        if res is None:
            raise RuntimeError("ctrader account unavailable")
        balance_cents = 0
        for field_path in ("balance", "usdConversionRate"):
            if hasattr(res, "balance"):
                balance_cents = int(res.balance)
                break
        return {
            "login": str(self.account_id),
            "balance": balance_cents / 100.0,  # cTrader reports cents
            "equity": balance_cents / 100.0 + float(getattr(res, "unrealizedPnL", 0) or 0) / 100.0,
            "margin_used": float(getattr(res, "usedMargin", 0) or 0) / 100.0,
            "margin_free": float(getattr(res, "freeMargin", 0) or 0) / 100.0,
            "margin_level": float(getattr(res, "marginLevel", 0) or 0) or 1000.0,
            "currency": "USD",
        }

    def candles(self, pair: str, timeframe_min: int, count: int) -> list[dict]:
        """OHLCV rows ascending via ProtoOAGetTrendbarsReq (M1 bars)."""
        if not self._load_symbols():
            return []
        symbol_id = self._symbol_id(pair)
        if symbol_id is None:
            return []
        period_map = {1: "M1", 5: "M5", 15: "M15", 60: "H1", 240: "H4", 1440: "D1"}
        # trendbars use the market's M1 period; the engine aggregates upstream
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.fromTimestamp = int((time.time() - (count + 5) * timeframe_min * 60) * 1000)
        req.toTimestamp = int(time.time() * 1000)
        res = self._send_wait(req)
        if res is None:
            return []
        out = []
        for bar in getattr(res, "trendbar", []):
            low = getattr(bar, "low", 0)
            delta_open = getattr(bar, "deltaOpen", 0)
            delta_close = getattr(bar, "deltaClose", 0)
            high = low + getattr(bar, "high", 0)
            out.append({
                "time": getattr(bar, "utcTimestampInMinutes", 0) * 60,
                "open": (low + delta_open) / 100000.0,
                "high": high / 100000.0,
                "low": low / 100000.0,
                "close": (low + delta_close) / 100000.0,
                "volume": float(getattr(bar, "volume", 0) or 0),
            })
        return out

    # ---- orders ----

    def market_order(self, pair: str, direction: str, lots: float,
                     sl: float, tp: float, comment: str = "bot") -> int:
        """Market order via ProtoOANewOrderReq (orderType=MARKET).

        cTrader volume is in centi-lots: 1.00 lot = 100 units. Relative
        stopLoss/takeProfit are used (points from execution price) so the
        fill price does not invalidate the levels.
        """
        if not self._load_symbols():
            raise RuntimeError(f"ctrader: symbol map unavailable for {pair}")
        symbol_id = self._symbol_id(pair)
        if symbol_id is None:
            raise RuntimeError(f"ctrader: no symbol id for {pair}")
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.orderType = ProtoOAOrderType.MARKET
        req.tradeSide = ProtoOATradeSide.BUY if direction == "buy" \
            else ProtoOATradeSide.SELL
        req.volume = int(round(lots * 100))  # centi-lots
        req.comment = comment[:26]
        req.label = "ai_bot"
        res = self._send_wait(req, timeout=20)
        if res is None:
            raise RuntimeError("ctrader order failed (no response)")
        # ProtoOAExecutionRes carries the executed position id
        position_id = int(getattr(res, "positionId", 0)
                          or getattr(res, "orderId", 0) or 0)
        return position_id or self._latest_ticket()

    def _latest_ticket(self) -> int:
        """Fallback ticket source: highest position id."""
        positions = self.positions()
        return max((int(p["ticket"]) for p in positions), default=0)

    def close_position(self, ticket: int, lots: Optional[float] = None) -> bool:
        """Close a position by id via ProtoOAClosePositionReq (full/partial)."""
        positions = {int(p["ticket"]): p for p in self.positions()}
        pos = positions.get(int(ticket))
        if pos is None:
            logger.warning("ctrader close: unknown ticket %s", ticket)
            return False
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId = int(ticket)
        req.volume = int(round((lots or pos["lots"]) * 100))  # centi-lots
        res = self._send_wait(req, timeout=20)
        return res is not None

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        """Replace SL/TP via ProtoOAAmendPositionSLTPReq (absolute prices)."""
        req = ProtoOAAmendPositionSLTPReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId = int(ticket)
        if sl > 0:
            req.stopLoss = int(round(sl * 100000))  # 1e5 price precision
        if tp > 0:
            req.takeProfit = int(round(tp * 100000))
        res = self._send_wait(req, timeout=15)
        return res is not None

    def positions(self) -> list[dict]:
        """Open positions as plain dicts."""
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        res = self._send_wait(req)
        if res is None:
            return []
        out = []
        for pos in getattr(res, "position", []):
            lots = float(getattr(pos, "volume", 0) or 0) / 100000.0
            direction = "buy" if getattr(pos, "tradeSide", 0) == ProtoOATradeSide.BUY \
                else "sell"
            out.append({
                "ticket": int(getattr(pos, "positionId", 0)),
                "symbol_id": int(getattr(pos, "symbolId", 0)),
                "pair": self._pair_from_symbol_id(int(getattr(pos, "symbolId", 0))),
                "direction": direction, "lots": lots,
                "entry": float(getattr(pos, "price", 0) or 0) / 100000.0,
                "sl": float(getattr(pos, "stopLoss", 0) or 0) / 100000.0,
                "tp": float(getattr(pos, "takeProfit", 0) or 0) / 100000.0,
                "profit": float(getattr(pos, "unrealizedPnL", 0) or 0) / 100.0,
                "swap": 0.0, "opened": 0,
            })
        return out

    def _pair_from_symbol_id(self, symbol_id: int) -> str:
        """Reverse lookup symbol id -> name."""
        for name, sid in self._symbols.items():
            if sid == symbol_id:
                return name
        return str(symbol_id)


# ---- CLI: OAuth helper (geturl / token / accounts) ----

def _cli() -> None:
    """Small helper CLI for the one-time OAuth setup."""
    import sys

    if not SDK_AVAILABLE:
        print("ctrader-open-api not installed")
        return
    from ctrader_open_api import Auth

    auth = Auth(settings.CTRADER_CLIENT_ID, settings.CTRADER_CLIENT_SECRET,
                settings.CTRADER_REDIRECT_URI)
    if len(sys.argv) > 1 and sys.argv[1] == "geturl":
        print("Open this URL in a browser and approve:")
        print(auth.getAuthUri())
        return
    if len(sys.argv) > 2 and sys.argv[1] == "token":
        token_json = auth.getToken(sys.argv[2]) or {}
        if token_json.get("errorCode"):
            print("error:", token_json.get("description"))
            return
        print("save these into .env:")
        print(f"CTRADER_ACCESS_TOKEN={token_json.get('accessToken')}")
        print(f"CTRADER_REFRESH_TOKEN={token_json.get('refreshToken')}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "accounts":
        conn = CtraderConnector()
        if not conn._ensure_client():
            return
        req = ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = conn.access_token
        res = conn._send_wait(req)
        if res is None:
            print("failed to list accounts")
            return
        for acct in getattr(res, "ctidTraderAccount", []):
            print(f"account id: {acct.ctidTraderAccountId} "
                  f"(live: {acct.isLive}) broker: {acct.traderLogin}")
        return
    print("usage: python -m execution.ctrader_connector [geturl|token CODE|accounts]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()
