"""OANDA REST API backup broker adapter.

Mirrors the MT5Connector interface so the execution engine can fail over
transparently. Uses the practice or live endpoint based on OANDA_ENV.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from config import settings
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["OandaConnector"]

UNITS = {"EURUSD": 100000, "GBPUSD": 100000, "USDJPY": 100000, "XAUUSD": 100,
         "NAS100": 1, "US30": 1}


class OandaConnector:
    """OANDA v20 REST adapter."""

    name = "oanda"

    def __init__(self, api_key: str = "", account_id: str = "") -> None:
        self.api_key = api_key or settings.OANDA_API_KEY
        self.account_id = account_id or settings.OANDA_ACCOUNT_ID
        env = "practice" if settings.OANDA_ENV != "live" else "www"
        self.base = f"https://{env}.oanda.com/v3"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
        self._available = bool(self.api_key and self.account_id)

    def available(self) -> bool:
        """Configured with key + account."""
        return self._available

    def _call(self, method: str, path: str, retries: int = 3,
              **kwargs) -> Optional[dict]:
        """HTTP with exponential backoff; None on final failure."""
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.request(method, f"{self.base}{path}",
                                            timeout=15, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                logger.warning("oanda %s %s attempt %d failed: %s", method, path,
                               attempt, exc)
                time.sleep(2 ** attempt)
        return None

    def connect(self) -> bool:
        """Validate credentials with an account summary call."""
        if not self._available:
            return False
        data = self._call("GET", f"/accounts/{self.account_id}/summary", retries=2)
        ok = data is not None
        logger.info("oanda connect: %s", "ok" if ok else "failed")
        return ok

    def healthy(self) -> bool:
        """Account reachable."""
        return self._available and self._call(
            "GET", f"/accounts/{self.account_id}/summary", retries=1) is not None

    def account(self) -> dict:
        """Balance/equity/margin snapshot."""
        data = self._call("GET", f"/accounts/{self.account_id}/summary")
        if not data:
            raise RuntimeError("oanda account unavailable")
        a = data["account"]
        return {
            "login": self.account_id, "balance": float(a["balance"]),
            "equity": float(a["NAV"]), "margin_used": float(a["marginUsed"]),
            "margin_free": float(a["marginAvailable"]),
            "margin_level": float(a["marginRate"] or 0) * 100.0 or 1000.0,
            "currency": a["currency"],
        }

    def candles(self, pair: str, timeframe_min: int, count: int) -> list[dict]:
        """OHLCV rows ascending from OANDA candles endpoint."""
        granularity = {1: "M1", 5: "M5", 15: "M15", 60: "H1", 240: "H4",
                       1440: "D"}.get(timeframe_min, "M15")
        pair_fmt = f"{pair[:3]}_{pair[3:]}" if pair.upper() not in ("NAS100", "US30") \
            else "NAS100_USD"
        data = self._call("GET", f"/instruments/{pair_fmt}/candles",
                          params={"granularity": granularity, "count": min(count, 500),
                                  "price": "M"})
        if not data:
            return []
        rows = []
        for c in data.get("candles", []):
            if not c.get("complete", False):
                continue
            mid = c["mid"]
            rows.append({"time": c["time"], "open": float(mid["o"]),
                         "high": float(mid["h"]), "low": float(mid["l"]),
                         "close": float(mid["c"]), "volume": float(c["volume"])})
        return rows

    def market_order(self, pair: str, direction: str, lots: float,
                     sl: float, tp: float, comment: str = "bot") -> int:
        """Market order with attached SL/TP; returns order id."""
        pair_fmt = f"{pair[:3]}_{pair[3:]}"
        units = int(UNITS.get(pair.upper(), 100000) * lots)
        if direction == "sell":
            units = -units
        body = {
            "order": {
                "type": "MARKET_ORDER", "instrument": pair_fmt, "units": str(units),
                "timeInForce": "FOK", "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": f"{sl:.5f}"},
                "takeProfitOnFill": {"price": f"{tp:.5f}"},
                "clientExtensions": {"comment": comment[:26]},
            }
        }
        data = self._call("POST", f"/accounts/{self.account_id}/orders", json=body)
        if not data or "orderFillTransaction" not in data:
            raise RuntimeError(f"oanda order failed: {str(data)[:150]}")
        return int(data["orderFillTransaction"]["id"])

    def close_position(self, ticket: int, lots: Optional[float] = None) -> bool:
        """Close a trade by id (partial via units when provided)."""
        data = self._call("GET", f"/accounts/{self.account_id}/trades/{ticket}")
        if not data:
            return False
        trade = data["trade"]
        units = trade["currentUnits"]
        if lots is not None:
            sign = -1 if int(units) > 0 else 1
            units = str(sign * int(UNITS.get(trade["instrument"].replace("_", ""), 100000)
                                   * lots))
        body = {"units": str(units)}
        resp = self._call("PUT", f"/accounts/{self.account_id}/trades/{ticket}/close",
                          json=body)
        return bool(resp and resp.get("orderFillTransaction"))

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        """Replace SL/TP orders on a trade."""
        ok_sl = self._call("PUT", f"/accounts/{self.account_id}/trades/{ticket}/orders",
                           json={"stopLoss": {"price": f"{sl:.5f}"}})
        ok_tp = self._call("PUT", f"/accounts/{self.account_id}/trades/{ticket}/orders",
                           json={"takeProfitOnFill": {"price": f"{tp:.5f}"}})
        return bool(ok_sl and ok_tp)

    def positions(self) -> list[dict]:
        """Open trades as plain dicts."""
        data = self._call("GET", f"/accounts/{self.account_id}/openTrades")
        if not data:
            return []
        out = []
        for t in data.get("trades", []):
            instrument = t["instrument"].replace("_", "")
            out.append({
                "ticket": int(t["id"]), "pair": instrument,
                "direction": "buy" if int(t["currentUnits"]) > 0 else "sell",
                "lots": abs(float(t["currentUnits"])) / UNITS.get(instrument, 100000),
                "entry": float(t["price"]), "sl": float(t.get("stopLossOrder", {})
                                                     .get("price", 0) or 0),
                "tp": 0.0, "profit": float(t.get("unrealizedPL", 0)),
                "swap": float(t.get("financing", 0)), "opened": 0,
            })
        return out
