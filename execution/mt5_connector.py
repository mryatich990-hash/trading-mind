"""MT5 connector: guarded-import broker adapter with reconnection and failover hooks.

MetaTrader5 is Windows/macOS only, so the import is guarded: on Railway the
connector reports unavailable and the execution engine falls back to OANDA or
shadow mode. All operations return plain dicts to keep the rest of the system
broker-agnostic.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from config import settings
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["BrokerError", "MT5Connector"]


class BrokerError(Exception):
    """Broker operation failure."""


class MT5Connector:
    """MetaTrader5 adapter; every method raises BrokerError when unavailable."""

    name = "mt5"

    def __init__(self, login: str = "", password: str = "", server: str = "") -> None:
        self.login = login or settings.MT5_LOGIN
        self.password = password or settings.MT5_PASSWORD
        self.server = server or settings.MT5_SERVER
        self.mt5: Any = None
        self._connected = False
        self._backoff = 1.0

    # ---- connection ----

    def available(self) -> bool:
        """True when the MetaTrader5 package imports."""
        try:
            import MetaTrader5  # noqa: F401
            return True
        except ImportError:
            return False

    def connect(self) -> bool:
        """Initialize + login with exponential backoff (1s, 2s, 4s... max 60s)."""
        if not self.available():
            logger.info("MetaTrader5 package unavailable; MT5 connector disabled")
            return False
        import MetaTrader5 as mt5
        self.mt5 = mt5
        for attempt in range(6):
            try:
                kwargs: dict[str, Any] = {}
                if self.server:
                    kwargs["server"] = self.server
                if not mt5.initialize(**kwargs):
                    raise BrokerError(f"initialize: {mt5.last_error()}")
                if self.login:
                    if not mt5.login(int(self.login), password=self.password,
                                     server=self.server):
                        raise BrokerError(f"login: {mt5.last_error()}")
                self._connected = True
                self._backoff = 1.0
                logger.info("MT5 connected (server=%s)", self.server)
                return True
            except Exception as exc:
                wait = min(self._backoff, 60.0)
                logger.warning("MT5 connect attempt %d failed: %s (retry in %.0fs)",
                               attempt + 1, exc, wait)
                time.sleep(wait)
                self._backoff *= 2
        return False

    def _require(self) -> Any:
        """Return the mt5 module or raise."""
        if not self._connected and not self.connect():
            raise BrokerError("MT5 not connected")
        return self.mt5

    def healthy(self) -> bool:
        """Terminal still reachable."""
        if not self._connected:
            return False
        try:
            info = self.mt5.account_info()
            return info is not None
        except Exception:
            return False

    # ---- account ----

    def account(self) -> dict:
        """Balance, equity, margin snapshot."""
        mt5 = self._require()
        info = mt5.account_info()
        if info is None:
            raise BrokerError("account_info returned None")
        return {
            "login": int(info.login), "balance": float(info.balance),
            "equity": float(info.equity), "margin_used": float(info.margin),
            "margin_free": float(info.margin_free),
            "margin_level": float(info.margin_level) if info.margin_level else 1000.0,
            "currency": info.currency,
        }

    # ---- prices ----

    def tick(self, pair: str) -> dict:
        """Latest bid/ask."""
        mt5 = self._require()
        t = mt5.symbol_info_tick(pair.upper())
        if t is None:
            raise BrokerError(f"no tick for {pair}")
        return {"bid": float(t.bid), "ask": float(t.ask),
                "time": int(t.time), "pair": pair.upper()}

    def candles(self, pair: str, timeframe_min: int, count: int) -> list[dict]:
        """OHLCV rows ascending."""
        mt5 = self._require()
        tf_map = {1: "M1", 5: "M5", 15: "M15", 60: "H1", 240: "H4", 1440: "D1"}
        tf = getattr(mt5, f"TIMEFRAME_{tf_map.get(timeframe_min, 'M15')}")
        rates = mt5.copy_rates_from_pos(pair.upper(), tf, 0, max(count, 30))
        if rates is None or not len(rates):
            raise BrokerError(f"no rates for {pair} {timeframe_min}m")
        return [{"time": int(r["time"]), "open": float(r["open"]), "high": float(r["high"]),
                 "low": float(r["low"]), "close": float(r["close"]),
                 "volume": float(r["tick_volume"])} for r in rates]

    def swap_rates(self, pair: str) -> tuple[float, float]:
        """(long_swap, short_swap) per lot per night."""
        mt5 = self._require()
        info = mt5.symbol_info(pair.upper())
        if info is None:
            return 0.0, 0.0
        return float(info.swap_long), float(info.swap_short)

    # ---- orders ----

    def market_order(self, pair: str, direction: str, lots: float,
                     sl: float, tp: float, comment: str = "bot") -> int:
        """Send a market order; returns the deal ticket. One requote retry."""
        mt5 = self._require()
        symbol = pair.upper()
        if not mt5.symbol_select(symbol, True):
            raise BrokerError(f"symbol_select failed for {symbol}")
        price_tick = mt5.symbol_info_tick(symbol)
        if price_tick is None:
            raise BrokerError(f"no tick for {symbol}")
        order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
        price = float(price_tick.ask) if direction == "buy" else float(price_tick.bid)
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
            "volume": float(lots), "type": order_type, "price": price,
            "sl": float(sl), "tp": float(tp), "deviation": 20,
            "magic": 20260901, "comment": comment[:26],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        for attempt in (1, 2):  # one requote retry at fresh market price
            result = mt5.order_send(request)
            if result is None:
                raise BrokerError("order_send returned None")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("MT5 order filled: %s %s %.2f @ %.5f (ticket %s)",
                            symbol, direction, lots, result.price, result.order)
                return int(result.order)
            if result.retcode == mt5.TRADE_RETCODE_REQUOTE and attempt == 1:
                tick = mt5.symbol_info_tick(symbol)
                request["price"] = float(tick.ask if direction == "buy" else tick.bid)
                continue
            raise BrokerError(f"order rejected: retcode={result.retcode} "
                              f"{getattr(result, 'comment', '')}")
        raise BrokerError("order failed after requote retry")

    def close_position(self, ticket: int, lots: Optional[float] = None) -> bool:
        """Close a full or partial position at market."""
        mt5 = self._require()
        positions = mt5.positions_get(ticket=ticket) or []
        if not positions:
            return False
        pos = positions[0]
        volume = float(lots) if lots else float(pos.volume)
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": int(ticket),
            "price": float(tick.bid if is_buy else tick.ask),
            "deviation": 20, "magic": 20260901, "comment": "bot-close",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise BrokerError(f"close failed: {getattr(result, 'retcode', 'None')}")
        return True

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        """Move stop loss / take profit on an open position."""
        mt5 = self._require()
        positions = mt5.positions_get(ticket=ticket) or []
        if not positions:
            return False
        request = {
            "action": mt5.TRADE_ACTION_SLTP, "position": int(ticket),
            "sl": float(sl), "tp": float(tp),
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    def positions(self) -> list[dict]:
        """All open positions as plain dicts (for state rebuild on restart)."""
        if not self._connected:
            return []
        mt5 = self.mt5
        rows = []
        for p in mt5.positions_get() or []:
            rows.append({
                "ticket": int(p.ticket), "pair": p.symbol,
                "direction": "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
                "lots": float(p.volume), "entry": float(p.price_open),
                "sl": float(p.sl), "tp": float(p.tp), "profit": float(p.profit),
                "swap": float(p.swap), "opened": int(p.time),
            })
        return rows
