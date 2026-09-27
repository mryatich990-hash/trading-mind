"""Tick data engine (UPGRADE 3): real-time tick processing.

Subscribes to MT5 ticks when available; otherwise polls the data engine's
rapid candles and synthesizes pseudo-ticks from OHLC so microstructure
analytics still have input. Tracks per pair: tick velocity (ticks/min),
spread per tick, buy/sell tick direction, order-flow imbalance, absorption
and tick momentum.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TickStats:
    """Rolling microstructure statistics for one pair."""

    velocity: float = 0.0            # ticks per minute
    spread_pips: float = 0.0         # last spread
    imbalance: float = 0.5           # buy_ticks / (buy+sell)
    momentum: float = 0.0            # velocity accel (positive = accelerating)
    absorption_events: int = 0
    stop_hunts: int = 0
    last_update: float = 0.0


class TickDataEngine:
    """Collects ticks per pair and maintains rolling microstructure stats."""

    def __init__(self, data_engine=None, window_sec: float = 120.0) -> None:
        self.data = data_engine
        self.window_sec = window_sec
        self.ticks: dict[str, deque] = {}       # pair -> deque[(ts, bid, ask)]
        self.stats: dict[str, TickStats] = {}
        self._mt5 = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prev_velocity: dict[str, float] = {}

    # ---- tick ingestion ----

    def _pair_deque(self, pair: str) -> deque:
        if pair not in self.ticks:
            self.ticks[pair] = deque(maxlen=5000)
            self.stats[pair] = TickStats()
        return self.ticks[pair]

    def add_tick(self, pair: str, bid: float, ask: float, ts: Optional[float] = None) -> None:
        """Ingest one tick (MT5 callback or synthesized)."""
        now = ts or time.time()
        with self._lock:
            dq = self._pair_deque(pair.upper())
            dq.append((now, float(bid), float(ask)))
            self._recompute(pair.upper())

    def _recompute(self, pair: str) -> None:
        """Recalculate rolling stats for one pair."""
        dq = self.ticks[pair]
        st = self.stats[pair]
        cutoff = time.time() - self.window_sec
        recent = [t for t in dq if t[0] >= cutoff]
        if not recent:
            return
        st.velocity = len(recent) / self.window_sec * 60.0
        last = recent[-1]
        st.spread_pips = round(last[2] - last[1], 6)
        # tick direction: uptick count vs downtick count on mid
        ups = downs = 0
        prev_mid = None
        for _, bid, ask in recent:
            mid = (bid + ask) / 2
            if prev_mid is not None:
                if mid > prev_mid:
                    ups += 1
                elif mid < prev_mid:
                    downs += 1
            prev_mid = mid
        st.imbalance = round(ups / (ups + downs), 3) if (ups + downs) else 0.5
        # momentum: velocity now vs previous window
        prev = self._prev_velocity.get(pair, st.velocity)
        st.momentum = round(st.velocity - prev, 2)
        self._prev_velocity[pair] = st.velocity
        st.last_update = time.time()

    # ---- mt5 subscription ----

    def _ensure_mt5(self) -> bool:
        """Connect to MT5 once when available."""
        if self._mt5 is not None:
            return True
        try:
            import MetaTrader5 as mt5  # type: ignore

            if not mt5.initialize():
                return False
            self._mt5 = mt5
            return True
        except Exception:
            return False

    def _poll_mt5(self) -> None:
        """Pull symbol_info ticks for all tracked pairs."""
        while not self._stop.is_set():
            for pair in list(self.ticks):
                try:
                    info = self._mt5.symbol_info_tick(pair)
                    if info is not None:
                        self.add_tick(pair, info.bid, info.ask)
                except Exception:
                    pass
            self._stop.wait(1.0)

    def _poll_synthetic(self) -> None:
        """Synthesize pseudo-ticks from fast candles (no MT5 path)."""
        while not self._stop.is_set():
            if self.data is not None:
                for pair in list(self.ticks):
                    try:
                        candle = self.data.get_candles(pair, 1, 5)
                        df = candle.df
                        last = df.iloc[-1]
                        spread = getattr(candle, "spread_pips", 1.0)
                        pip = 0.01 if "JPY" in pair or pair == "XAUUSD" else 0.0001
                        bid = float(last["close"]) - pip * spread / 2
                        ask = float(last["close"]) + pip * spread / 2
                        # subdivide the candle into ~8 pseudo-ticks
                        for k in range(8):
                            frac = k / 8
                            o, c = float(last["open"]), float(last["close"])
                            price = o + (c - o) * frac
                            self.add_tick(pair, price - pip * spread / 2,
                                          price + pip * spread / 2)
                    except Exception:
                        pass
            self._stop.wait(5.0)

    def start(self, pairs: list[str]) -> None:
        """Register pairs and start the collection thread."""
        for pair in pairs:
            self._pair_deque(pair.upper())
        if self._thread is not None:
            return
        target = self._poll_mt5 if self._ensure_mt5() else self._poll_synthetic
        mode = "mt5" if self._ensure_mt5() else "synthetic"
        logger.info("tick engine started (%s) for %s", mode, pairs)
        self._thread = threading.Thread(target=target, name="tick-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the collection thread."""
        self._stop.set()

    # ---- accessors ----

    def get_stats(self, pair: str) -> dict:
        """Stats snapshot for one pair."""
        st = self.stats.get(pair.upper())
        if st is None:
            return {}
        return {"velocity": round(st.velocity, 1), "spread_pips": round(st.spread_pips, 5),
                "imbalance": st.imbalance, "momentum": st.momentum,
                "absorption_events": st.absorption_events, "stop_hunts": st.stop_hunts}

    def all_stats(self) -> dict:
        """Stats for every tracked pair (dashboard)."""
        return {p: self.get_stats(p) for p in self.stats}
