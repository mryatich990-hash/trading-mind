"""Statistical arbitrage (UPGRADE 5): correlation divergence + triangular.

Watches pre-defined FX relationships; when correlation diverges below its
normal band, one leg is expected to mean-revert within 1-4 hours. Also
monitors the EURUSD x USDJPY ≈ EURJPY synthetic for 3+ pip dislocations.
Hard rails: max 1 position, 0.5% per leg, 4h time limit, 10-pip stop on
spread widening.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

PAIR_SETS = [("EURUSD", "GBPUSD", 0.85, 0.90),   # (a, b, enter_below, exit_above)
             ("USDJPY", "USDCHF", 0.80, 0.88)]


@dataclass
class StatArbPosition:
    """One open stat-arb position."""

    kind: str                 # correlation | triangular
    legs: list = field(default_factory=list)
    opened_at: float = 0.0
    entry_gap: float = 0.0    # pips
    max_gap: float = 0.0


class StatArbEngine:
    """Divergence + triangular monitors with conservative execution."""

    def __init__(self, data_engine=None, notifier=None) -> None:
        self.data = data_engine
        self.notifier = notifier
        self.enabled = settings.STAT_ARB_ENABLED
        self.position: Optional[StatArbPosition] = None
        self.history: list[dict] = []
        self.correlation_cache: dict[tuple, float] = {}
        self._last_corr_poll = 0.0

    # ---- analytics ----

    def correlation(self, a: str, b: str, days: int = 20) -> Optional[float]:
        """Return correlation of daily closes for a pair (cached 15 min)."""
        key = (a, b)
        if time.time() - self._last_corr_poll < 900 and key in self.correlation_cache:
            return self.correlation_cache[key]
        if self.data is None:
            return None
        try:
            closes = {}
            for sym in (a, b):
                df = self.data.get_candles(sym, 1440, days + 5).df
                closes[sym] = df["close"].astype(float).reset_index(drop=True)
            n = min(len(closes[a]), len(closes[b]))
            corr = float(np.corrcoef(closes[a].tail(n), closes[b].tail(n))[0, 1])
            self.correlation_cache[key] = round(corr, 3)
            self._last_corr_poll = time.time()
            return self.correlation_cache[key]
        except Exception as exc:
            logger.warning("correlation %s/%s failed: %s", a, b, exc)
            return None

    def relative_move(self, a: str, b: str, hours: int = 4) -> Optional[float]:
        """A's % move minus B's % move over the last hours (divergence size)."""
        if self.data is None:
            return None
        try:
            moves = []
            for sym in (a, b):
                df = self.data.get_candles(sym, 60, hours + 5).df
                recent = float(df["close"].iloc[-1])
                base = float(df["close"].iloc[0])
                moves.append(recent / base - 1.0)
            return (moves[0] - moves[1]) * 100.0
        except Exception as exc:
            logger.warning("relative move failed: %s", exc)
            return None

    def synthetic_gap_pips(self, base: str = "EURJPY",
                           leg1: str = "EURUSD", leg2: str = "USDJPY") -> Optional[float]:
        """|EURJPY - EURUSD*USDJPY| in pips."""
        if self.data is None:
            return None
        try:
            prices = {}
            for sym in (base, leg1, leg2):
                prices[sym] = float(self.data.get_candles(sym, 15, 5).df["close"].iloc[-1])
            synthetic = prices[leg1] * prices[leg2]
            pip = 0.01 if "JPY" in base else 0.0001
            return round(abs(prices[base] - synthetic) / pip, 1)
        except Exception as exc:
            logger.debug("synthetic gap failed: %s", exc)
            return None

    # ---- opportunity scan ----

    def scan(self) -> list[dict]:
        """Scan for divergence opportunities (requires no open position)."""
        if not self.enabled or self.position is not None:
            return []
        opportunities = []
        for a, b, enter_below, exit_above in PAIR_SETS:
            corr = self.correlation(a, b)
            if corr is None or corr >= enter_below:
                continue
            gap = self.relative_move(a, b)
            if gap is None or abs(gap) < 0.05:
                continue
            # buy the laggard: if A outperformed, buy B
            laggard = b if gap > 0 else a
            opportunities.append({
                "kind": "correlation", "pair": laggard,
                "detail": f"{a}/{b} corr {corr:.2f} < {enter_below}, gap {gap:+.2f}%",
                "gap_pct": round(gap, 2), "exit_corr": exit_above})
        gap3 = self.synthetic_gap_pips()
        if gap3 is not None and gap3 >= 3.0:
            opportunities.append({"kind": "triangular", "pair": "EURJPY",
                                  "detail": f"synthetic gap {gap3} pips", "gap_pips": gap3})
        return opportunities

    def open_position(self, opp: dict) -> bool:
        """Record a new stat-arb position (execution delegated to caller)."""
        if self.position is not None or not self.enabled:
            return False
        self.position = StatArbPosition(
            kind=opp["kind"], legs=[opp["pair"]], opened_at=time.time(),
            entry_gap=opp.get("gap_pips", 0.0) or abs(opp.get("gap_pct", 0.0)) * 10,
            max_gap=opp.get("gap_pips", 10.0) or 10.0)
        if self.notifier:
            self.notifier.send(f"⚖️ StatArb opened: {opp['kind']} {opp['pair']} — {opp['detail']}")
        return True

    def manage(self) -> list[str]:
        """Time-stop and spread-widening stop; returns closed reasons."""
        if self.position is None:
            return []
        closed = []
        pos = self.position
        held_min = (time.time() - pos.opened_at) / 60.0
        if held_min >= settings.STATARB_MAX_HOLD_MIN:
            closed.append(f"time limit {held_min:.0f} min")
            self.position = None
        if pos.kind == "triangular":
            gap = self.synthetic_gap_pips()
            if gap is not None and gap > pos.entry_gap + 10.0:
                closed.append(f"spread widened to {gap} pips (-10 pip stop)")
                self.position = None
        for reason in closed:
            self.history.append({"closed": reason, "held_min": round(held_min, 1),
                                 "ts": time.time()})
        return closed

    # ---- dashboard ----

    def status(self) -> dict:
        """Dashboard payload."""
        corrs = {}
        for a, b, _, _ in PAIR_SETS:
            c = self.correlation(a, b)
            if c is not None:
                corrs[f"{a}/{b}"] = c
        return {"enabled": self.enabled,
                "position": None if self.position is None else {
                    "kind": self.position.kind, "legs": self.position.legs,
                    "held_min": round((time.time() - self.position.opened_at) / 60, 1)},
                "correlations": corrs,
                "synthetic_gap_pips": self.synthetic_gap_pips(),
                "history": self.history[-10:]}
