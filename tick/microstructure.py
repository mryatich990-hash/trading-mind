"""Microstructure analysis (UPGRADE 3): stop hunts, icebergs, imbalance.

Works off the TickDataEngine's per-pair tick history:
- stop hunt: spike beyond a short-term level that fully reverses within 30s
- iceberg/absorption: many ticks at one price level without net movement
- order-flow imbalance ratio passed through to the Groq data package
"""

from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

STOP_HUNT_WINDOW_SEC = 30.0
STOP_HUNT_MIN_SPIKE_PIPS = 1.0
ABSORB_MIN_TICKS = 20
ABSORB_MAX_DRIFT_PIPS = 0.8


class MicrostructureAnalyzer:
    """Event detection over tick streams."""

    def __init__(self, tick_engine=None, pip_size_fn=None) -> None:
        self.tick_engine = tick_engine
        self.pip_size_fn = pip_size_fn or (lambda p: 0.0001 if "JPY" not in p else 0.01)
        self.events: dict[str, deque] = {}  # pair -> deque[dict]
        self.today_counts: dict[str, int] = {"stop_hunt": 0, "absorption": 0}

    def _events(self, pair: str) -> deque:
        if pair not in self.events:
            self.events[pair] = deque(maxlen=200)
        return self.events[pair]

    def detect(self, pair: str) -> list[dict]:
        """Run all detectors for one pair; returns new events."""
        pair = pair.upper()
        if self.tick_engine is None or pair not in self.tick_engine.ticks:
            return []
        dq = list(self.tick_engine.ticks[pair])
        if len(dq) < 30:
            return []
        pip = self.pip_size_fn(pair)
        events: list[dict] = []

        # ---- stop hunt: spike + full reversal inside 30s ----
        window = [t for t in dq if time.time() - t[0] <= STOP_HUNT_WINDOW_SEC]
        if len(window) >= 10:
            highs = [(t[0], t[2]) for t in window]
            lows = [(t[0], t[1]) for t in window]
            peak_t, peak = max(highs, key=lambda x: x[1])
            trough_t, trough = min(lows, key=lambda x: x[1])
            first, last = window[0], window[-1]
            pre_mid = (first[1] + first[2]) / 2
            last_mid = (last[1] + last[2]) / 2
            # upside hunt: spike high then back near pre-level
            if (peak - pre_mid) / pip >= STOP_HUNT_MIN_SPIKE_PIPS and \
               abs(last_mid - pre_mid) / pip <= 0.5 and \
               1.0 <= time.time() - peak_t <= STOP_HUNT_WINDOW_SEC:
                events.append({"type": "stop_hunt", "side": "bearish_reversal",
                               "pair": pair, "ts": peak_t, "detail": "spike high reverted"})
                self.tick_engine.stats[pair].stop_hunts += 1
            # downside hunt: spike low then back
            if (pre_mid - trough) / pip >= STOP_HUNT_MIN_SPIKE_PIPS and \
               abs(last_mid - pre_mid) / pip <= 0.5 and \
               1.0 <= time.time() - trough_t <= STOP_HUNT_WINDOW_SEC:
                events.append({"type": "stop_hunt", "side": "bullish_reversal",
                               "pair": pair, "ts": trough_t, "detail": "spike low reverted"})
                self.tick_engine.stats[pair].stop_hunts += 1

        # ---- absorption: many ticks, tiny net movement ----
        if len(window) >= ABSORB_MIN_TICKS:
            first_mid = (window[0][1] + window[0][2]) / 2
            last_mid = (window[-1][1] + window[-1][2]) / 2
            drift = abs(last_mid - first_mid) / pip
            if drift <= ABSORB_MAX_DRIFT_PIPS and self.tick_engine.stats[pair].velocity > 30:
                events.append({"type": "absorption", "pair": pair, "ts": time.time(),
                               "detail": f"{len(window)} ticks, drift {drift:.1f} pips"})
                self.tick_engine.stats[pair].absorption_events += 1

        for ev in events:
            self.today_counts[ev["type"]] = self.today_counts.get(ev["type"], 0) + 1
            self._events(pair).append(ev)
        return events

    def imbalance_ratio(self, pair: str) -> Optional[float]:
        """buy_ticks / (buy+sell); None when no data."""
        st = self.tick_engine.stats.get(pair.upper()) if self.tick_engine else None
        return None if st is None else st.imbalance

    def recent_events(self, pair: str = "", limit: int = 20) -> list[dict]:
        """Event feed for the dashboard."""
        if pair:
            return list(self._events(pair.upper()))[-limit:]
        merged = [ev for dq in self.events.values() for ev in dq]
        return sorted(merged, key=lambda e: e["ts"])[-limit:]

    def today_summary(self) -> dict:
        """Counts for the dashboard panel."""
        return dict(self.today_counts)
