"""Wyckoff analyzer: accumulation / distribution phase and spring / upthrust events."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["WyckoffState", "WyckoffAnalyzer"]


@dataclass
class WyckoffState:
    """Wyckoff read for one frame."""

    phase: str  # accumulation / distribution / markup / markdown / ranging
    event: str  # spring / upthrust / selling_climax / buying_climax / ""
    event_price: float = 0.0
    range_high: float = 0.0
    range_low: float = 0.0


class WyckoffAnalyzer:
    """Detects Wyckoff events from range structure and volume behaviour."""

    def __init__(self, range_lookback: int = 60, volume_decline_ratio: float = 0.85) -> None:
        self.range_lookback = range_lookback
        self.volume_decline_ratio = volume_decline_ratio

    def analyze(self, df: pd.DataFrame) -> WyckoffState:
        """Detect a recent spring (bullish) or upthrust (bearish)."""
        if len(df) < self.range_lookback + 5:
            return WyckoffState(phase="ranging")
        window = df.tail(self.range_lookback)
        vol = window["volume"]
        avg_vol = float(vol.mean())
        recent_vol = float(vol.tail(3).mean())
        declining = recent_vol < avg_vol * self.volume_decline_ratio

        body = window.iloc[:-3]
        range_high = float(body["high"].max())
        range_low = float(body["low"].min())
        rng = max(range_high - range_low, 1e-9)
        # range-bound market: last closes stay inside 80% of the range
        inside = window["close"].tail(5).between(range_low + rng * 0.1, range_high - rng * 0.1)
        ranging = bool(inside.mean() >= 0.6)

        spring = upthrust = False
        last = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        prev = df.iloc[-3] if len(df) >= 3 else last
        # spring: wick below range low, close back inside, above-average volume
        if (float(last["low"]) < range_low and float(last["close"]) > range_low
                and float(last["volume"]) > avg_vol and ranging):
            spring = True
        # upthrust: wick above range high, close back inside
        if (float(last["high"]) > range_high and float(last["close"]) < range_high
                and float(last["volume"]) > avg_vol and ranging):
            upthrust = True

        if spring:
            state = WyckoffState("accumulation", "spring", float(last["low"]), range_high, range_low)
        elif upthrust:
            state = WyckoffState("distribution", "upthrust", float(last["high"]), range_high, range_low)
        elif declining and ranging:
            state = WyckoffState("ranging", "", range_high, range_low)
        else:
            # trend direction from the window's net move
            change = float(window["close"].iloc[-1] - window["close"].iloc[0])
            state = WyckoffState("markup" if change > 0 else "markdown", "",
                                 range_high=range_high, range_low=range_low)
        logger.debug("wyckoff: %s %s", state.phase, state.event)
        return state
