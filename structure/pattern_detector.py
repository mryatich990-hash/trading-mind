"""Classical pattern detector: double top/bottom, H&S, wedges, flags, triangles."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["Pattern", "PatternDetector"]


@dataclass
class Pattern:
    """A detected classical pattern."""

    name: str
    direction: str  # buy / sell
    target: float
    invalidation: float
    confidence: int


class PatternDetector:
    """Detects classical chart patterns with measured-move targets."""

    def __init__(self, tolerance_pct: float = 0.08) -> None:
        self.tolerance_pct = tolerance_pct

    def _pivots(self, df: pd.DataFrame, lookback: int = 3) -> list[tuple[int, float, str]]:
        """Swing pivots."""
        out = []
        highs, lows = df["high"].values, df["low"].values
        for i in range(lookback, len(df) - lookback):
            if highs[i] == highs[i - lookback: i + lookback + 1].max():
                out.append((i, float(highs[i]), "high"))
            if lows[i] == lows[i - lookback: i + lookback + 1].min():
                out.append((i, float(lows[i]), "low"))
        return out

    def detect(self, df: pd.DataFrame) -> Optional[Pattern]:
        """Return the strongest pattern present, or None."""
        if len(df) < 60:
            return None
        pivots = self._pivots(df)
        highs = [p for p in pivots if p[2] == "high"][-3:]
        lows = [p for p in pivots if p[2] == "low"][-3:]
        price = float(df["close"].iloc[-1])
        tol = price * self.tolerance_pct / 100.0

        # double top: two equal highs, confirmed by break below the valley
        if len(highs) >= 2:
            h1, h2 = highs[-2][1], highs[-1][1]
            if abs(h2 - h1) <= tol:
                valley = min(l for i, l, k in pivots if highs[-2][0] < i < highs[-1][0] and k == "low") \
                    if any(highs[-2][0] < i < highs[-1][0] and k == "low" for i, l, k in pivots) else None
                if valley and price < valley:
                    target = valley - (h1 - valley)
                    return Pattern("double_top", "sell", round(target, 6), round(h2 + tol, 6), 65)
                if valley:
                    return Pattern("double_top_forming", "sell", round(valley - (h1 - valley), 6),
                                   round(h2 + tol, 6), 45)

        # double bottom
        if len(lows) >= 2:
            l1, l2 = lows[-2][1], lows[-1][1]
            if abs(l2 - l1) <= tol:
                peak = max(h for i, h, k in pivots if lows[-2][0] < i < lows[-1][0] and k == "high") \
                    if any(lows[-2][0] < i < lows[-1][0] and k == "high" for i, h, k in pivots) else None
                if peak and price > peak:
                    target = peak + (peak - l1)
                    return Pattern("double_bottom", "buy", round(target, 6), round(l1 - tol, 6), 65)
                if peak:
                    return Pattern("double_bottom_forming", "buy", round(peak + (peak - l1), 6),
                                   round(l1 - tol, 6), 45)

        # head and shoulders: high, higher high, lower high
        if len(highs) == 3:
            ls, head, rs = highs[0][1], highs[1][1], highs[2][1]
            if head > ls and head > rs and abs(rs - ls) <= tol * 1.5:
                neck = min((l for i, l, k in pivots if highs[0][0] < i < highs[2][0] and k == "low"),
                           default=None)
                if neck and price < neck:
                    target = neck - (head - neck)
                    return Pattern("head_and_shoulders", "sell", round(target, 6), round(head, 6), 70)
                if neck:
                    return Pattern("hns_forming", "sell", round(neck - (head - neck), 6),
                                   round(head, 6), 50)

        # inverse head and shoulders
        if len(lows) == 3:
            ls, head, rs = lows[0][1], lows[1][1], lows[2][1]
            if head < ls and head < rs and abs(rs - ls) <= tol * 1.5:
                neck = max((h for i, h, k in pivots if lows[0][0] < i < lows[2][0] and k == "high"),
                           default=None)
                if neck and price > neck:
                    target = neck + (neck - head)
                    return Pattern("inverse_hns", "buy", round(target, 6), round(head, 6), 70)
                if neck:
                    return Pattern("inv_hns_forming", "buy", round(neck + (neck - head), 6),
                                   round(head, 6), 50)

        # wedges: compare last 3 highs slope vs last 3 lows slope
        if len(highs) >= 3 and len(lows) >= 3:
            hs = [h[1] for h in highs[-3:]]
            ls_ = [l[1] for l in lows[-3:]]
            h_slope = hs[-1] - hs[0]
            l_slope = ls_[-1] - ls_[0]
            if h_slope > 0 and l_slope > 0 and l_slope > h_slope * 1.5:
                return Pattern("rising_wedge", "sell", round(price - (hs[-1] - ls_[0]), 6),
                               round(hs[-1] + tol, 6), 60)
            if h_slope < 0 and l_slope < 0 and h_slope < l_slope * 1.5:
                return Pattern("falling_wedge", "buy", round(price + (ls_[-1] - hs[0]), 6),
                               round(ls_[-1] - tol, 6), 60)
        return None
