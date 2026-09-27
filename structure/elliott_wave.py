"""Elliott Wave counter: identifies 5-wave impulse positions from ZigZag swings."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["WaveState", "ElliottWaveAnalyzer"]


@dataclass
class WaveState:
    """Current Elliott position."""

    wave: int  # 0 = no count
    phase: str  # impulse / correction / none
    guidance: str  # trading guidance text
    confidence: int = 0


class ElliottWaveAnalyzer:
    """Counts impulse waves using ZigZag pivots and Fibonacci rules."""

    def __init__(self, zigzag_pct: float = 0.2) -> None:
        self.zigzag_pct = zigzag_pct

    def _swings(self, df: pd.DataFrame) -> list[dict]:
        """ZigZag pivots."""
        from structure.smart_money_detector import SmartMoneyDetector

        return SmartMoneyDetector(zigzag_pct=self.zigzag_pct).zigzag(df)

    def analyze(self, df: pd.DataFrame) -> WaveState:
        """Locate price within a 5-wave impulse, applying classic ratios."""
        swings = self._swings(df)
        if len(swings) < 6:
            return WaveState(0, "none", "insufficient swings for a wave count")
        # take the last 7 alternating pivots as a candidate sequence
        seq = swings[-7:]
        # normalize so the sequence starts with the impulse origin
        if seq[0]["kind"] == "high":
            seq = seq[1:]
        if len(seq) < 6:
            return WaveState(0, "none", "no bullish-aligned sequence")

        p0, p1, p2, p3, p4, p5 = (s["price"] for s in seq[:6])
        wave1 = p1 - p0
        if wave1 <= 0:
            return WaveState(0, "none", "wave1 not positive")
        wave2_retrace = (p1 - p2) / wave1
        wave3 = p3 - p2
        wave4_retrace = (p3 - p4) / wave3 if wave3 > 0 else 0

        rules_ok = (0.38 <= wave2_retrace <= 0.786          # wave 2 depth sanity
                    and wave3 >= wave1                       # wave 3 never shortest vs w1
                    and p2 > p0                              # wave 2 above origin
                    and 0.2 <= wave4_retrace <= 0.7)         # wave 4 depth sanity
        if not rules_ok:
            return WaveState(0, "none", "ratios do not fit an impulse count")

        price = float(df["close"].iloc[-1])
        wave3_ext = p2 + 1.618 * wave1
        if price < wave3_ext * 0.98:
            return WaveState(3, "impulse",
                             f"wave 3 in progress (w1={wave1:.5f}, w3={wave3:.5f}); trend trade with large targets",
                             confidence=70)
        if price < p5 or p5 == p3:
            return WaveState(4, "impulse", "wave 4 territory; expect wave 5 push, lighter size", 55)
        return WaveState(5, "impulse",
                         "wave 5 / post-impulse: watch for reversal and ABC correction", 45)
