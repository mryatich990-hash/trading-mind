"""Harmonic pattern detector: Gartley / Bat / Butterfly / Crab with PRZ zones."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["HarmonicPattern", "HarmonicDetector"]


@dataclass
class HarmonicPattern:
    """A harmonic pattern with its Potential Reversal Zone."""

    name: str
    direction: str  # buy (bullish PRZ) / sell
    prz_high: float
    prz_low: float
    x: float
    a: float
    d: float
    confidence: int


class HarmonicDetector:
    """Finds harmonic patterns from ZigZag pivots using Fib ratio windows."""

    # (name, AB retrace of XA, CD extension of XA range) — tolerance ±6%
    PATTERNS: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
        ("gartley", (0.618, 0.618), (0.618, 0.786)),
        ("bat", (0.382, 0.50), (0.886, 0.886)),
        ("butterfly", (0.786, 0.786), (1.27, 1.618)),
        ("crab", (0.382, 0.618), (1.618, 1.618)),
    ]

    def __init__(self, zigzag_pct: float = 0.2, tolerance: float = 0.06) -> None:
        self.zigzag_pct = zigzag_pct
        self.tolerance = tolerance

    def _swings(self, df: pd.DataFrame) -> list[dict]:
        """ZigZag pivots."""
        from structure.smart_money_detector import SmartMoneyDetector

        return SmartMoneyDetector(zigzag_pct=self.zigzag_pct).zigzag(df)

    def detect(self, df: pd.DataFrame) -> Optional[HarmonicPattern]:
        """Scan the latest 5-pivot XABCD sequence for a harmonic pattern."""
        swings = self._swings(df)
        if len(swings) < 5:
            return None
        seq = swings[-5:]
        kinds = [s["kind"] for s in seq]
        # bullish pattern: X low, A high, B low, C high, D low
        if kinds == ["low", "high", "low", "high", "low"]:
            xa = seq[1]["price"] - seq[0]["price"]
            ab = seq[1]["price"] - seq[2]["price"]
            cd = seq[3]["price"] - seq[4]["price"]
            direction = "buy"
            extreme = seq[4]["price"]
        # bearish: X high, A low, B high, C low, D high
        elif kinds == ["high", "low", "high", "low", "high"]:
            xa = seq[0]["price"] - seq[1]["price"]
            ab = seq[2]["price"] - seq[1]["price"]
            cd = seq[4]["price"] - seq[3]["price"]
            direction = "sell"
            extreme = seq[4]["price"]
        else:
            return None
        if xa <= 0 or ab <= 0 or cd <= 0:
            return None

        ab_ratio = ab / xa
        cd_ratio = cd / xa
        for name, (ab_lo, ab_hi), (cd_lo, cd_hi) in self.PATTERNS:
            if not (ab_lo * (1 - self.tolerance) <= ab_ratio <= ab_hi * (1 + self.tolerance)):
                continue
            if not (cd_lo * (1 - self.tolerance) <= cd_ratio <= cd_hi * (1 + self.tolerance)):
                continue
            # PRZ around the D extreme: ±0.618 of the CD leg
            prz_span = cd * 0.618
            if direction == "buy":
                prz_high, prz_low = extreme + prz_span * 0.3, extreme - prz_span * 0.7
            else:
                prz_high, prz_low = extreme + prz_span * 0.7, extreme - prz_span * 0.3
            return HarmonicPattern(
                name=name, direction=direction,
                prz_high=round(prz_high, 6), prz_low=round(prz_low, 6),
                x=seq[0]["price"], a=seq[1]["price"], d=extreme,
                confidence=60 if name in ("gartley", "bat") else 55,
            )
        return None
