"""Volume profile: POC / VAH / VAL / HVN / LVN with previous-day VPOC magnet."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["VolumeProfile", "VolumeProfileEngine"]


@dataclass
class VolumeProfile:
    """Session volume profile."""

    poc: float
    vah: float
    val: float
    hvn: list[float]
    lvn: list[float]
    bins: int
    price: float
    poc_side: str  # "above" (magnet up) / "below" (magnet down) / "at"


class VolumeProfileEngine:
    """Builds session volume profiles from M15 candles."""

    def __init__(self, bins: int = 24, hvn_count: int = 3) -> None:
        self.bins = bins
        self.hvn_count = hvn_count

    def compute(self, df: pd.DataFrame, pair: str,
                day: Optional[pd.Timestamp] = None) -> VolumeProfile:
        """Profile for a session day; falls back to the whole frame when empty."""
        df = df.copy()
        ts = pd.to_datetime(df["time"], utc=True)
        if day is not None:
            session = df[ts.dt.normalize() == day.normalize()]
            df = session if len(session) >= 8 else df

        price_low, price_high = float(df["low"].min()), float(df["high"].max())
        if price_high <= price_low:
            price = float(df["close"].iloc[-1])
            return VolumeProfile(price, price, price, [], [], self.bins, price, "at")

        typical = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy(float)
        vol = df["volume"].to_numpy(float)
        bin_edges = np.linspace(price_low, price_high, self.bins + 1)
        idx = np.clip(np.digitize(typical, bin_edges) - 1, 0, self.bins - 1)
        profile = np.zeros(self.bins)
        for i, v in zip(idx, vol):
            profile[i] += v

        total = profile.sum() or 1.0
        poc_bin = int(profile.argmax())
        poc = float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0)

        # value area: expand from POC until 70% of volume captured
        lo = hi = poc_bin
        captured = profile[poc_bin]
        while captured < total * 0.70 and (lo > 0 or hi < self.bins - 1):
            down = profile[lo - 1] if lo > 0 else -1.0
            up = profile[hi + 1] if hi < self.bins - 1 else -1.0
            if up >= down:
                hi += 1
                captured += max(up, 0)
            else:
                lo -= 1
                captured += max(down, 0)
        vah = float((bin_edges[hi] + bin_edges[hi + 1]) / 2.0)
        val = float((bin_edges[lo] + bin_edges[lo + 1]) / 2.0)

        order = np.argsort(profile)[::-1]
        hvn = [float((bin_edges[b] + bin_edges[b + 1]) / 2.0) for b in order[: self.hvn_count]]
        lvn_threshold = np.percentile(profile, 25)
        lvn_bins = [b for b in range(self.bins) if profile[b] <= lvn_threshold]
        lvn = [float((bin_edges[b] + bin_edges[b + 1]) / 2.0) for b in lvn_bins[: self.hvn_count]]

        price = float(df["close"].iloc[-1])
        side = "at" if abs(price - poc) < (price_high - price_low) / self.bins else (
            "above" if poc > price else "below"
        )
        return VolumeProfile(
            poc=round(poc, 6), vah=round(vah, 6), val=round(val, 6),
            hvn=[round(v, 6) for v in hvn], lvn=[round(v, 6) for v in lvn],
            bins=self.bins, price=round(price, 6), poc_side=side,
        )
