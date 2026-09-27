"""VWAP engine: daily, weekly and anchored VWAP with sigma bands."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["VWAPSnapshot", "VWAPEngine"]


@dataclass
class VWAPSnapshot:
    """VWAP values and band positions."""

    daily: float
    weekly: float
    anchored: float
    anchored_swing: str  # "high" | "low"
    sigma1_upper: float
    sigma1_lower: float
    sigma2_upper: float
    sigma2_lower: float
    sigma3_upper: float
    sigma3_lower: float
    price: float
    position_vs_daily: str  # above / below


class VWAPEngine:
    """Computes VWAP variants with 1/2/3 sigma bands."""

    @staticmethod
    def _vwap(df: pd.DataFrame) -> float:
        """Typical-price VWAP of a frame."""
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].replace(0, np.nan)
        return float((typical * vol).sum() / vol.sum()) if vol.notna().any() else float(df["close"].iloc[-1])

    @staticmethod
    def _sigma(df: pd.DataFrame, vwap: float) -> float:
        """Volume-weighted standard deviation around VWAP."""
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].replace(0, 1.0)
        variance = (vol * (typical - vwap) ** 2).sum() / vol.sum()
        return float(np.sqrt(max(variance, 0.0)))

    def compute(self, m15: pd.DataFrame, now: pd.Timestamp) -> VWAPSnapshot:
        """Build the snapshot from an M15 frame covering several days."""
        df = m15.copy()
        ts = pd.to_datetime(df["time"], utc=True)
        price = float(df["close"].iloc[-1])

        day_start = now.normalize()
        weekly_start = day_start - pd.Timedelta(days=now.dayofweek)
        day_df = df[ts >= day_start]
        week_df = df[ts >= weekly_start]

        daily = self._vwap(day_df) if len(day_df) else price
        weekly = self._vwap(week_df) if len(week_df) else price

        # anchored VWAP: anchor to the last significant swing (5% range extreme)
        lookback = df.tail(96)
        swing_high_t = lookback.loc[lookback["high"].idxmax(), "time"]
        swing_low_t = lookback.loc[lookback["low"].idxmin(), "time"]
        if swing_high_t > swing_low_t:
            anchor_t, swing = swing_high_t, "high"
        else:
            anchor_t, swing = swing_low_t, "low"
        anchored_df = df[ts >= anchor_t]
        anchored = self._vwap(anchored_df) if len(anchored_df) else daily

        sigma = self._sigma(day_df, daily) if len(day_df) > 5 else 0.0
        return VWAPSnapshot(
            daily=round(daily, 6),
            weekly=round(weekly, 6),
            anchored=round(anchored, 6),
            anchored_swing=swing,
            sigma1_upper=round(daily + sigma, 6),
            sigma1_lower=round(daily - sigma, 6),
            sigma2_upper=round(daily + 2 * sigma, 6),
            sigma2_lower=round(daily - 2 * sigma, 6),
            sigma3_upper=round(daily + 3 * sigma, 6),
            sigma3_lower=round(daily - 3 * sigma, 6),
            price=round(price, 6),
            position_vs_daily="above" if price > daily else "below",
        )
