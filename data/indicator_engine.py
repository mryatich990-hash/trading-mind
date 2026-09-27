"""IndicatorEngine: all indicators computed in-house via pandas (pandas-ta when available).

Never passes indicator names to the AI without numeric values alongside.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    ta = None

from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["IndicatorSet", "IndicatorEngine", "atr", "rsi", "ema", "macd", "stochastic"]


def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR series."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]).abs(),
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def macd(close: pd.Series) -> pd.DataFrame:
    """MACD(12,26,9): macd, signal, hist columns."""
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": signal, "hist": line - signal})


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic (14,3,3): k and d columns."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["close"] - low_min) / denom
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k.fillna(50), "d": d.fillna(50)})


def delta_volume(df: pd.DataFrame) -> pd.Series:
    """Proxy cumulative delta: close>open candles add volume, else subtract."""
    direction = np.where(df["close"] >= df["open"], 1.0, -1.0)
    return pd.Series(direction * df["volume"].to_numpy(float), index=df.index)


@dataclass
class IndicatorSet:
    """Full numeric indicator snapshot for one timeframe."""

    timeframe: str
    ema9: float
    ema20: float
    ema50: float
    ema200: float
    atr14: float
    atr_avg20: float
    rsi14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    stoch_k: float
    stoch_d: float
    volume_vs_avg_pct: float
    delta_5: float
    close: float


class IndicatorEngine:
    """Computes IndicatorSet snapshots from validated candle frames."""

    def compute(self, df: pd.DataFrame, timeframe: str) -> IndicatorSet:
        """Calculate every indicator on the frame; returns latest values."""
        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
        m = macd(close)
        st = stochastic(df)
        atr_series = atr(df, 14)
        vol_avg = float(vol.rolling(20).mean().iloc[-1])
        return IndicatorSet(
            timeframe=timeframe,
            ema9=round(float(ema(close, 9).iloc[-1]), 6),
            ema20=round(float(ema(close, 20).iloc[-1]), 6),
            ema50=round(float(ema(close, 50).iloc[-1]), 6),
            ema200=round(float(ema(close, 200).iloc[-1]), 6),
            atr14=round(float(atr_series.iloc[-1]), 6),
            atr_avg20=round(float(atr_series.rolling(20).mean().iloc[-1]), 6),
            rsi14=round(float(rsi(close, 14).iloc[-1]), 2),
            macd_line=round(float(m["macd"].iloc[-1]), 6),
            macd_signal=round(float(m["signal"].iloc[-1]), 6),
            macd_hist=round(float(m["hist"].iloc[-1]), 6),
            macd_hist_prev=round(float(m["hist"].iloc[-2]), 6),
            bb_upper=round(float((ema(close, 20) + 2 * close.rolling(20).std(ddof=0)).iloc[-1]), 6),
            bb_middle=round(float(ema(close, 20).iloc[-1]), 6),
            bb_lower=round(float((ema(close, 20) - 2 * close.rolling(20).std(ddof=0)).iloc[-1]), 6),
            stoch_k=round(float(st["k"].iloc[-1]), 1),
            stoch_d=round(float(st["d"].iloc[-1]), 1),
            volume_vs_avg_pct=round((float(vol.iloc[-1]) / vol_avg * 100.0) if vol_avg > 0 else 0.0, 1),
            delta_5=round(float(delta_volume(df).tail(5).sum()), 1),
            close=round(float(close.iloc[-1]), 6),
        )
