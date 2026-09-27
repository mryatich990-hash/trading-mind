"""BaseStrategy ABC + MarketContext: contract for all 15 strategies.

Strategies are pure functions of MarketContext -> Optional[StrategySignal];
all I/O lives in the research engine and execution layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.logging_utils import get_logger

__all__ = ["MarketContext", "StrategySignal", "BaseStrategy", "atr", "rsi", "ema", "macd",
           "pips_of", "pip_size"]


def pip_size(pair: str) -> float:
    """Pip size per instrument."""
    pair = pair.upper()
    if pair in ("USDJPY", "GBPJPY", "EURJPY"):
        return 0.01
    if pair == "XAUUSD":
        return 0.1
    if pair in ("NAS100", "US30"):
        return 1.0
    return 0.0001


def pips_of(pair: str, n: float) -> float:
    """n pips as price distance."""
    return n * pip_size(pair)


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
    """Wilder ATR."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(), (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def macd(close: pd.Series) -> pd.DataFrame:
    """MACD(12,26,9)."""
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": signal, "hist": line - signal})


@dataclass
class MarketContext:
    """Everything a strategy may inspect for one pair on one candle close."""

    pair: str
    now: pd.Timestamp
    m1: pd.DataFrame
    m15: pd.DataFrame
    h1: pd.DataFrame
    h4: pd.DataFrame
    daily: pd.DataFrame
    spread_pips: float = 0.0
    # injected research data (optional per strategy)
    liquidity_levels: list[float] = field(default_factory=list)
    vwap_daily: float = 0.0
    vwap_sigma2_upper: float = 0.0
    vwap_sigma2_lower: float = 0.0
    vwap_sigma3_upper: float = 0.0
    vwap_sigma3_lower: float = 0.0
    vpoc_prev: float = 0.0
    hvn_levels: list[float] = field(default_factory=list)
    wyckoff_event: str = ""
    wyckoff_range_high: float = 0.0
    wyckoff_range_low: float = 0.0
    harmonic: Optional[object] = None
    daily_atr_avg: float = 0.0
    news_spike: Optional[dict] = None
    cot_bias: str = "neutral"
    retail_long_pct: float = 50.0

    def price(self) -> float:
        """Latest M15 close."""
        return float(self.m15["close"].iloc[-1])

    def atr_m15(self) -> float:
        """Current M15 ATR."""
        return float(atr(self.m15).iloc[-1])

    def asian_range(self) -> tuple[float, float, int]:
        """(high, low, candles) of today's Asian session."""
        today = self.now.normalize()
        part = self.m15[(self.m15["time"] >= today)
                        & (self.m15["time"] < today + pd.Timedelta(hours=7))]
        if part.empty:
            return 0.0, 0.0, 0
        return float(part["high"].max()), float(part["low"].min()), len(part)

    def london_range(self) -> tuple[float, float]:
        """(high, low) of today's London session."""
        today = self.now.normalize()
        part = self.m15[(self.m15["time"] >= today + pd.Timedelta(hours=7))
                        & (self.m15["time"] < today + pd.Timedelta(hours=12))]
        if part.empty:
            return 0.0, 0.0
        return float(part["high"].max()), float(part["low"].min())

    def minutes_since(self, hour: int) -> float:
        """Minutes since HH:00 UTC today."""
        anchor = self.now.normalize() + pd.Timedelta(hours=hour)
        return (self.now - anchor).total_seconds() / 60.0

    def h1_trend(self) -> str:
        """H1 EMA stack trend label."""
        c = self.h1["close"]
        if len(c) < 200:
            return "ranging"
        if ema(c, 20).iloc[-1] > ema(c, 50).iloc[-1] > ema(c, 200).iloc[-1]:
            return "bullish"
        if ema(c, 20).iloc[-1] < ema(c, 50).iloc[-1] < ema(c, 200).iloc[-1]:
            return "bearish"
        return "ranging"

    def daily_trend(self) -> str:
        """Daily trend from EMA50 vs EMA200."""
        c = self.daily["close"]
        if len(c) < 200:
            return "range"
        return "up" if ema(c, 50).iloc[-1] > ema(c, 200).iloc[-1] else "down"

    def h4_bias(self) -> str:
        """H4 bias from EMA50 vs EMA200."""
        c = self.h4["close"]
        if len(c) < 200:
            return "ranging"
        return "bullish" if ema(c, 50).iloc[-1] > ema(c, 200).iloc[-1] else "bearish"


@dataclass
class StrategySignal:
    """A candidate signal returned by a strategy."""

    strategy: str
    pair: str
    direction: str  # buy / sell
    entry: float
    sl: float
    tp: float
    session: str
    confluences: list[str] = field(default_factory=list)

    @property
    def sl_pips(self) -> float:
        """Risk in pips."""
        return abs(self.entry - self.sl) / pip_size(self.pair)

    @property
    def rr(self) -> float:
        """Planned RR."""
        return round(abs(self.tp - self.entry) / max(abs(self.entry - self.sl), 1e-9), 2)


class BaseStrategy(ABC):
    """Contract: detect_setup, get_sl, get_tp plus selector scoring hooks."""

    name: str = "base"
    sessions: tuple[str, ...] = ("London", "NewYork", "Overlap")
    atr_range_pips: tuple[float, float] = (3.0, 40.0)

    def __init__(self) -> None:
        self.logger = get_logger(f"strategy.{self.name}")

    def detect_setup(self, ctx: MarketContext) -> Optional[dict]:
        """Return setup details (zone, direction, notes) or None. Default: evaluate()."""
        sig = self.evaluate(ctx)
        return {"direction": sig.direction} if sig else None

    def get_sl(self, ctx: MarketContext, setup: dict) -> float:
        """SL price from setup details."""
        return float(setup.get("sl", ctx.price()))

    def get_tp(self, ctx: MarketContext, setup: dict) -> float:
        """TP price from setup details."""
        return float(setup.get("tp", ctx.price()))

    @abstractmethod
    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Return a signal when the setup is valid."""

    def session_match(self, ctx: MarketContext) -> bool:
        """Active session check."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now) in self.sessions

    def volatility_match(self, ctx: MarketContext) -> bool:
        """ATR-in-band check."""
        atr_pips = ctx.atr_m15() / pip_size(ctx.pair)
        lo, hi = self.atr_range_pips
        return lo <= atr_pips <= hi
