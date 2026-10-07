"""Strategy 5 — EMA Trend Rider (unified).

Bullish only when EMA20 > EMA50 > EMA200 on H1 (reverse for bearish); enter on
a rejection candle when price pulls back to the M15 EMA20 with RSI 40-60.
SL beyond EMA50; TP at the previous swing high/low.
"""

from __future__ import annotations

from typing import Optional

from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    ema,
    rsi,
)


def last_swing_high(df, lookback: int = 5) -> float:
    """Highest high of the previous N candles excluding the most recent ones."""
    tail = df["high"].iloc[-(lookback + 2):-1]
    return float(tail.max()) if len(tail) else float(df["high"].iloc[-1])


def last_swing_low(df, lookback: int = 5) -> float:
    """Lowest low of the previous N candles excluding the most recent ones."""
    tail = df["low"].iloc[-(lookback + 2):-1]
    return float(tail.min()) if len(tail) else float(df["low"].iloc[-1])


class EMATrendRiderStrategy(BaseStrategy):
    """Ride H1 trends via M15 pullbacks to the EMA20."""

    name = "ema_trend_rider"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (3.0, 40.0)
    base_confluences = 4

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Enter on M15 rejection at EMA20 inside a clean H1 EMA stack."""
        trend = ctx.h1_trend()
        if trend not in ("bullish", "bearish"):
            return None

        c = ctx.m15["close"]
        e20_series = ema(c, 20)
        e50_series = ema(c, 50)
        e20 = float(e20_series.iloc[-1])
        e50 = float(e50_series.iloc[-1])
        rsi_v = float(rsi(c, 14).iloc[-1])
        price = ctx.price()
        atr_v = ctx.atr_m15()
        last = ctx.m15.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        rng = max(float(last["high"]) - float(last["low"]), 1e-12)

        if not (40.0 <= rsi_v <= 60.0):
            return None  # RSI extended -> not a healthy pullback

        touched = abs(price - e20) <= max(rng, (e50 - e20) * 0.15)
        if not touched:
            return None

        if trend == "bullish":
            rejection = float(last["close"]) > float(last["open"]) and body / rng >= 0.4
            if not rejection:
                return None
            # SL below price, ALWAYS: in a deep M15 pullback the M15 EMA50 can
            # sit ABOVE price, which produced GBPJPY #5 (buy, SL 21.9 pips
            # above entry, broker "stopped out" in profit). ATR-buffered
            # swing stop is correct-sided by construction.
            sl = min(last_swing_low(ctx.m15), e50) - 2.0 * atr_v
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="buy",
                entry=price, sl=sl, tp=last_swing_high(ctx.m15),
                session=self._session(ctx),
                confluences=[f"h1_{trend}_ema_stack", "m15_pullback_to_ema20",
                             f"rsi_{rsi_v:.1f}_neutral", "bullish_rejection_candle"],
            )

        rejection = float(last["close"]) < float(last["open"]) and body / rng >= 0.4
        if not rejection:
            return None
        # mirror image of the buy stop: ABOVE price, always
        sl = max(last_swing_high(ctx.m15), e50) + 2.0 * atr_v
        return StrategySignal(
            strategy=self.name, pair=ctx.pair, direction="sell",
            entry=price, sl=sl, tp=last_swing_low(ctx.m15),
            session=self._session(ctx),
            confluences=[f"h1_{trend}_ema_stack", "m15_pullback_to_ema20",
                         f"rsi_{rsi_v:.1f}_neutral", "bearish_rejection_candle"],
        )

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
