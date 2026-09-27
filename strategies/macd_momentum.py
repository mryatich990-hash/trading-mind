"""Strategy 8 — MACD Momentum (unified).

H1 MACD cross (line over/under signal) with an expanding histogram; drop to M15
and enter on the pullback to EMA20 while H1 MACD stays positive/negative.
SL beyond the last M15 swing; TP minimum 1:2 RR.
"""

from __future__ import annotations

from typing import Optional

from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    ema,
    macd,
    pips_of,
)


class MACDMomentumStrategy(BaseStrategy):
    """Trade H1 MACD crosses with an M15 EMA20 pullback entry."""

    name = "macd_momentum"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (4.0, 40.0)
    base_confluences = 4

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Enter M15 pullback while H1 MACD momentum is fresh and expanding."""
        h1 = ctx.h1
        if len(h1) < 40:
            return None
        m = macd(h1["close"])
        line, sig, hist = m["macd"], m["signal"], m["hist"]
        crossed_up = line.iloc[-2] <= sig.iloc[-2] and line.iloc[-1] > sig.iloc[-1]
        crossed_down = line.iloc[-2] >= sig.iloc[-2] and line.iloc[-1] < sig.iloc[-1]
        hist_expanding = abs(hist.iloc[-1]) > abs(hist.iloc[-2])
        if not (crossed_up or crossed_down) or not hist_expanding:
            return None

        c = ctx.m15["close"]
        e20 = float(ema(c, 20).iloc[-1])
        price = ctx.price()
        last = ctx.m15.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        rng = max(float(last["high"]) - float(last["low"]), 1e-12)
        if abs(price - e20) > max(rng, 3 * pips_of(ctx.pair, 1)):
            return None  # not at the pullback zone

        swing_low = float(ctx.m15["low"].tail(6).min())
        swing_high = float(ctx.m15["high"].tail(6).max())

        if crossed_up and line.iloc[-1] > 0 and body / rng >= 0.35:
            entry, sl = price, swing_low - pips_of(ctx.pair, 2)
            risk = entry - sl
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="buy",
                entry=entry, sl=sl, tp=entry + 2 * risk,  # 1:2 minimum
                session=self._session(ctx),
                confluences=["h1_macd_bullish_cross", "histogram_expanding",
                             "m15_ema20_pullback", "rejection_candle"],
            )

        if crossed_down and line.iloc[-1] < 0 and body / rng >= 0.35:
            entry, sl = price, swing_high + pips_of(ctx.pair, 2)
            risk = sl - entry
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="sell",
                entry=entry, sl=sl, tp=entry - 2 * risk,
                session=self._session(ctx),
                confluences=["h1_macd_bearish_cross", "histogram_expanding",
                             "m15_ema20_pullback", "rejection_candle"],
            )
        return None

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
