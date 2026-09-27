"""Strategy 10 — Asian Range Fade (unified).

During a quiet Asian session (ATR below 50% of the daily average) fade moves to
the range boundaries: sell the top with RSI > 65, buy the bottom with RSI < 35.
SL 8 pips beyond the boundary; TP at the opposite boundary. Invalidated as soon
as price breaks the range by more than 10 pips.
"""

from __future__ import annotations

from typing import Optional

from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    pips_of,
    pip_size,
    rsi,
)


class AsianRangeFadeStrategy(BaseStrategy):
    """Fade Asian range extremes in quiet conditions."""

    name = "asian_range_fade"
    sessions = ("Asian",)
    atr_range_pips = (0.5, 12.0)
    base_confluences = 3

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Fade the Asian range boundaries when the market is quiet."""
        hi, lo, candles = ctx.asian_range()
        if candles < 8:
            return None
        range_pips = (hi - lo) / pip_size(ctx.pair)
        if range_pips < 10.0:
            return None  # too tight to fade
        price = ctx.price()

        # invalidation: price has broken the range by more than 10 pips already
        if price > hi + pips_of(ctx.pair, 10):
            return None
        if price < lo - pips_of(ctx.pair, 10):
            return None

        # volatility filter: ATR below 50% of the daily average
        atr_v = ctx.atr_m15()
        daily_avg = ctx.daily_atr_avg
        if daily_avg <= 0:
            # fallback: 20-period average of H4 ranges approximated to daily
            daily_avg = float((ctx.h4["high"] - ctx.h4["low"]).tail(20).mean()) / 4.0
        if daily_avg > 0 and atr_v > 0.5 * daily_avg:
            self.logger.debug("ATR too high for range fade: %.5f vs avg %.5f",
                              atr_v, daily_avg)
            return None

        rsi_v = float(rsi(ctx.m15["close"], 14).iloc[-1])

        # sell the top
        if price >= hi - pips_of(ctx.pair, 2) and rsi_v > 65:
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="sell",
                entry=price, sl=hi + pips_of(ctx.pair, 8), tp=lo,
                session="Asian",
                confluences=[f"asian_range_{range_pips:.0f}p", "at_boundary_top",
                             f"rsi_{rsi_v:.0f}_overbought", "quiet_atr"],
            )

        # buy the bottom
        if price <= lo + pips_of(ctx.pair, 2) and rsi_v < 35:
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="buy",
                entry=price, sl=lo - pips_of(ctx.pair, 8), tp=hi,
                session="Asian",
                confluences=[f"asian_range_{range_pips:.0f}p", "at_boundary_bottom",
                             f"rsi_{rsi_v:.0f}_oversold", "quiet_atr"],
            )
        return None
