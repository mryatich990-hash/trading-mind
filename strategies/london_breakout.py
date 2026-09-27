"""Strategy 1 — London Breakout (unified).

Asian range (00:00-07:00 UTC) is marked; at London open (07:00) we wait for a
break of that range, then enter on the retest of the broken level. Valid only
in the first 90 minutes of London; skipped when the Asian range is < 10 pips
(too tight) or > 60 pips (too volatile).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config import settings
from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    pips_of,
    pip_size,
)


class LondonBreakoutStrategy(BaseStrategy):
    """Trade the first break + retest of the Asian range at London open."""

    name = "london_breakout"
    sessions = ("London",)
    atr_range_pips = (3.0, 30.0)
    base_confluences = 3

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Detect an Asian-range break + retest inside the 90-minute window."""
        mins = ctx.minutes_since(7)  # since London open 07:00 UTC
        if not (0 <= mins <= 90):
            return None

        hi, lo, candles = ctx.asian_range()
        if candles == 0:
            return None
        range_pips = (hi - lo) / pip_size(ctx.pair)
        if not (10.0 <= range_pips <= 60.0):
            return None

        price = ctx.price()
        today = ctx.now.normalize()
        ldn = ctx.m15[ctx.m15["time"] >= today + pd.Timedelta(hours=7)]
        if ldn.empty:
            return None

        broke_up = bool((ldn["high"] > hi).any())
        broke_down = bool((ldn["low"] < lo).any())
        if broke_up == broke_down:  # no break, or both (chop) -> skip
            return None

        if broke_up:
            direction = "buy"
            level = hi
            post_break = ldn[ldn["high"] > hi]
            if post_break.empty or float(post_break["low"].iloc[-1]) > level + pips_of(ctx.pair, 2):
                return None
            sl = level - pips_of(ctx.pair, 10)  # 10 pips beyond Asian range
            tp = price + 1.5 * (hi - lo)
        else:
            direction = "sell"
            level = lo
            post_break = ldn[ldn["low"] < lo]
            if post_break.empty or float(post_break["high"].iloc[-1]) < level - pips_of(ctx.pair, 2):
                return None
            sl = level + pips_of(ctx.pair, 10)
            tp = price - 1.5 * (hi - lo)

        return StrategySignal(
            strategy=self.name, pair=ctx.pair, direction=direction,
            entry=price, sl=sl, tp=tp, session="London",
            confluences=["asian_range_defined", "range_break_confirmed",
                         "retest_of_level", "london_kill_zone"],
        )
