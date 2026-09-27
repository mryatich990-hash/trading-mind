"""Strategy 2 — New York Reversal (unified).

Identify the London-session extreme; at NY open (13:00 UTC) watch for a sweep
of that level (liquidity grab), then enter the reversal after a confirmation
candle. Valid only in the first 60 minutes of NY.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    pips_of,
)


class NYReversalStrategy(BaseStrategy):
    """Fade London-high/low liquidity grabs at the NY open."""

    name = "ny_reversal"
    sessions = ("NewYork", "Overlap")
    atr_range_pips = (4.0, 35.0)
    base_confluences = 3

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Enter on London-extreme sweep + confirmation candle, first 60 min of NY."""
        mins = ctx.minutes_since(13)  # NY open 13:00 UTC
        if not (0 <= mins <= 60):
            return None

        ldn_hi, ldn_lo = ctx.london_range()
        if ldn_hi <= ldn_lo:
            return None

        today = ctx.now.normalize()
        ny = ctx.m15[ctx.m15["time"] >= today + pd.Timedelta(hours=13)]
        if ny.empty:
            return None

        price = ctx.price()
        last = ny.iloc[-1]
        prev = ny.iloc[-2] if len(ny) >= 2 else None
        session = "Overlap" if ctx.now.hour == 13 else "NewYork"

        # --- sweep of London high -> sell reversal ---
        swept_high = bool((ny["high"] > ldn_hi).any())
        if swept_high:
            sweep_wick = float(ny[ny["high"] > ldn_hi]["high"].max())
            if prev is not None and float(last["close"]) < ldn_hi < float(last["high"]):
                london_range = ldn_hi - ldn_lo
                return StrategySignal(
                    strategy=self.name, pair=ctx.pair, direction="sell",
                    entry=price, sl=sweep_wick + pips_of(ctx.pair, 8),
                    tp=price - 0.5 * london_range, session=session,
                    confluences=["london_high_swept", "reversal_confirmation_candle",
                                 "ny_open_window"],
                )

        # --- sweep of London low -> buy reversal ---
        swept_low = bool((ny["low"] < ldn_lo).any())
        if swept_low:
            sweep_wick = float(ny[ny["low"] < ldn_lo]["low"].min())
            if prev is not None and float(last["close"]) > ldn_lo > float(last["low"]):
                london_range = ldn_hi - ldn_lo
                return StrategySignal(
                    strategy=self.name, pair=ctx.pair, direction="buy",
                    entry=price, sl=sweep_wick - pips_of(ctx.pair, 8),
                    tp=price + 0.5 * london_range, session=session,
                    confluences=["london_low_swept", "reversal_confirmation_candle",
                                 "ny_open_window"],
                )
        return None
