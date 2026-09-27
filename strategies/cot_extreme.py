"""Strategy 15 — COT Extreme Reversal.

Swing setup: COT commercials at an extreme percentile, retail positioned the
other way, price at a major level with H1 confirmation. Wider SL, larger TP;
hold 2-5 days when in profit after day one.
"""

from typing import Optional

from strategies.base_strategy import BaseStrategy, MarketContext, StrategySignal, pip_size


class COTExtremeStrategy(BaseStrategy):
    """Position with commercials against the retail crowd at extremes."""

    name = "cot_extreme"
    sessions = ("London", "NewYork", "Overlap")  # entry timing, but swing hold
    atr_range_pips = (5.0, 60.0)

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Require COT extreme + retail contrarian + major level + H1 confirmation."""
        if ctx.cot_bias == "neutral":
            return None
        # retail must lean the opposite way (crowd on the wrong side)
        retail_confirms = (
            (ctx.cot_bias == "buy" and ctx.retail_long_pct <= 25.0)
            or (ctx.cot_bias == "sell" and ctx.retail_long_pct >= 75.0)
        )
        if not retail_confirms:
            return None
        price = ctx.price()
        # price at a major liquidity level (within 15 pips)
        near_level = any(abs(price - lv) <= 15 * pip_size(ctx.pair)
                         for lv in ctx.liquidity_levels)
        if not near_level:
            return None
        # H1 confirmation candle
        last = ctx.h1.iloc[-1]
        bullish_candle = float(last["close"]) > float(last["open"])
        if ctx.cot_bias == "buy" and not bullish_candle:
            return None
        if ctx.cot_bias == "sell" and bullish_candle:
            return None

        atr_v = ctx.atr_m15()
        if ctx.cot_bias == "buy":
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="buy",
                entry=price, sl=price - 2 * atr_v, tp=price + 5 * atr_v,
                session=self._session(ctx),
                confluences=[f"cot_bias_{ctx.cot_bias}", f"retail_long_{ctx.retail_long_pct:.0f}pct",
                             "major_level_proximity", "h1_confirmation", "swing_hold"],
            )
        return StrategySignal(
            strategy=self.name, pair=ctx.pair, direction="sell",
            entry=price, sl=price + 2 * atr_v, tp=price - 5 * atr_v,
            session=self._session(ctx),
            confluences=[f"cot_bias_{ctx.cot_bias}", f"retail_long_{ctx.retail_long_pct:.0f}pct",
                         "major_level_proximity", "h1_confirmation", "swing_hold"],
        )

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
