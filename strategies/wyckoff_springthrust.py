"""Strategy 13 — Wyckoff Spring / Upthrust.

Trade the Wyckoff events: spring (stop hunt below accumulation range closing
back inside, above-average volume) -> buy; upthrust -> sell. SL beyond the
event wick; TP at the opposite range boundary.
"""

from typing import Optional

from strategies.base_strategy import BaseStrategy, MarketContext, StrategySignal


class WyckoffSpringThrustStrategy(BaseStrategy):
    """Trade spring and upthrust events with volume confirmation."""

    name = "wyckoff_springthrust"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (3.0, 40.0)

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Enter on a fresh spring or upthrust event from the Wyckoff analyzer."""
        if not ctx.wyckoff_event:
            return None
        price = ctx.price()
        if ctx.wyckoff_event == "spring" and ctx.wyckoff_range_low > 0:
            entry = price
            sl = ctx.wyckoff_range_low - (ctx.wyckoff_range_low - min(
                float(ctx.m15['low'].iloc[-2]), ctx.wyckoff_range_low)) - 2e-9
            sl = min(sl, float(ctx.m15["low"].iloc[-2])) if len(ctx.m15) >= 2 else sl
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="buy",
                entry=entry, sl=float(ctx.m15["low"].iloc[-2]) - 1e-9
                if len(ctx.m15) >= 2 else entry * 0.999,
                tp=ctx.wyckoff_range_high, session=self._session(ctx),
                confluences=["wyckoff_spring", f"range_low_{ctx.wyckoff_range_low:.5f}",
                             "volume_confirmation"],
            )
        if ctx.wyckoff_event == "upthrust" and ctx.wyckoff_range_high > 0:
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="sell",
                entry=price,
                sl=float(ctx.m15["high"].iloc[-2]) + 1e-9 if len(ctx.m15) >= 2 else price * 1.001,
                tp=ctx.wyckoff_range_low, session=self._session(ctx),
                confluences=["wyckoff_upthrust", f"range_high_{ctx.wyckoff_range_high:.5f}",
                             "volume_confirmation"],
            )
        return None

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
