"""Strategy 12 — Volume Profile POC Magnet.

When price moves through a Low Volume Node with the previous day's VPOC acting
as a magnet, enter pullback trades toward VPOC. SL beyond the opposing HVN;
TP at VPOC.
"""

from typing import Optional

from strategies.base_strategy import (
    BaseStrategy, MarketContext, StrategySignal, pip_size, rsi,
)


class VPOCMagnetStrategy(BaseStrategy):
    """Trade toward the previous day's VPOC through LVNs."""

    name = "vpoc_magnet"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (3.0, 45.0)

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Enter when price is between LVNs with VPOC as the nearest magnet."""
        if ctx.vpoc_prev <= 0:
            return None
        price = ctx.price()
        distance_pips = abs(price - ctx.vpoc_prev) / pip_size(ctx.pair)
        if distance_pips < 10 or distance_pips > 80:
            return None  # too close (already there) or too far (magnet stale)

        rsi_v = float(rsi(ctx.m15["close"], 14).iloc[-1])
        last = ctx.m15.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        rng = max(float(last["high"]) - float(last["low"]), 1e-12)

        # VPOC above -> magnet up -> buy pullbacks
        if ctx.vpoc_prev > price and rsi_v < 55 and body / rng >= 0.3:
            sl_candidates = [h for h in ctx.hvn_levels if h < price]
            sl = (min(sl_candidates) if sl_candidates else price - distance_pips * pip_size(ctx.pair) * 0.6)
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="buy",
                entry=price, sl=sl, tp=ctx.vpoc_prev, session=self._session(ctx),
                confluences=[f"vpoc_{ctx.vpoc_prev:.5f}_above", f"distance_{distance_pips:.0f}p",
                             f"rsi_{rsi_v:.0f}_supportive", "lvn_transit"],
            )
        # VPOC below -> magnet down -> sell rallies
        if ctx.vpoc_prev < price and rsi_v > 45 and body / rng >= 0.3:
            sl_candidates = [h for h in ctx.hvn_levels if h > price]
            sl = (max(sl_candidates) if sl_candidates else price + distance_pips * pip_size(ctx.pair) * 0.6)
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="sell",
                entry=price, sl=sl, tp=ctx.vpoc_prev, session=self._session(ctx),
                confluences=[f"vpoc_{ctx.vpoc_prev:.5f}_below", f"distance_{distance_pips:.0f}p",
                             f"rsi_{rsi_v:.0f}_supportive", "lvn_transit"],
            )
        return None

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
