"""Strategy 6 — Liquidity Sweep and Reversal (unified).

Equal highs/lows get swept during a kill zone; enter the reversal when the
next candle closes back inside the range. SL 8 pips beyond the sweep wick;
TP at the opposite liquidity pool.
"""

from __future__ import annotations

from typing import Optional

from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    pips_of,
)

KILL_ZONE_HOURS = ((7, 12), (12, 17))  # London, NY (incl. overlap)


def equal_levels(df, tolerance: float, min_touches: int = 2) -> tuple[list[float], list[float]]:
    """(equal_highs, equal_lows) clustered within tolerance from recent candles."""
    window = df.tail(96)
    highs = [float(x) for x in window["high"].tail(48)]
    lows = [float(x) for x in window["low"].tail(48)]

    def cluster(points: list[float]) -> list[float]:
        points = sorted(points)
        pools: list[list[float]] = []
        for p in points:
            if pools and abs(p - pools[-1][-1]) <= tolerance:
                pools[-1].append(p)
            else:
                pools.append([p])
        return [sum(g) / len(g) for g in pools if len(g) >= min_touches]

    return cluster(highs), cluster(lows)


class LiquiditySweepStrategy(BaseStrategy):
    """Fade sweeps of equal highs/lows during London/NY kill zones."""

    name = "liquidity_sweep"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (3.0, 40.0)
    base_confluences = 4

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Enter reversal after a kill-zone sweep closes back inside the range."""
        hour = ctx.now.hour
        if not any(start <= hour < end for start, end in KILL_ZONE_HOURS):
            return None

        tol = pips_of(ctx.pair, 5)
        eq_highs, eq_lows = equal_levels(ctx.m15, tol)
        # merge external liquidity levels if provided
        for lv in ctx.liquidity_levels:
            eq_highs.append(lv)
            eq_lows.append(lv)

        price = ctx.price()
        last = ctx.m15.iloc[-1]
        prev = ctx.m15.iloc[-2] if len(ctx.m15) >= 2 else None
        if prev is None:
            return None

        # --- sweep above equal highs, close back inside -> sell ---
        for lvl in sorted(set(eq_highs), reverse=True):
            if float(last["high"]) > lvl >= float(prev["high"]) and float(last["close"]) < lvl:
                opposite = min((p for p in set(eq_lows) if p < price), default=None)
                tp = opposite if opposite else price - (float(last["high"]) - lvl) * 3
                return StrategySignal(
                    strategy=self.name, pair=ctx.pair, direction="sell",
                    entry=price, sl=float(last["high"]) + pips_of(ctx.pair, 8), tp=tp,
                    session=self._session(ctx),
                    confluences=[f"equal_highs_{lvl:.5f}_swept", "close_back_inside",
                                 "kill_zone_active"],
                )

        # --- sweep below equal lows, close back inside -> buy ---
        for lvl in sorted(set(eq_lows)):
            if float(last["low"]) < lvl <= float(prev["low"]) and float(last["close"]) > lvl:
                opposite = max((p for p in set(eq_highs) if p > price), default=None)
                tp = opposite if opposite else price + (lvl - float(last["low"])) * 3
                return StrategySignal(
                    strategy=self.name, pair=ctx.pair, direction="buy",
                    entry=price, sl=float(last["low"]) - pips_of(ctx.pair, 8), tp=tp,
                    session=self._session(ctx),
                    confluences=[f"equal_lows_{lvl:.5f}_swept", "close_back_inside",
                                 "kill_zone_active"],
                )
        return None

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
