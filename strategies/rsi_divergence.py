"""Strategy 7 — RSI Divergence (unified).

Regular divergence between price and RSI on M15 (bullish: price LL, RSI HL;
bearish: price HH, RSI LH), taken only at key S/R levels or OB zones for
confluence. SL beyond the divergence swing; TP at the next structure level.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    pips_of,
    rsi,
)


def _swings(df: pd.DataFrame, lookback: int = 3) -> list[tuple[int, float, str]]:
    """List of (index, price, kind) swing points."""
    out: list[tuple[int, float, str]] = []
    highs, lows = df["high"].values, df["low"].values
    for i in range(lookback, len(df) - lookback):
        if highs[i] == highs[i - lookback: i + lookback + 1].max():
            out.append((i, float(highs[i]), "high"))
        if lows[i] == lows[i - lookback: i + lookback + 1].min():
            out.append((i, float(lows[i]), "low"))
    return out


class RSIDivergenceStrategy(BaseStrategy):
    """Trade regular RSI divergence at key levels on M15."""

    name = "rsi_divergence"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (3.0, 40.0)
    base_confluences = 4

    def _near_key_level(self, ctx: MarketContext, price: float) -> Optional[str]:
        """True when price is near a liquidity level or a recent OB zone."""
        for lv in ctx.liquidity_levels:
            if abs(price - lv) <= pips_of(ctx.pair, 10):
                return f"liquidity_level_{lv:.5f}"
        # simple OB proxy: last opposite-color candle zone
        last = ctx.m15.iloc[-2] if len(ctx.m15) >= 2 else ctx.m15.iloc[-1]
        zone_high = max(float(last["open"]), float(last["close"]))
        zone_low = min(float(last["open"]), float(last["close"]))
        if zone_low <= price <= zone_high:
            return "recent_candle_zone"
        return None

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Detect the most recent regular divergence with level confluence."""
        df = ctx.m15
        if len(df) < 60:
            return None
        r = rsi(df["close"], 14)
        swings = _swings(df, lookback=3)
        lows = [s for s in swings if s[2] == "low"][-2:]
        highs = [s for s in swings if s[2] == "high"][-2:]
        price = ctx.price()

        # --- regular bullish: price lower low, RSI higher low ---
        if len(lows) == 2:
            (i1, p1, _), (i2, p2, _) = lows
            if p2 < p1 and r.iloc[i2] > r.iloc[i1] + 2:
                key = self._near_key_level(ctx, price)
                if key:
                    structure_levels = sorted(
                        [s[1] for s in swings if s[2] == "high" and s[1] > price])
                    tp = structure_levels[0] if structure_levels else price + (p1 - p2) * 2
                    return StrategySignal(
                        strategy=self.name, pair=ctx.pair, direction="buy",
                        entry=price, sl=p2 - pips_of(ctx.pair, 3), tp=tp,
                        session=self._session(ctx),
                        confluences=[f"regular_bullish_divergence_rsi_{r.iloc[i2]:.0f}",
                                     f"key_level_{key}"],
                    )

        # --- regular bearish: price higher high, RSI lower high ---
        if len(highs) == 2:
            (i1, p1, _), (i2, p2, _) = highs
            if p2 > p1 and r.iloc[i2] < r.iloc[i1] - 2:
                key = self._near_key_level(ctx, price)
                if key:
                    structure_levels = sorted(
                        ([s[1] for s in swings if s[2] == "low" and s[1] < price]),
                        reverse=True)
                    tp = structure_levels[0] if structure_levels else price - (p2 - p1) * 2
                    return StrategySignal(
                        strategy=self.name, pair=ctx.pair, direction="sell",
                        entry=price, sl=p2 + pips_of(ctx.pair, 3), tp=tp,
                        session=self._session(ctx),
                        confluences=[f"regular_bearish_divergence_rsi_{r.iloc[i2]:.0f}",
                                     f"key_level_{key}"],
                    )
        return None

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
