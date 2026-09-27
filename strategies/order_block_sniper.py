"""Strategy 3 — Order Block Sniper (unified).

H1 order blocks (last opposing candle before a major move); entry when price
returns to the OB on M15 with RSI < 35 (bullish) / > 65 (bearish) and volume
above 120% of average (filter skipped when the feed has no volume, e.g. FX).
SL 5 pips beyond the OB; TP at the next liquidity level.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    atr,
    pips_of,
    rsi,
)


def find_ob(df: pd.DataFrame, direction: str, impulse_atr_mult: float = 2.0) -> Optional[dict]:
    """Find the most recent order block on a frame.

    A bullish OB is the last bearish candle before a bullish impulse whose move
    exceeds ``impulse_atr_mult`` * ATR within 3 candles.
    """
    a = float(atr(df, 14).iloc[-1]) if len(df) > 20 else 0.0
    if a <= 0:
        return None
    n = len(df)
    for i in range(n - 4, 2, -1):
        o, c = float(df["open"].iloc[i]), float(df["close"].iloc[i])
        impulse = float(df["close"].iloc[min(i + 3, n - 1)]) - c
        if direction == "bullish":
            if c < o and impulse > impulse_atr_mult * a:
                return {"high": float(df["high"].iloc[i]), "low": float(df["low"].iloc[i]),
                        "time": df["time"].iloc[i]}
        else:
            if c > o and impulse < -impulse_atr_mult * a:
                return {"high": float(df["high"].iloc[i]), "low": float(df["low"].iloc[i]),
                        "time": df["time"].iloc[i]}
    return None


class OrderBlockSniperStrategy(BaseStrategy):
    """Snipe retests of fresh H1 order blocks with RSI + volume confirmation."""

    name = "order_block_sniper"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (3.0, 40.0)
    base_confluences = 4

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Return a signal when price is inside a fresh H1 OB with filters met."""
        ob = find_ob(ctx.h1, "bullish")
        bearish_ob = find_ob(ctx.h1, "bearish")
        price = ctx.price()
        rsi_v = float(rsi(ctx.m15["close"], 14).iloc[-1])
        vol = ctx.m15["volume"]
        vol_avg = float(vol.rolling(20).mean().iloc[-1])
        vol_last = float(vol.iloc[-1])
        if vol_avg > 0 and vol_last > 0:
            vol_pct = vol_last / vol_avg * 100.0
        else:
            vol_pct = -1.0  # volume unavailable on this feed -> filter is neutral
        vol_tag = f"volume_{vol_pct:.0f}pct" if vol_pct >= 0 else "volume_n/a"

        # --- bullish OB ---
        if ob and ob["low"] <= price <= ob["high"]:
            if rsi_v >= 35.0:
                self.logger.debug("bullish OB touched but RSI %.1f >= 35", rsi_v)
            elif 0.0 <= vol_pct < 120.0:
                self.logger.debug("bullish OB touched but volume %.0f%% < 120%%", vol_pct)
            else:
                levels_above = sorted(lv for lv in ctx.liquidity_levels if lv > price)
                tp = levels_above[0] if levels_above else price + 2 * (price - ob["low"])
                return StrategySignal(
                    strategy=self.name, pair=ctx.pair, direction="buy",
                    entry=price, sl=ob["low"] - pips_of(ctx.pair, 5), tp=tp,
                    session=self._session(ctx),
                    confluences=["h1_bullish_ob", "m15_touch",
                                 f"rsi_{rsi_v:.1f}_oversold", vol_tag],
                )

        # --- bearish OB ---
        if bearish_ob and bearish_ob["low"] <= price <= bearish_ob["high"]:
            if rsi_v <= 65.0:
                self.logger.debug("bearish OB touched but RSI %.1f <= 65", rsi_v)
            elif 0.0 <= vol_pct < 120.0:
                self.logger.debug("bearish OB touched but volume %.0f%% < 120%%", vol_pct)
            else:
                levels_below = sorted((lv for lv in ctx.liquidity_levels if lv < price),
                                      reverse=True)
                tp = levels_below[0] if levels_below else price - 2 * (bearish_ob["high"] - price)
                return StrategySignal(
                    strategy=self.name, pair=ctx.pair, direction="sell",
                    entry=price, sl=bearish_ob["high"] + pips_of(ctx.pair, 5), tp=tp,
                    session=self._session(ctx),
                    confluences=["h1_bearish_ob", "m15_touch",
                                 f"rsi_{rsi_v:.1f}_overbought", vol_tag],
                )
        return None

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
