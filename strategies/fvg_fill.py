"""Strategy 4 — Fair Value Gap Fill (unified).

Detect 3-candle FVGs on M15 formed during high-momentum moves; trade fills in
the direction of the H1 trend. SL beyond the FVG + 5 pip buffer; TP at the
opposing end of the FVG + 10 pips. FVGs older than 8 hours are invalidated.
"""

from __future__ import annotations

from typing import Optional

from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    pips_of,
)


class FVGFillStrategy(BaseStrategy):
    """Trade M15 FVG fills aligned with the H1 trend."""

    name = "fvg_fill"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (3.0, 35.0)
    base_confluences = 4

    @staticmethod
    def detect_fvgs(ctx: MarketContext, momentum_atr_mult: float = 1.5) -> list[dict]:
        """Find recent M15 FVGs born from momentum candles, newest first."""
        df = ctx.m15
        atr_v = float(df["close"].rolling(14).std().iloc[-1]) if len(df) > 20 else 0.0
        gaps: list[dict] = []
        n = len(df)
        for i in range(n - 1, 2, -1):
            t0 = df["time"].iloc[i - 2]
            c1_high, c1_low = float(df["high"].iloc[i - 2]), float(df["low"].iloc[i - 2])
            c3_high, c3_low = float(df["high"].iloc[i]), float(df["low"].iloc[i])
            mid_body = abs(float(df["close"].iloc[i - 1]) - float(df["open"].iloc[i - 1]))
            age_hours = (ctx.now - t0).total_seconds() / 3600.0
            if age_hours > 8.0:
                break  # newest-first: everything older is invalid
            momentum = mid_body > momentum_atr_mult * atr_v if atr_v > 0 else False
            if c3_low > c1_high and momentum:
                gaps.append({"kind": "bullish_fvg", "high": c3_low, "low": c1_high,
                             "time": t0, "age_hours": age_hours})
            elif c3_high < c1_low and momentum:
                gaps.append({"kind": "bearish_fvg", "high": c1_low, "low": c3_high,
                             "time": t0, "age_hours": age_hours})
            if len(gaps) >= 5:
                break
        return gaps

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Enter when price trades into a fresh momentum FVG aligned with H1 trend."""
        trend = ctx.h1_trend()
        if trend not in ("bullish", "bearish"):
            return None
        price = ctx.price()
        for gap in self.detect_fvgs(ctx):
            if not (gap["low"] <= price <= gap["high"]):
                continue
            if trend == "bullish" and gap["kind"] == "bullish_fvg":
                entry = price
                sl = gap["low"] - pips_of(ctx.pair, 5)
                tp = gap["high"] + pips_of(ctx.pair, 10)
                direction = "buy"
            elif trend == "bearish" and gap["kind"] == "bearish_fvg":
                entry = price
                sl = gap["high"] + pips_of(ctx.pair, 5)
                tp = gap["low"] - pips_of(ctx.pair, 10)
                direction = "sell"
            else:
                continue  # counter-trend gap -> skip
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction=direction,
                entry=entry, sl=sl, tp=tp, session=self._session(ctx),
                confluences=[f"m15_{gap['kind']}_momentum", f"h1_trend_{trend}",
                             f"gap_age_{gap['age_hours']:.1f}h", "price_entering_gap"],
            )
        return None

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
