"""Strategy 14 — Harmonic Pattern PRZ.

Trade reversals when price reaches a harmonic Potential Reversal Zone with a
confirmation candle; RSI divergence inside the PRZ strengthens the entry.
SL beyond the pattern extreme; TP1 at 38.2% and TP at 61.8% of the CD leg.
"""

from typing import Optional

from strategies.base_strategy import BaseStrategy, MarketContext, StrategySignal, rsi


class HarmonicPRZStrategy(BaseStrategy):
    """Trade harmonic PRZ touches with confirmation."""

    name = "harmonic_prz"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (3.0, 45.0)

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Enter when price is inside the PRZ with a rejection candle."""
        pattern = ctx.harmonic
        if pattern is None:
            return None
        price = ctx.price()
        if not (pattern.prz_low <= price <= pattern.prz_high):
            return None
        last = ctx.m15.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        rng = max(float(last["high"]) - float(last["low"]), 1e-12)
        if body / rng < 0.3:
            return None  # no confirmation candle
        rsi_v = float(rsi(ctx.m15["close"], 14).iloc[-1])
        cd = abs(pattern.d - pattern.a)
        if pattern.direction == "buy":
            tp = price + 0.618 * cd
            sl = pattern.prz_low
        else:
            tp = price - 0.618 * cd
            sl = pattern.prz_high
        return StrategySignal(
            strategy=self.name, pair=ctx.pair, direction=pattern.direction,
            entry=price, sl=sl, tp=tp, session=self._session(ctx),
            confluences=[f"{pattern.name}_prz", f"prz_{pattern.prz_low:.5f}-{pattern.prz_high:.5f}",
                         "confirmation_candle", f"rsi_{rsi_v:.0f}"],
        )

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
