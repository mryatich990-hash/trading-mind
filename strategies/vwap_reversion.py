"""Strategy 11 — VWAP Reversion.

Fade extensions to the 2-sigma band around daily VWAP: RSI overbought/oversold
confirmation plus a rejection candle at the band. TP at VWAP, SL beyond 3-sigma.
"""

from typing import Optional

import pandas as pd

from strategies.base_strategy import (
    BaseStrategy, MarketContext, StrategySignal, rsi, pip_size,
)


class VWAPReversionStrategy(BaseStrategy):
    """Mean-reversion to daily VWAP from 2-sigma extensions."""

    name = "vwap_reversion"
    sessions = ("London", "NewYork", "Overlap")
    atr_range_pips = (3.0, 50.0)

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Enter at the 2-sigma band with RSI + rejection confirmation."""
        if ctx.vwap_daily <= 0 or ctx.vwap_sigma2_upper <= ctx.vwap_sigma2_lower:
            return None
        price = ctx.price()
        rsi_v = float(rsi(ctx.m15["close"], 14).iloc[-1])
        last = ctx.m15.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        rng = max(float(last["high"]) - float(last["low"]), 1e-12)
        rejection = body / rng >= 0.35

        # stretched above VWAP -> sell back to VWAP
        if price >= ctx.vwap_sigma2_upper and rsi_v > 70 and rejection:
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="sell",
                entry=price, sl=ctx.vwap_sigma3_upper, tp=ctx.vwap_daily,
                session=self._session(ctx),
                confluences=[f"price_{price:.5f}_above_2sigma", f"rsi_{rsi_v:.0f}_overbought",
                             "rejection_at_band"],
            )
        # stretched below VWAP -> buy back to VWAP
        if price <= ctx.vwap_sigma2_lower and rsi_v < 30 and rejection:
            return StrategySignal(
                strategy=self.name, pair=ctx.pair, direction="buy",
                entry=price, sl=ctx.vwap_sigma3_lower, tp=ctx.vwap_daily,
                session=self._session(ctx),
                confluences=[f"price_{price:.5f}_below_2sigma", f"rsi_{rsi_v:.0f}_oversold",
                             "rejection_at_band"],
            )
        return None

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
