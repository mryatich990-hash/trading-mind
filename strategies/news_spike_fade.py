"""Strategy 9 — News Spike Fade (unified).

After a high-impact release spikes price 30+ pips in under 2 minutes, wait for
two consecutive M1 stall candles and fade the move. SL 15 pips beyond the spike
wick; TP 50% retracement of the spike. Valid 5 minutes post-release only, and
skipped when the spread is above 5 pips (broker widening).
"""

from __future__ import annotations

from typing import Optional

from strategies.base_strategy import (
    MarketContext,
    StrategySignal,
    BaseStrategy,
    pips_of,
)


class NewsSpikeFadeStrategy(BaseStrategy):
    """Fade overextended news spikes once momentum stalls."""

    name = "news_spike_fade"
    sessions = ("London", "NewYork", "Overlap", "Asian")
    atr_range_pips = (1.0, 100.0)  # volatility is the point here
    base_confluences = 3

    def evaluate(self, ctx: MarketContext) -> Optional[StrategySignal]:
        """Fade a stalled news spike within the 5-minute validity window."""
        spike = ctx.news_spike
        if not spike:
            return None
        minutes_since = spike.get("minutes_since", 999)
        if not (0 <= minutes_since <= 5):
            return None
        if spike.get("size_pips", 0) < 30.0:
            return None
        if ctx.spread_pips > 5.0:
            self.logger.info("News spike skipped: spread %.1f > 5 pips", ctx.spread_pips)
            return None

        m1 = ctx.m1
        if len(m1) < 3:
            return None
        # stall: 2 consecutive candles with small bodies relative to the spike
        stall_bars = m1.tail(2)
        max_stall_body = pips_of(ctx.pair, 2.0)
        stalled = all(
            abs(float(r["close"]) - float(r["open"])) <= max_stall_body
            for _, r in stall_bars.iterrows()
        )
        if not stalled:
            return None

        spike_dir = spike.get("direction", "up")
        price = ctx.price()
        spike_high = spike.get("spike_high", price)
        spike_low = spike.get("spike_low", price)

        if spike_dir == "up":
            entry = price
            sl = spike_high + pips_of(ctx.pair, 15)
            tp = price - 0.5 * (spike_high - spike_low)  # 50% retrace
            direction = "sell"
            confluences = [f"spike_{spike.get('size_pips', 0):.0f}p_up",
                           "m1_double_stall", "within_5min_window", "spread_acceptable"]
        else:
            entry = price
            sl = spike_low - pips_of(ctx.pair, 15)
            tp = price + 0.5 * (spike_high - spike_low)
            direction = "buy"
            confluences = [f"spike_{spike.get('size_pips', 0):.0f}p_down",
                           "m1_double_stall", "within_5min_window", "spread_acceptable"]

        return StrategySignal(
            strategy=self.name, pair=ctx.pair, direction=direction,
            entry=entry, sl=sl, tp=tp, session=self._session(ctx),
            confluences=confluences,
        )

    @staticmethod
    def _session(ctx: MarketContext) -> str:
        """Session name."""
        from strategies.strategy_selector import session_of
        return session_of(ctx.now)
