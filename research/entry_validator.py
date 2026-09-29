"""EntryValidator: Research Step 5 (volume/order flow) + Step 6 (M15 entry checklist).

Step 6 requires a minimum of 8 of 10 points before a signal may proceed to
Groq; each item is a hard, numeric check so the result is fully auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import settings
from core.logging_utils import get_logger
from data.indicator_engine import IndicatorEngine
from data.market_data_engine import pip_size
from strategies.strategy_selector import session_of

logger = get_logger(__name__)

__all__ = ["EntryChecklist", "EntryValidator"]

MAJOR_MAX_SPREAD = 2.0
GOLD_MAX_SPREAD = 5.0


@dataclass
class EntryChecklist:
    """Step 5 + 6 results with the 10-item M15 checklist."""

    # step 5
    volume_pct: float = 100.0
    delta_5: float = 0.0
    delta_aligned: bool = False
    at_hvn: bool = False
    absorption: bool = False
    # step 6 items (True = point earned)
    at_zone: bool = False
    rsi_aligned: bool = False
    rejection_candle: bool = False
    volume_confirmed: bool = False
    macd_aligned: bool = False
    ema20_h1_aligned: bool = False
    session_ok: bool = False
    spread_ok: bool = False
    no_red_news: bool = False
    cot_not_against: bool = False
    score: int = 0
    required: int = settings.MIN_CONFLUENCE
    failed_items: list[str] = field(default_factory=list)
    rsi14: float = 50.0
    macd_hist: float = 0.0
    macd_hist_trend: str = "flat"
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    candle_pattern: str = "none"

    @property
    def passed(self) -> bool:
        """Checklist passed at the required threshold."""
        return self.score >= self.required


class EntryValidator:
    """Validates the M15 entry with the 10-point checklist."""

    def __init__(self) -> None:
        self.indicators = IndicatorEngine()

    def _rejection_candle(self, df: pd.DataFrame, direction: str) -> tuple[bool, str]:
        """Pin bar, engulfing or doji in the trade direction on the last candle."""
        last = df.iloc[-1]
        o, h, l, c = (float(last[k]) for k in ("open", "high", "low", "close"))
        rng = max(h - l, 1e-9)
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        if body / rng <= 0.12:
            return True, "doji"
        if direction == "buy":
            if lower_wick / rng >= 0.55:
                return True, "bullish_pin_bar"
            prev = df.iloc[-2]
            if c > o and float(prev["close"]) < float(prev["open"]) and c > float(prev["open"]) \
                    and o < float(prev["close"]):
                return True, "bullish_engulfing"
        else:
            if upper_wick / rng >= 0.55:
                return True, "bearish_pin_bar"
            prev = df.iloc[-2]
            if c < o and float(prev["close"]) > float(prev["open"]) and c < float(prev["open"]) \
                    and o > float(prev["close"]):
                return True, "bearish_engulfing"
        return False, "none"

    def validate(self, pair: str, direction: str, frames: dict[str, pd.DataFrame],
                 htf, macro, confluence_bonus: int = 0) -> EntryChecklist:
        """Score the 10-point checklist. htf: HTFResult, macro: MacroResult.

        When the data feed carries no volume (Yahoo FX candles are all-zero),
        the volume item is EXCLUDED from both score and requirement instead of
        failing permanently — otherwise FX trades are mathematically impossible
        (max score 7 < required 8).
        """
        chk = EntryChecklist()
        m15, h1 = frames["m15"], frames["h1"]
        price = float(m15["close"].iloc[-1])
        direction_buy = direction == "buy"
        m15_vol = pd.to_numeric(m15["volume"], errors="coerce").tail(50).fillna(0)
        m15_vol_sum = float(m15_vol.sum())
        # Sparse-volume feeds (Yahoo FX: only a few session bars carry any
        # volume, the rest are 0) slip past an all-zero check but poison the
        # ratio — the last bar is usually 0, so volume_pct reads 0 and the
        # check fails as phantom noise (it was the #1 or #2 failure in every
        # confluence bucket this week). Fewer than 20% of bars carrying any
        # volume is noise, not data: exclude the item like an all-zero feed.
        nonzero_bars = int((m15_vol > 0).sum())
        volume_usable = m15_vol_sum > 0.0 and \
            nonzero_bars >= max(5, int(0.2 * max(len(m15_vol), 1)))

        # step 5 volume metrics only make sense when the feed has volume
        try:
            ind_m15 = self.indicators.compute(m15, "M15")
            if volume_usable:
                chk.volume_pct = ind_m15.volume_vs_avg_pct
            chk.delta_5 = ind_m15.delta_5
            chk.delta_aligned = (chk.delta_5 > 0) if direction_buy else (chk.delta_5 < 0)
            chk.at_hvn = any(abs(price - hvn) < 2 * pip_size(pair)
                             for hvn in getattr(htf, "hvn_levels", []))
        except Exception as exc:
            logger.warning("volume metrics failed: %s", exc)

        # ---- step 6 checklist ----
        zone_price = htf.ob_high if direction_buy else htf.ob_low
        in_ob = (htf.ob_kind != "none"
                 and htf.ob_low <= price <= htf.ob_high)
        in_fvg = (htf.fvg_kind != "none"
                  and htf.fvg_low <= price <= htf.fvg_high)
        chk.at_zone = in_ob or in_fvg

        try:
            ind = self.indicators.compute(m15, "M15")
            chk.rsi14 = ind.rsi14
            chk.macd_hist = ind.macd_hist
            chk.macd_hist_trend = ("rising" if ind.macd_hist > ind.macd_hist_prev
                                   else "falling")
            chk.stoch_k, chk.stoch_d = ind.stoch_k, ind.stoch_d
            chk.rsi_aligned = chk.rsi14 < 55 if direction_buy else chk.rsi14 > 45
            chk.macd_aligned = chk.macd_hist > 0 if direction_buy else chk.macd_hist < 0
        except Exception as exc:
            logger.warning("indicators failed: %s", exc)

        ok, pattern = self._rejection_candle(m15, direction)
        chk.rejection_candle = ok
        chk.candle_pattern = pattern

        chk.volume_confirmed = volume_usable and chk.volume_pct >= 115.0

        try:
            e20_h1 = float(h1["close"].ewm(span=20, adjust=False).mean().iloc[-1])
            chk.ema20_h1_aligned = (price > e20_h1) if direction_buy else (price < e20_h1)
        except Exception as exc:
            logger.warning("h1 ema failed: %s", exc)

        session = session_of(pd.Timestamp(m15["time"].iloc[-1]))
        chk.session_ok = session in ("London", "NewYork", "Overlap")

        max_spread = GOLD_MAX_SPREAD if pair.upper() == "XAUUSD" else MAJOR_MAX_SPREAD
        chk.spread_ok = getattr(macro, "spread_pips", 0.0) <= max_spread if \
            hasattr(macro, "spread_pips") else True

        chk.no_red_news = not macro.blocked
        chk.cot_not_against = macro.cot_bias in ("neutral", direction)

        # confluence bonus items count toward the score when present
        bonus = min(max(confluence_bonus, 0), 2)

        checks = {
            "at_zone": chk.at_zone, "rsi_aligned": chk.rsi_aligned,
            "rejection_candle": chk.rejection_candle,
            "volume_confirmed": chk.volume_confirmed,
            "macd_aligned": chk.macd_aligned,
            "ema20_h1_aligned": chk.ema20_h1_aligned,
            "session_ok": chk.session_ok, "spread_ok": chk.spread_ok,
            "no_red_news": chk.no_red_news, "cot_not_against": chk.cot_not_against,
        }
        if not volume_usable:
            # feed has no volume data: drop the item from score AND denial
            # list instead of letting a dead check veto every FX trade
            checks.pop("volume_confirmed")
        chk.score = sum(1 for v in checks.values() if v) + bonus
        chk.failed_items = [k for k, v in checks.items() if not v]
        logger.info("entry checklist %s %s: %d/%d%s (failed: %s)", pair, direction,
                    chk.score, len(checks), "+bonus" if bonus else "",
                    ", ".join(chk.failed_items) or "none")
        return chk
