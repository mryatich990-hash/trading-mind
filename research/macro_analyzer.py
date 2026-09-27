"""MacroAnalyzer: Research Step 1 (macro context) + Step 2 (news gate).

Combines intermarket bias scores, COT positioning, retail sentiment and news
sentiment into per-currency macro bias scores (-100..+100) and enforces the
news gate that aborts research inside red-event windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core.logging_utils import get_logger
from institutional.central_bank_tracker import CentralBankTracker
from institutional.cot_reader import COTReader
from institutional.intermarket_analyzer import IntermarketAnalyzer, IntermarketSnapshot
from institutional.retail_sentiment import RetailSentimentReader
from news.calendar_engine import CalendarEngine
from news.sentiment_engine import SentimentEngine

logger = get_logger(__name__)

__all__ = ["MacroResult", "MacroAnalyzer"]

COT_CONFLUENCE_BONUS = 15
RETAIL_CONFLUENCE_BONUS = 10


@dataclass
class MacroResult:
    """Step 1 + 2 output."""

    usd_bias: float = 0.0
    base_bias: float = 0.0
    quote_bias: float = 0.0
    base_currency: str = ""
    quote_currency: str = ""
    pair_bias: float = 0.0
    dxy_trend: str = "neutral"
    vix: float = 0.0
    vix_term: str = "normal"
    yield_10y: float = 0.0
    yield_trend: str = "flat"
    cot_bias: str = "neutral"          # buy / sell / neutral
    cot_aligned: bool = False
    cot_commercial_net: float = 0.0
    cot_percentile: float = 50.0
    retail_long_pct: float = 50.0
    retail_contrarian: str = "neutral"  # buy / sell / neutral
    news_gate: str = "clear"            # clear / blocked:<reason>
    sentiment_summary: str = "neutral"
    next_red_event: str = "none"
    next_red_minutes: int = 0
    size_multipliers: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """True when the news gate blocks trading."""
        return self.news_gate.startswith("blocked")

    def directional_bias(self, direction: str) -> int:
        """+1 when macro agrees with the direction, -1 against, 0 neutral."""
        want = 1 if direction == "buy" else -1
        if self.pair_bias * want > 15:
            return 1
        if self.pair_bias * want < -15:
            return -1
        return 0


class MacroAnalyzer:
    """Aggregates every institutional source into one macro read."""

    def __init__(self, intermarket: Optional[IntermarketAnalyzer] = None,
                 cot: Optional[COTReader] = None,
                 retail: Optional[RetailSentimentReader] = None,
                 calendar: Optional[CalendarEngine] = None,
                 sentiment: Optional[SentimentEngine] = None,
                 banks: Optional[CentralBankTracker] = None) -> None:
        self.intermarket = intermarket or IntermarketAnalyzer()
        self.cot = cot or COTReader()
        self.retail = retail or RetailSentimentReader()
        self.calendar = calendar or CalendarEngine()
        self.sentiment = sentiment
        self.banks = banks or CentralBankTracker()

    def analyze(self, pair: str, direction: str, now: Optional[datetime] = None) -> MacroResult:
        """Full macro read for one pair/direction (never raises)."""
        result = MacroResult()
        now = now or datetime.now(timezone.utc)
        pair = pair.upper()
        if pair == "XAUUSD":
            base, quote = "XAU", "USD"
        elif pair in ("NAS100", "US30"):
            base, quote = "USD", "USD"
        else:
            base, quote = pair[:3], pair[3:]
        result.base_currency, result.quote_currency = base, quote

        # ---- intermarket snapshot ----
        try:
            snap: IntermarketSnapshot = self.intermarket.snapshot()
            result.usd_bias = snap.usd_bias
            result.base_bias = snap.bias_for(base)
            result.quote_bias = snap.bias_for(quote)
            if base == quote:  # index: only USD leg
                result.base_bias = snap.usd_bias
                result.quote_bias = 0.0
            result.pair_bias = round(result.base_bias - result.quote_bias, 1)
            result.dxy_trend = snap.dxy_trend
            result.vix = snap.vix
            result.vix_term = "inverted" if snap.vix3m and snap.vix < snap.vix3m else "normal"
            result.yield_10y = snap.yield_10y
            result.yield_trend = "flat"
            result.size_multipliers["vix"] = snap.vix_position_scale
            if snap.vix_halt:
                result.news_gate = "blocked:vix_halt"
                result.notes.append("VIX halt level")
        except Exception as exc:
            logger.warning("intermarket snapshot failed: %s", exc)
            result.notes.append(f"intermarket unavailable: {exc}")

        # ---- COT ----
        try:
            cot_bias = self.cot.bias_for_pair(pair)
            result.cot_bias = cot_bias
            result.cot_aligned = cot_bias == direction or cot_bias == "neutral"
            snaps = self.cot.snapshots()
            cur = base if base in snaps else quote
            if cur in snaps:
                s = snaps[cur]
                result.cot_commercial_net = s.commercial_net
                result.cot_percentile = s.commercial_pctile
        except Exception as exc:
            logger.warning("COT read failed: %s", exc)
            result.notes.append(f"COT unavailable: {exc}")

        # ---- retail sentiment ----
        try:
            retail = self.retail.snapshots([pair])
            if pair in retail:
                s = retail[pair]
                result.retail_long_pct = s.pct_long
                result.retail_contrarian = s.contrarian_bias
        except Exception as exc:
            logger.warning("retail sentiment failed: %s", exc)

        # ---- news gate + sentiment ----
        try:
            result.news_gate = self.calendar.block_reason(pair, now) or "clear"
            title, minutes = self.calendar.next_red_event(pair)
            result.next_red_event = title or "none"
            result.next_red_minutes = minutes or 0
            result.size_multipliers["calendar"] = self.calendar.size_factor(pair, now)
        except Exception as exc:
            logger.warning("calendar gate failed: %s", exc)
            result.notes.append(f"calendar unavailable: {exc}")

        if self.sentiment is not None:
            try:
                curs = {base, quote, "USD"} - {""}
                result.sentiment_summary = self.sentiment.summary(curs)
            except Exception as exc:
                logger.warning("news sentiment failed: %s", exc)

        # ---- central bank proximity ----
        try:
            result.size_multipliers["central_bank"] = self.banks.size_multiplier(pair, now)
        except Exception as exc:
            logger.warning("central bank tracker failed: %s", exc)

        logger.info("macro %s %s: bias %.0f cot=%s retail=%.0f%% gate=%s",
                    pair, direction, result.pair_bias, result.cot_bias,
                    result.retail_long_pct, result.news_gate)
        return result
