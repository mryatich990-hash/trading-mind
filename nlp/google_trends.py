"""Google Trends integration (UPGRADE 2): retail fear/panic tracker.

pytrends when installed (unofficial API, may rate-limit — errors are always
swallowed); tracks fear terms per region. Rising "gold" searches = fear =
gold-supportive; spikes in "dollar collapse" = retail panic = contrarian USD
bullish. Weekly digest feeds the Sunday Groq journal.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

FEAR_TERMS = {
    "gold_price": ("gold price", "rising_gold_fear"),
    "dollar_collapse": ("dollar collapse", "usd_panic"),
    "forex_crash": ("forex crash", "market_fear"),
    "inflation": ("inflation", "inflation_anxiety"),
    "recession": ("recession", "recession_fear"),
}


class GoogleTrends:
    """Search-interest tracker with contrarian interpretation."""

    def __init__(self) -> None:
        self.enabled = settings.GOOGLE_TRENDS_ENABLED
        self._pytrends = None
        if self.enabled:
            try:
                from pytrends.request import TrendReq  # type: ignore

                self._pytrends = TrendReq(hl="en-US", tz=0)
            except Exception as exc:
                logger.warning("pytrends unavailable (%s) - trends inactive", exc)
                self._pytrends = None
        self.last_poll = 0.0
        self.data: dict[str, dict] = {}

    def poll(self) -> dict:
        """Refresh interest-over-time for all tracked terms (7d window)."""
        if not self.enabled or self._pytrends is None:
            return self.data
        for key, (term, _) in FEAR_TERMS.items():
            try:
                self._pytrends.build_payload([term], timeframe="now 7-d")
                df = self._pytrends.interest_over_time()
                if df is None or df.empty:
                    continue
                series = df[term]
                recent = float(series.tail(2).mean())
                baseline = float(series.mean()) or 1.0
                spike = round((recent / baseline - 1.0) * 100.0, 1)  # % above avg
                self.data[key] = {"term": term, "spike_pct": spike,
                                  "recent": round(recent, 1)}
            except Exception as exc:
                logger.debug("trends %s failed: %s", key, exc)
        self.last_poll = time.time()
        return self.data

    def signals(self) -> dict:
        """Contrarian/fear signals derived from trend spikes."""
        out = {}
        gold = self.data.get("gold_price", {}).get("spike_pct", 0.0)
        usd_panic = self.data.get("dollar_collapse", {}).get("spike_pct", 0.0)
        out["gold_fear_bid"] = round(min(max(gold, 0.0) / 10.0, 5.0), 1)  # 0..5 tilt
        out["usd_contrarian_bullish"] = round(min(max(usd_panic, 0.0) / 20.0, 5.0), 1)
        out["fear_level"] = round(
            sum(max(0.0, v.get("spike_pct", 0.0)) for v in self.data.values())
            / max(1, len(self.data)), 1)
        return out

    def weekly_digest(self) -> str:
        """One-paragraph digest for the Sunday Groq journal."""
        if not self.data:
            return "Google Trends: no data collected this week."
        parts = [f"{v['term']} {v['spike_pct']:+.0f}%" for v in self.data.values()]
        sig = self.signals()
        return (f"Google Trends weekly: {', '.join(parts)}. "
                f"Fear level {sig['fear_level']}/100. "
                f"Gold fear-bid {sig['gold_fear_bid']}, USD contrarian tilt "
                f"{sig['usd_contrarian_bullish']}.")

    def status(self) -> dict:
        """Dashboard payload."""
        return {"enabled": self.enabled, "active": self._pytrends is not None,
                "last_poll": self.last_poll, "data": self.data,
                "signals": self.signals()}
