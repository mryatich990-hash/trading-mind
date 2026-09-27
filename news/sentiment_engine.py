"""News sentiment engine: multi-source headlines classified by Groq, aggregated 2h."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import requests

from config import settings
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["Headline", "SentimentEngine"]

CURRENCY_KEYWORDS = {
    "USD": ("usd", "dollar", "fed ", "federal reserve", "powell", "treasury"),
    "EUR": ("eur", "euro", "ecb", "eurozone"),
    "GBP": ("gbp", "pound", "sterling", "boe"),
    "JPY": ("jpy", "yen", "boj"),
    "XAU": ("gold", "xau", "bullion"),
}


@dataclass
class Headline:
    """A classified headline."""

    text: str
    source: str
    published_at: Optional[datetime]
    currencies: list[str] = field(default_factory=list)
    sentiment: str = "neutral"  # positive / negative / neutral
    impact: str = "low"  # high / medium / low


class SentimentEngine:
    """Fetches headlines and scores them via Groq (cached 20 minutes)."""

    def __init__(self, groq_ask_json: Optional[callable] = None) -> None:
        """
        Args:
            groq_ask_json: callable(prompt, pair, stage) -> dict | None. Wired to
                ai.groq_brain.GroqBrain.ask_json in production; keyword fallback
                is used when Groq is unavailable.
        """
        self.groq = groq_ask_json
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "trading-bot/1.0"})
        self._headlines: list[Headline] = []
        self._fetched_at: Optional[datetime] = None
        self._classification_cache: dict[str, tuple[datetime, Headline]] = {}

    def refresh(self, force: bool = False) -> list[Headline]:
        """Fetch headlines from NewsAPI + Reuters/Bloomberg/FXStreet RSS (15-min TTL)."""
        now = datetime.now(timezone.utc)
        if not force and self._fetched_at and now - self._fetched_at < timedelta(minutes=15):
            return self._headlines
        collected: list[Headline] = []

        if settings.NEWS_API_KEY:
            try:
                resp = self.session.get(
                    "https://newsapi.org/v2/top-headlines",
                    params={"apiKey": settings.NEWS_API_KEY, "category": "business",
                            "language": "en", "pageSize": 15},
                    timeout=12,
                )
                resp.raise_for_status()
                for a in resp.json().get("articles", []):
                    published = None
                    if a.get("publishedAt"):
                        try:
                            published = datetime.fromisoformat(
                                a["publishedAt"].replace("Z", "+00:00"))
                        except ValueError:
                            pass
                    collected.append(Headline(a.get("title", ""), "newsapi", published))
            except Exception as exc:
                logger.warning("NewsAPI fetch failed: %s", exc)

        for name, url in (("reuters", settings.REUTERS_RSS),
                          ("bloomberg", settings.BLOOMBERG_RSS),
                          ("fxstreet", settings.FXSTREET_RSS)):
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:12]:
                    parsed = getattr(entry, "published_parsed", None)
                    when = datetime(*parsed[:6], tzinfo=timezone.utc) if parsed else None
                    collected.append(Headline(getattr(entry, "title", ""), name, when))
            except Exception as exc:
                logger.warning("%s RSS failed: %s", name, exc)

        self._headlines = [h for h in collected if h.text][:30]
        self._fetched_at = now
        self._classify_all()
        logger.info("headlines: %d collected", len(self._headlines))
        return self._headlines

    def _classify_all(self) -> None:
        """Classify uncached headlines (Groq when wired, keyword fallback otherwise)."""
        now = datetime.now(timezone.utc)
        for h in self._headlines:
            key = h.text.strip().lower()
            cached = self._classification_cache.get(key)
            if cached and now - cached[0] < timedelta(minutes=20):
                h.currencies, h.sentiment, h.impact = (
                    cached[1].currencies, cached[1].sentiment, cached[1].impact)
                continue
            currencies = [c for c, kws in CURRENCY_KEYWORDS.items()
                          if any(k in h.text.lower() for k in kws)]
            sentiment, impact = "neutral", "low"
            if self.groq is not None:
                prompt = (
                    "Does this headline affect USD, EUR, GBP, JPY or Gold? "
                    f"Headline: {h.text}\n"
                    'Return JSON only: {"currencies_affected": [], '
                    '"sentiment": "positive|negative|neutral", "impact": "high|medium|low", '
                    '"summary": "one sentence"}'
                )
                parsed = self.groq(prompt, pair="NEWS", stage="sentiment")
                if parsed and isinstance(parsed.get("currencies_affected"), list):
                    currencies = [str(c).upper() for c in parsed["currencies_affected"]
                                  if str(c).upper() in CURRENCY_KEYWORDS]
                    sentiment = str(parsed.get("sentiment", "neutral"))
                    impact = str(parsed.get("impact", "low"))
            h.currencies, h.sentiment, h.impact = currencies, sentiment, impact
            self._classification_cache[key] = (now, h)

    def aggregate(self, currencies: set[str], window_hours: float = 2.0) -> dict[str, int]:
        """Net sentiment score per currency over the last window (-100..+100)."""
        self.refresh()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        scores = {c: 0 for c in currencies}
        weights = {"high": 3, "medium": 2, "low": 1}
        for h in self._headlines:
            if h.published_at and h.published_at < cutoff:
                continue
            if not (set(h.currencies) & currencies):
                continue
            w = weights.get(h.impact, 1)
            if h.sentiment == "positive":
                sign = 1
            elif h.sentiment == "negative":
                sign = -1
            else:
                continue
            for c in set(h.currencies) & currencies:
                scores[c] = scores.get(c, 0) + sign * w
        for c in scores:
            scores[c] = max(-100, min(100, scores[c] * 10))
        return scores

    def summary(self, currencies: set[str]) -> str:
        """Human-readable 2h sentiment summary for prompts."""
        scores = self.aggregate(currencies)
        parts = [f"{c}: {s:+d}" for c, s in sorted(scores.items()) if s != 0]
        return ", ".join(parts) if parts else "neutral"
