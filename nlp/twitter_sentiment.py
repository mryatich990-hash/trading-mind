"""Twitter/X crowd sentiment (UPGRADE 2): contrarian crowd gauge.

Twitter API v2 recent-search when TWITTER_BEARER_TOKEN is set; without a
token the engine stays inactive (never raises). Tweets are scored by the
FinBERT engine; extreme crowd bullishness is treated as contrarian bearish
and vice versa. Refresh cadence: 30 minutes.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from config import settings

logger = logging.getLogger(__name__)

SEARCH_TERMS = {
    "EURUSD": "(\"EURUSD\" OR \"$EURUSD\" OR \"euro dollar\") -is:retweet lang:en",
    "GBPUSD": "(\"GBPUSD\" OR \"$GBPUSD\" OR \"pound dollar\") -is:retweet lang:en",
    "XAUUSD": "(\"gold price\" OR \"$GOLD\" OR \"xauusd\") -is:retweet lang:en",
    "USD": "(\"dollar index\" OR \"$DXY\" OR \"usdollar\") -is:retweet lang:en",
}


class TwitterSentiment:
    """Crowd sentiment per pair with contrarian interpretation."""

    def __init__(self, finbert=None, refresh_sec: int = 1800) -> None:
        self.token = settings.TWITTER_BEARER_TOKEN
        self.finbert = finbert
        self.refresh_sec = refresh_sec
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.token}"
        self.last_poll = 0.0
        self.scores: dict[str, dict] = {}
        self.active = bool(self.token) and finbert is not None
        if not self.active:
            logger.info("twitter sentiment inactive (%s)", "no token" if not self.token
                        else "no finbert engine")

    def _fetch_tweets(self, term: str, max_results: int = 50) -> list[str]:
        """Recent tweets for one search term (errors -> empty)."""
        try:
            resp = self.session.get(
                "https://api.twitter.com/2/tweets/search/recent",
                params={"query": term, "max_results": min(max_results, 100)},
                timeout=15)
            resp.raise_for_status()
            return [t["text"] for t in resp.json().get("data", [])]
        except Exception as exc:
            logger.warning("twitter fetch failed: %s", exc)
            return []

    def poll(self) -> dict:
        """Refresh all pair scores; returns {pair: {crowd, contrarian, samples}}."""
        if not self.active:
            return self.scores
        for pair, term in SEARCH_TERMS.items():
            tweets = self._fetch_tweets(term)
            if not tweets:
                continue
            scored = [self.finbert.score_text(t)["score"] for t in tweets]
            mean = sum(scored) / len(scored)
            crowd = round(max(-100.0, min(100.0, mean)), 1)
            # contrarian: extreme crowd sentiment flips the signal
            contrarian = 0.0
            if crowd >= 40:
                contrarian = -(crowd - 40) * 1.5
            elif crowd <= -40:
                contrarian = -(crowd + 40) * 1.5
            self.scores[pair] = {"crowd": crowd,
                                 "contrarian_signal": round(contrarian, 1),
                                 "samples": len(scored)}
        self.last_poll = time.time()
        return self.scores

    def maybe_poll(self) -> dict:
        """Poll only when the refresh window elapsed."""
        if self.active and time.time() - self.last_poll >= self.refresh_sec:
            return self.poll()
        return self.scores

    def status(self) -> dict:
        """Dashboard payload."""
        return {"active": self.active, "last_poll": self.last_poll,
                "scores": self.scores}
