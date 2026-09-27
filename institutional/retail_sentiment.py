"""Retail sentiment: contrarian indicator from Myfxbook community outlook."""

import time
from dataclasses import dataclass
from typing import Optional

import requests

from config import settings
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["SentimentSnapshot", "RetailSentimentReader"]

MYFXBOOK_URL = "https://www.myfxbook.com/api"


@dataclass
class SentimentSnapshot:
    """Retail positioning for one pair."""

    pair: str
    pct_long: float
    pct_short: float

    @property
    def contrarian_bias(self) -> str:
        """Contrarian bias: extreme retail long -> sell, extreme short -> buy."""
        if self.pct_long >= 75:
            return "sell"
        if self.pct_short >= 75:
            return "buy"
        return "neutral"


class RetailSentimentReader:
    """Fetches retail long/short percentages (cached 1 hour)."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self._cache: Optional[tuple[float, dict[str, SentimentSnapshot]]] = None
        self._ttl = 3600.0

    def snapshots(self, pairs: Optional[list[str]] = None) -> dict[str, SentimentSnapshot]:
        """Return sentiment for the requested pairs (best effort, never raises)."""
        pairs = pairs or ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        now = time.monotonic()
        if self._cache and now - self._cache[0] < self._ttl:
            return {p: self._cache[1][p] for p in pairs if p in self._cache[1]}

        fetched: dict[str, SentimentSnapshot] = {}
        # Myfxbook community outlook has no fully open API; use their public
        # outlook endpoint when a session is configured, else graceful skip.
        if settings.MYFXBOOK_API_KEY:
            try:
                resp = self.session.get(
                    f"{MYFXBOOK_URL}/get-community-outlook.json",
                    params={"api_key": settings.MYFXBOOK_API_KEY}, timeout=15,
                )
                resp.raise_for_status()
                for item in resp.json().get("symbols", []):
                    symbol = str(item.get("name", "")).replace("/", "").upper()
                    pct_long = float(item.get("longPercentage", 50.0))
                    fetched[symbol] = SentimentSnapshot(
                        pair=symbol, pct_long=round(pct_long, 1),
                        pct_short=round(100.0 - pct_long, 1),
                    )
            except Exception as exc:
                logger.warning("retail sentiment fetch failed: %s", exc)
        else:
            logger.info("MYFXBOOK_API_KEY not set; retail sentiment disabled")
        self._cache = (now, fetched)
        return {p: fetched[p] for p in pairs if p in fetched}
