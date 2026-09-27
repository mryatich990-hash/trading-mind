"""Economic calendar engine: ForexFactory RSS, red-event windows, surprise tracking."""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import requests

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["CalendarEvent", "CalendarEngine"]

# per-event-type windows (before_min, after_min) for red events
RED_WINDOWS = {
    "nfp": (45, 90), "nonfarm": (45, 90),
    "interest rate": (60, 120), "rate decision": (60, 120), "fomc": (60, 120),
    "cpi": (30, 60),
}
DEFAULT_RED_WINDOW = (120, 60)  # generic red: 2h before, 1h after
ORANGE_SIZE_FACTOR = 0.7  # orange event: reduce size 30%

PAIR_CURRENCIES = {
    "EURUSD": {"EUR", "USD"}, "GBPUSD": {"GBP", "USD"}, "XAUUSD": {"XAU", "USD"},
    "USDJPY": {"USD", "JPY"}, "GBPJPY": {"GBP", "JPY"}, "EURJPY": {"EUR", "JPY"},
    "USDCHF": {"USD", "CHF"}, "AUDUSD": {"AUD", "USD"}, "NAS100": {"USD"}, "US30": {"USD"},
}


@dataclass
class CalendarEvent:
    """One calendar entry."""

    title: str
    country: str
    impact: str  # red / orange / yellow
    event_time: Optional[datetime]


class CalendarEngine:
    """Fetches the weekly calendar and enforces trading windows."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "trading-bot/1.0"})
        self._events: list[CalendarEvent] = []
        self._fetched_at: Optional[datetime] = None

    def refresh(self, force: bool = False) -> list[CalendarEvent]:
        """Fetch the ForexFactory weekly calendar (30-min TTL)."""
        now = datetime.now(timezone.utc)
        if not force and self._fetched_at and now - self._fetched_at < timedelta(minutes=30):
            return self._events
        events: list[CalendarEvent] = []
        for attempt in range(3):
            try:
                feed = feedparser.parse(settings.FOREXFACTORY_RSS)
                for entry in feed.entries:
                    title = getattr(entry, "title", "")
                    blob = f"{title} {getattr(entry, 'summary', '')}".lower()
                    if "red" in blob or "high" in blob:
                        impact = "red"
                    elif "orange" in blob or "medium" in blob:
                        impact = "orange"
                    else:
                        impact = "yellow"
                    when = None
                    for attr in ("published_parsed", "updated_parsed"):
                        parsed = getattr(entry, attr, None)
                        if parsed:
                            when = datetime(*parsed[:6], tzinfo=timezone.utc)
                            break
                    country = ""
                    for cur in ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"):
                        if cur.lower() in title.lower():
                            country = cur
                            break
                    events.append(CalendarEvent(title, country, impact, when))
                self._events, self._fetched_at = events, now
                logger.info("calendar refreshed: %d events", len(events))
                return events
            except Exception as exc:
                logger.warning("calendar fetch attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        return self._events

    @staticmethod
    def _window_for(title: str) -> tuple[int, int]:
        """Before/after minutes for a red event by its title."""
        low = title.lower()
        for key, window in RED_WINDOWS.items():
            if key in low:
                return window
        return DEFAULT_RED_WINDOW

    def block_reason(self, pair: str, now: Optional[datetime] = None) -> str:
        """Return the blocking reason, or '' when trading is allowed."""
        now = now or datetime.now(timezone.utc)
        self.refresh()
        currencies = PAIR_CURRENCIES.get(pair.upper(), {pair[:3], pair[3:]})
        for ev in self._events:
            if ev.event_time is None or ev.country not in currencies:
                continue
            delta_min = (ev.event_time - now).total_seconds() / 60.0
            if ev.impact == "red":
                before, after = self._window_for(ev.title)
                if -after <= delta_min <= before:
                    return f"red event: {ev.title} in {delta_min:.0f} min"
            elif ev.impact == "orange" and -30 <= delta_min <= 60:
                return ""  # orange reduces size, handled by size_factor
        return ""

    def size_factor(self, pair: str, now: Optional[datetime] = None) -> float:
        """Size multiplier from calendar proximity (orange events -> 0.7)."""
        now = now or datetime.now(timezone.utc)
        self.refresh()
        currencies = PAIR_CURRENCIES.get(pair.upper(), {pair[:3], pair[3:]})
        for ev in self._events:
            if ev.event_time is None or ev.country not in currencies:
                continue
            delta_min = (ev.event_time - now).total_seconds() / 60.0
            if ev.impact == "orange" and -30 <= delta_min <= 60:
                return ORANGE_SIZE_FACTOR
        return 1.0

    def next_red_event(self, pair: str) -> tuple[Optional[str], Optional[int]]:
        """(title, minutes) of the next red event for a pair."""
        self.refresh()
        currencies = PAIR_CURRENCIES.get(pair.upper(), {pair[:3], pair[3:]})
        now = datetime.now(timezone.utc)
        best_t, best_title = None, None
        for ev in self._events:
            if ev.impact != "red" or ev.event_time is None or ev.country not in currencies:
                continue
            if ev.event_time > now and (best_t is None or ev.event_time < best_t):
                best_t, best_title = ev.event_time, ev.title
        if best_t:
            return best_title, int((best_t - now).total_seconds() / 60.0)
        return None, None
