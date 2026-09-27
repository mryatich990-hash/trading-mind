"""Central bank tracker: meeting dates drive size reductions and trade blocks."""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["CentralBankTracker"]

# currency -> bank name; meeting dates seeded for 2026, updatable in DB
DEFAULT_BANKS: dict[str, str] = {
    "USD": "Federal Reserve", "EUR": "ECB", "GBP": "Bank of England",
    "JPY": "Bank of Japan", "CHF": "SNB", "AUD": "RBA", "CAD": "BoC", "NZD": "RBNZ",
}

DEFAULT_MEETINGS_2026 = [
    {"currency": "USD", "date": "2026-01-28"}, {"currency": "USD", "date": "2026-03-18"},
    {"currency": "USD", "date": "2026-04-29"}, {"currency": "USD", "date": "2026-06-17"},
    {"currency": "USD", "date": "2026-07-29"}, {"currency": "USD", "date": "2026-09-16"},
    {"currency": "USD", "date": "2026-10-28"}, {"currency": "USD", "date": "2026-12-09"},
    {"currency": "EUR", "date": "2026-02-05"}, {"currency": "EUR", "date": "2026-03-12"},
    {"currency": "EUR", "date": "2026-04-30"}, {"currency": "EUR", "date": "2026-06-11"},
    {"currency": "EUR", "date": "2026-07-23"}, {"currency": "EUR", "date": "2026-09-10"},
    {"currency": "EUR", "date": "2026-10-29"}, {"currency": "EUR", "date": "2026-12-17"},
    {"currency": "GBP", "date": "2026-02-05"}, {"currency": "GBP", "date": "2026-03-19"},
    {"currency": "GBP", "date": "2026-05-07"}, {"currency": "GBP", "date": "2026-06-18"},
    {"currency": "GBP", "date": "2026-08-06"}, {"currency": "GBP", "date": "2026-09-17"},
    {"currency": "GBP", "date": "2026-11-05"}, {"currency": "GBP", "date": "2026-12-17"},
    {"currency": "JPY", "date": "2026-01-23"}, {"currency": "JPY", "date": "2026-03-19"},
    {"currency": "JPY", "date": "2026-05-01"}, {"currency": "JPY", "date": "2026-06-16"},
    {"currency": "JPY", "date": "2026-07-31"}, {"currency": "JPY", "date": "2026-09-18"},
    {"currency": "JPY", "date": "2026-10-30"}, {"currency": "JPY", "date": "2026-12-18"},
]


class CentralBankTracker:
    """Tracks meeting dates; windows reduce size or block trading."""

    def __init__(self) -> None:
        self._ensure_seeded()

    @staticmethod
    def _ensure_seeded() -> None:
        """Seed meeting dates into system_state once."""
        if db.get_state("cb_meetings", "") == "":
            db.set_state("cb_meetings", json.dumps(DEFAULT_MEETINGS_2026))

    def meetings(self) -> list[dict]:
        """All known meeting dates."""
        try:
            return json.loads(db.get_state("cb_meetings", "[]"))
        except ValueError:
            return []

    def add_meeting(self, currency: str, date: str) -> None:
        """Operator adds a meeting date (YYYY-MM-DD)."""
        meetings = self.meetings()
        meetings.append({"currency": currency.upper(), "date": date})
        db.set_state("cb_meetings", json.dumps(meetings))
        db.audit("calendar", "cb_meeting_added", f"{currency} {date}")

    def _pair_currencies(self, pair: str) -> set[str]:
        """Currencies relevant to a pair."""
        pair = pair.upper()
        if pair == "XAUUSD":
            return {"XAU", "USD"}
        if pair in ("NAS100", "US30"):
            return {"USD"}
        return {pair[:3], pair[3:]}

    def is_decision_day(self, pair: str, now: Optional[datetime] = None) -> bool:
        """True when a linked central bank decides today."""
        now = now or datetime.now(timezone.utc)
        today = now.date()
        for m in self.meetings():
            if m["currency"] not in self._pair_currencies(pair):
                continue
            try:
                if datetime.strptime(m["date"], "%Y-%m-%d").date() == today:
                    return True
            except ValueError:
                continue
        return False

    def meeting_within_week(self, pair: str, now: Optional[datetime] = None) -> bool:
        """True when a linked meeting happens within 7 days (reduce size 20%)."""
        now = now or datetime.now(timezone.utc)
        window = now + timedelta(days=7)
        for m in self.meetings():
            if m["currency"] not in self._pair_currencies(pair):
                continue
            try:
                meeting = datetime.strptime(m["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if now <= meeting <= window:
                return True
        return False

    def size_multiplier(self, pair: str, now: Optional[datetime] = None) -> float:
        """Position multiplier from central-bank proximity (0 or 0.8)."""
        if self.is_decision_day(pair, now):
            return 0.0
        if self.meeting_within_week(pair, now):
            return 0.8
        return 1.0
