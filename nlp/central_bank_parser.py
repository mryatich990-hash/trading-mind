"""Central bank statement parser (UPGRADE 2): Fed / ECB / BOE.

Fetches RSS feeds, extracts statements, scores hawkish/dovish language from
-100 (max dovish) to +100 (max hawkish), persists to central_bank_statements
and alerts via Telegram — with spam controls:

  1. ONLY items published in the last 24h are processed (no RSS archives).
  2. Items must contain at least one monetary-policy keyword (rate,
     inflation, policy, GDP, employment, hike, cut) — press releases about
     museum exhibits or staff appointments are skipped.
  3. Max 3 alerts per bank per UTC day.
  4. Dedup by headline + link, persisted in system_state — restarts and
     process crashes can never re-alert the same headline.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

from config import settings
from core import db

logger = logging.getLogger(__name__)

FEEDS = {
    "FED": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
    "BOE": "https://www.bankofengland.co.uk/rss/news",
}

HAWKISH_TERMS = {
    "rate hike": 25, "raise rates": 20, "tightening": 18, "hawkish": 20,
    "inflation concern": 15, "robust growth": 15, "strong labor market": 12,
    "above target": 12, "restrictive": 10,
}
DOVISH_TERMS = {
    "rate cut": 25, "lower rates": 20, "easing": 18, "dovish": 20,
    "economic concern": 15, "accommodative": 12, "downside risks": 12,
    "below target": 12, "stimulus": 10,
}

# Requirement 2: only monetary-policy-relevant items are processed at all.
POLICY_KEYWORDS = re.compile(
    r"\b(rate|rates|inflation|policy|gdp|employment|jobs|hike|cut|"
    r"monetary|interest|bank rate|mpc|fomc)\b", re.IGNORECASE)

CURRENCY = {"FED": "USD", "ECB": "EUR", "BOE": "GBP"}

MAX_AGE = timedelta(hours=24)          # requirement 1: 24h recency window
MAX_ALERTS_PER_BANK_PER_DAY = 3        # requirement 3
_ALERT_DAY_KEY = "cb_alert_day"        # system_state keys for the daily caps
_ALERT_COUNT_KEY = "cb_alert_counts"
_SEEN_KEY = "cb_seen_hashes"


@dataclass
class Statement:
    """One parsed central bank item."""

    bank: str
    title: str
    link: str
    score: float = 0.0
    label: str = "neutral"
    highlights: list = field(default_factory=list)
    published: datetime | None = None
    skipped: str = ""      # non-empty when the item was filtered out


def score_statement(text: str) -> tuple[float, str, list]:
    """Hawkish/dovish score -100..+100 with the matched phrases."""
    lowered = (text or "").lower()
    score, highlights = 0.0, []
    for term, weight in HAWKISH_TERMS.items():
        if term in lowered:
            score += weight
            highlights.append(f"+{term}")
    for term, weight in DOVISH_TERMS.items():
        if term in lowered:
            score -= weight
            highlights.append(f"-{term}")
    score = max(-100.0, min(100.0, score))
    label = "hawkish" if score >= 25 else ("dovish" if score <= -25 else "neutral")
    return round(score, 1), label, highlights[:8]


def _item_hash(bank: str, title: str, link: str) -> str:
    """Stable dedup key: bank + normalized headline (link as tiebreaker)."""
    norm = re.sub(r"\s+", " ", (title or "").strip().lower())
    return hashlib.sha256(f"{bank}|{norm}|{link}".encode()).hexdigest()[:24]


class _SpamState:
    """Persistent dedup + daily-cap counters (survives restarts)."""

    MAX_SEEN = 2000

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._day: str = ""
        self._counts: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        try:
            import json

            self._seen = set(json.loads(db.get_state(_SEEN_KEY, "[]")))
            self._day = db.get_state(_ALERT_DAY_KEY, "")
            self._counts = json.loads(db.get_state(_ALERT_COUNT_KEY, "{}"))
        except Exception as exc:
            logger.warning("cb spam-state load failed: %s", exc)

    def _save(self) -> None:
        try:
            import json

            seen = list(self._seen)[-self.MAX_SEEN:]
            db.set_state(_SEEN_KEY, json.dumps(seen))
            db.set_state(_ALERT_DAY_KEY, self._day)
            db.set_state(_ALERT_COUNT_KEY, json.dumps(self._counts))
        except Exception as exc:
            logger.warning("cb spam-state save failed: %s", exc)

    def roll_day(self) -> None:
        """Reset counters when the UTC date changes."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._counts = {}
            self._save()

    def seen(self, h: str) -> bool:
        return h in self._seen

    def mark_seen(self, h: str) -> None:
        self._seen.add(h)
        if len(self._seen) > self.MAX_SEEN * 2:
            self._seen = set(sorted(self._seen)[-self.MAX_SEEN:])
        # persist immediately: a restart must never re-alert a headline
        self._save()

    def can_alert(self, bank: str) -> bool:
        """True when this bank is still under the daily alert cap."""
        return self._counts.get(bank, 0) < MAX_ALERTS_PER_BANK_PER_DAY

    def count_alert(self, bank: str) -> None:
        self._counts[bank] = self._counts.get(bank, 0) + 1
        self._save()


class CentralBankParser:
    """Polls central bank RSS feeds and stores/alerts new statements."""

    def __init__(self, notifier=None) -> None:
        self.notifier = notifier
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (trading-bot)"
        self.last_poll = 0.0
        self.spam = _SpamState()
        self._skipped = 0

    # ---- filtering (requirements 1 & 2) ----

    @staticmethod
    def _parse_pubdate(raw: str) -> datetime | None:
        """RSS pubDate -> aware UTC datetime; None when absent/unparseable."""
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            dt = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            try:  # ISO fallback (some feeds)
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @classmethod
    def _is_recent(cls, published: datetime | None) -> bool:
        """Requirement 1: only the last 24h. Items without a parseable
        pubDate are treated as NOT recent — archived pages stay silent."""
        if published is None:
            return False
        return datetime.now(timezone.utc) - published <= MAX_AGE

    @staticmethod
    def _is_policy_relevant(text: str) -> bool:
        """Requirement 2: must mention monetary policy."""
        return bool(POLICY_KEYWORDS.search(text or ""))

    # ---- fetching ----

    def _fetch(self, bank: str, url: str) -> list[Statement]:
        """Parse one RSS feed into statements (network errors swallowed)."""
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            logger.warning("cb feed %s failed: %s", bank, exc)
            return []
        items = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = (item.findtext("description") or "").strip()
            pub_raw = item.findtext("pubDate") or item.findtext(
                "{http://purl.org/dc/elements/1.1/}date") or ""
            if not title:
                continue
            published = self._parse_pubdate(pub_raw)
            text = f"{title}. {re.sub('<[^>]+>', ' ', desc)}"
            st = Statement(bank=bank, title=title, link=link, published=published)
            if not self._is_recent(published):
                st.skipped = f"older than 24h ({(pub_raw or 'no pubDate')[:32]})"
            elif not self._is_policy_relevant(text):
                st.skipped = "not monetary policy"
            else:
                score, label, highlights = score_statement(text)
                st.score, st.label, st.highlights = score, label, highlights
            items.append(st)
        return items

    # ---- polling ----

    def poll(self) -> list[Statement]:
        """Fetch all feeds; persist + alert qualifying new statements.

        Returns the items that were actually processed (new + relevant).
        Everything else is logged as skipped with its reason.
        """
        new: list[Statement] = []
        self.spam.roll_day()
        self._skipped = 0
        for bank, url in FEEDS.items():
            for st in self._fetch(bank, url):
                h = _item_hash(st.bank, st.title, st.link)
                if st.skipped:
                    self._skipped += 1
                    logger.debug("cb %s skipped: %s -- %s",
                                 st.bank, st.skipped, st.title[:80])
                    continue
                if self.spam.seen(h):
                    continue  # requirement 4: never alert the same headline
                self.spam.mark_seen(h)
                new.append(st)
                self._persist(st)
                if self.spam.can_alert(st.bank):
                    self._alert(st)
                    self.spam.count_alert(st.bank)
                else:
                    logger.info("cb %s alert suppressed (daily cap %d): %s",
                                st.bank, MAX_ALERTS_PER_BANK_PER_DAY,
                                st.title[:80])
        self.last_poll = time.time()
        if new or self._skipped:
            logger.info("cb poll: %d new, %d skipped (age/keywords)",
                        len(new), self._skipped)
        return new

    # ---- storage / alerts ----

    def _persist(self, st: Statement) -> None:
        """Store in central_bank_statements (table created on demand)."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine, _WRITE_LOCK, pg_compatible

            with engine.begin() as conn, _WRITE_LOCK:
                conn.execute(sqltext(pg_compatible(
                    "CREATE TABLE IF NOT EXISTS central_bank_statements ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TIMESTAMP "
                    "DEFAULT CURRENT_TIMESTAMP, bank VARCHAR(8), currency VARCHAR(4), "
                    "title TEXT, link TEXT, score DOUBLE, label VARCHAR(12))")))
                cur = CURRENCY.get(st.bank, "")
                conn.execute(sqltext(
                    "INSERT INTO central_bank_statements (bank, currency, title, link, "
                    "score, label) VALUES (:b, :c, :t, :l, :s, :la)"),
                    {"b": st.bank, "c": cur, "t": st.title[:500], "l": st.link[:500],
                     "s": st.score, "la": st.label})
        except Exception as exc:
            logger.warning("cb persist failed: %s", exc)

    def _alert(self, st: Statement) -> None:
        """Telegram alert on new qualifying statement."""
        if self.notifier is None:
            return
        cur = CURRENCY.get(st.bank, "")
        direction = "bullish" if st.score >= 25 else (
            "bearish" if st.score <= -25 else "neutral")
        try:
            self.notifier.send(f"🏦 {st.bank} statement detected — {st.label} "
                               f"score: {st.score:+.0f} — {cur} {direction} bias\n"
                               f"{st.title[:160]}")
        except Exception:
            pass

    def latest(self, limit: int = 10) -> list[dict]:
        """Recent statements for the dashboard."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine

            with engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT created_at, bank, label, score, title FROM "
                    "central_bank_statements ORDER BY id DESC LIMIT :l"),
                    {"l": limit}).mappings().all()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def scores(self) -> dict:
        """Current hawkish/dovish score per bank."""
        out = {}
        for bank in FEEDS:
            rows = [r for r in self.latest(20) if r["bank"] == bank]
            out[bank] = round(sum(r["score"] for r in rows) / len(rows), 1) if rows else 0.0
        return out
