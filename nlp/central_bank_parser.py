"""Central bank statement parser (UPGRADE 2): Fed / ECB / BOE.

Fetches RSS feeds, extracts statements, scores hawkish/dovish language from
-100 (max dovish) to +100 (max hawkish), persists to central_bank_statements
and alerts via Telegram on every new statement.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

from config import settings

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

CURRENCY = {"FED": "USD", "ECB": "EUR", "BOE": "GBP"}


@dataclass
class Statement:
    """One parsed central bank item."""

    bank: str
    title: str
    link: str
    score: float = 0.0
    label: str = "neutral"
    highlights: list = field(default_factory=list)


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


class CentralBankParser:
    """Polls central bank RSS feeds and stores/alerts new statements."""

    def __init__(self, notifier=None) -> None:
        self.notifier = notifier
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (trading-bot)"
        self._seen_links: set[str] = set()
        self.last_poll = 0.0

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
            if not title or link in self._seen_links:
                continue
            text = f"{title}. {re.sub('<[^>]+>', ' ', desc)}"
            score, label, highlights = score_statement(text)
            items.append(Statement(bank=bank, title=title, link=link,
                                   score=score, label=label, highlights=highlights))
        return items

    def poll(self) -> list[Statement]:
        """Fetch all feeds; persist + alert new statements. Returns new items."""
        new: list[Statement] = []
        for bank, url in FEEDS.items():
            for st in self._fetch(bank, url):
                self._seen_links.add(st.link)
                new.append(st)
                self._persist(st)
                self._alert(st)
        self.last_poll = __import__("time").time()
        return new

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
        """Telegram alert on new statement."""
        if self.notifier is None:
            return
        cur = CURRENCY.get(st.bank, "")
        direction = "bullish" if st.score >= 25 else ("bearish" if st.score <= -25 else "neutral")
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
