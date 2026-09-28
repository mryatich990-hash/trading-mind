"""Tests: central bank feed spam controls (24h recency, keyword filter,
3/day cap, persistent dedup)."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nlp.central_bank_parser import (
    MAX_ALERTS_PER_BANK_PER_DAY, CentralBankParser, _item_hash, score_statement)


def _stmt(bank="BOE", title="Bank Rate increased to 5.25%", age_hours=1.0,
          link="https://example.org/x"):
    """Build a Statement pre-marked with an age-relative published time."""
    from nlp.central_bank_parser import Statement

    return Statement(bank=bank, title=title, link=link,
                     published=datetime.now(timezone.utc) - timedelta(hours=age_hours))


class TestFilters:
    def test_old_item_skipped(self, temp_db):
        p = CentralBankParser()
        assert not p._is_recent(_stmt(age_hours=30).published)

    def test_fresh_item_passes(self, temp_db):
        p = CentralBankParser()
        assert p._is_recent(_stmt(age_hours=2).published)

    def test_missing_pubdate_treated_as_old(self, temp_db):
        """Archived pages without parseable dates must never alert."""
        p = CentralBankParser()
        assert not p._is_recent(None)

    def test_policy_keywords_pass(self, temp_db):
        assert CentralBankParser._is_policy_relevant(
            "Bank Rate increased: monetary policy summary")
        assert CentralBankParser._is_policy_relevant(
            "Inflation falls; committee considers rate cut")

    def test_non_policy_skipped(self, temp_db):
        assert not CentralBankParser._is_policy_relevant(
            "Bank of England announces new office space in Leeds")
        assert not CentralBankParser._is_policy_relevant(
            "Museum exhibition opening tonight")

    def test_pubdate_parsing(self, temp_db):
        dt = CentralBankParser._parse_pubdate(
            "Fri, 25 Sep 2026 07:30:00 +0100")
        assert dt is not None and dt.tzinfo is not None


class TestDedupAndCaps:
    def _parser(self, sent):
        # parser expects a Bot-like object with .send()
        return CentralBankParser(
            notifier=SimpleNamespace(send=lambda t: sent.append(t)))

    def _pipe(self, p, st, text):
        """Run one item through the poll-path logic without network."""
        h = _item_hash(st.bank, st.title, st.link)
        if p.spam.seen(h):
            return False
        p.spam.mark_seen(h)
        if p.spam.can_alert(st.bank):
            p._alert(st)
            p.spam.count_alert(st.bank)
        return True

    def test_same_headline_never_twice(self, temp_db):
        sent = []
        p = self._parser(sent)
        st = _stmt()
        assert self._pipe(p, st, "") is True
        assert self._pipe(p, st, "") is False   # duplicate suppressed
        assert len(sent) == 1

    def test_dedup_survives_restart(self, temp_db):
        sent = []
        p1 = self._parser(sent)
        st = _stmt()
        self._pipe(p1, st, "")
        # simulate process restart: brand-new parser instance
        p2 = self._parser(sent)
        assert self._pipe(p2, st, "") is False
        assert len(sent) == 1

    def test_daily_cap_three_per_bank(self, temp_db):
        sent = []
        p = self._parser(sent)
        for i in range(5):
            st = _stmt(title=f"Rate decision variant {i}", link=f"https://x.org/{i}")
            self._pipe(p, st, "")
        assert len(sent) == MAX_ALERTS_PER_BANK_PER_DAY

    def test_cap_resets_next_day(self, temp_db):
        sent = []
        p = self._parser(sent)
        for i in range(3):
            p.spam.count_alert("BOE")
        assert not p.spam.can_alert("BOE")
        p.spam._day = "2000-01-01"      # force day rollover
        p.spam.roll_day()
        assert p.spam.can_alert("BOE")

    def test_other_bank_unaffected_by_cap(self, temp_db):
        sent = []
        p = self._parser(sent)
        for i in range(3):
            p.spam.count_alert("BOE")
        assert p.spam.can_alert("FED")

    def test_hash_stable(self, temp_db):
        assert _item_hash("BOE", "Rate  up", "l") == \
            _item_hash("BOE", "  rate   up ", "l")


class TestScore:
    def test_hawkish_dovish_bounds(self, temp_db):
        s, label, _ = score_statement("rate hike hawkish inflation concern")
        assert s > 0 and label == "hawkish"
        s, label, _ = score_statement("rate cut dovish easing stimulus")
        assert s < 0 and label == "dovish"
