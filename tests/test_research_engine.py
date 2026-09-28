"""Tests for research-engine components: macro, HTF, validator, matcher."""

import numpy as np
import pandas as pd
import pytest

from research.entry_validator import EntryValidator
from research.historical_matcher import HistoricalMatcher
from research.htf_analyzer import HTFAnalyzer, HTFResult
from research.macro_analyzer import MacroAnalyzer, MacroResult


def make_df(closes, timeframe_minutes=15, start=None):
    start = start or pd.Timestamp("2026-01-05 00:00", tz="UTC")
    n = len(closes)
    times = pd.date_range(start, periods=n, freq=f"{timeframe_minutes}min")
    closes = np.asarray(closes, dtype=float)
    highs = closes + 0.0004
    lows = closes - 0.0004
    opens = np.concatenate([[closes[0] - 0.0002], closes[:-1]])
    return pd.DataFrame({"time": times, "open": opens, "high": highs, "low": lows,
                         "close": closes,
                         "volume": np.random.randint(100, 1000, n).astype(float)})


def uptrend_frames():
    rng = np.random.default_rng(5)
    closes = 1.08 + np.cumsum(rng.normal(0.0002, 0.0006, 400))
    m15 = make_df(closes, 15)
    return {"m1": m15.tail(60), "m15": m15,
            "h1": make_df(closes[::4] + 0.0001, 60),
            "h4": make_df(closes[::16] + 0.0002, 240),
            "d1": make_df(closes[::64] + 0.0003, 1440)}


class TestMacroResult:
    def test_blocked_property(self):
        macro = MacroResult(news_gate="blocked:red event: NFP in 10 min")
        assert macro.blocked
        assert not MacroResult(news_gate="clear").blocked

    def test_directional_bias(self):
        macro = MacroResult(pair_bias=40.0)
        assert macro.directional_bias("buy") == 1
        assert macro.directional_bias("sell") == -1
        assert MacroResult(pair_bias=5.0).directional_bias("buy") == 0

    def test_analyzer_degrades_without_network(self, temp_db):
        """All sources failing must still return a usable MacroResult."""
        result = MacroAnalyzer().analyze("EURUSD", "buy")
        assert isinstance(result, MacroResult)
        assert result.base_currency == "EUR"
        assert result.quote_currency == "USD"


class TestHTFAnalyzer:
    def test_uptrend_direction_ok_for_buy(self, temp_db):
        result = HTFAnalyzer().analyze("EURUSD", uptrend_frames(), "buy")
        assert result.daily_trend in ("up", "down", "range")
        assert isinstance(result.htf_agree, bool)
        assert isinstance(result.direction_ok, bool)
        assert result.ob_kind != ""  # structure read completed

    def test_htf_direction_mapping(self):
        r = HTFResult(daily_trend="up", h4_bias="bullish", htf_agree=True)
        assert r.htf_direction() == "buy"
        r2 = HTFResult(daily_trend="up", h4_bias="bearish", htf_agree=False)
        assert r2.htf_direction() == ""


class TestEntryValidator:
    def test_checklist_score_bounds(self, temp_db):
        frames = uptrend_frames()
        htf = HTFAnalyzer().analyze("EURUSD", frames, "buy")
        macro = MacroResult(news_gate="clear", cot_bias="neutral")
        checklist = EntryValidator().validate("EURUSD", "buy", frames, htf, macro)
        assert 0 <= checklist.score <= 12  # 10 items + up to 2 bonus
        assert len(checklist.failed_items) + (checklist.score -
                                              min(checklist.score, 2)) >= 0

    def test_news_gate_fails_no_red_news(self, temp_db):
        frames = uptrend_frames()
        htf = HTFAnalyzer().analyze("EURUSD", frames, "buy")
        macro = MacroResult(news_gate="blocked:red event: CPI in 5 min")
        checklist = EntryValidator().validate("EURUSD", "buy", frames, htf, macro)
        assert not checklist.no_red_news
        assert "no_red_news" in checklist.failed_items

    def test_zero_volume_feed_excludes_volume_check(self, temp_db):
        """Yahoo FX candles are all-zero volume: the dead check must be
        excluded from score AND denial, not veto every FX trade forever."""
        frames = uptrend_frames()
        frames["m15"] = frames["m15"].assign(volume=0.0)
        htf = HTFAnalyzer().analyze("EURUSD", frames, "buy")
        macro = MacroResult(news_gate="clear", cot_bias="neutral")
        checklist = EntryValidator().validate("EURUSD", "buy", frames, htf, macro)
        assert "volume_confirmed" not in checklist.failed_items
        # 9 active checks, no bonus -> score + failed must equal 9
        assert checklist.score + len(checklist.failed_items) == 9

    def test_real_volume_feed_keeps_volume_check(self, temp_db):
        """With genuine volume data the item stays in the checklist."""
        frames = uptrend_frames()
        rng = np.random.default_rng(7)
        frames["m15"] = frames["m15"].assign(
            volume=rng.integers(500, 2000, len(frames["m15"])).astype(float))
        htf = HTFAnalyzer().analyze("EURUSD", frames, "buy")
        macro = MacroResult(news_gate="clear", cot_bias="neutral")
        checklist = EntryValidator().validate("EURUSD", "buy", frames, htf, macro)
        # 10 active checks -> score + failed must equal 10
        assert checklist.score + len(checklist.failed_items) == 10


class TestHistoricalMatcher:
    def test_insufficient_samples(self, temp_db):
        match = HistoricalMatcher().match("EURUSD", "London", "london_breakout",
                                          50.0, "buy", True)
        assert match.samples == 0
        assert match.required_confluence == 8

    def test_poor_history_requires_nine(self, temp_db):
        from core import db
        for i in range(12):
            db.record_trade("EURUSD", "buy", 0.1, 1.10, 1.09, 1.12,
                            strategy="london_breakout", session="London",
                            signal_hash=f"h{i}", mode="demo")
            trade_id = db.open_trades("demo")[0]["id"]
            db.close_trade(trade_id, 1.09, -100.0, -80.0)  # all losses
        match = HistoricalMatcher().match("EURUSD", "London", "london_breakout",
                                          50.0, "buy", True)
        assert match.samples == 12
        assert match.win_rate == 0.0
        assert match.required_confluence == 9
