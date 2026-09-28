"""Tests for the CRITICAL FIX: feed failover, breaker escalation, AutoRecovery."""

import sys
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest


def _candles(count=60, timeframe_min=15, age_min=1.0):
    """Valid-looking candle frame ending `age_min` minutes ago (varied closes)."""
    import numpy as np
    now = datetime.now(timezone.utc) - timedelta(minutes=age_min)
    ts = pd.date_range(end=now, periods=count, freq=f"{timeframe_min}min", tz="UTC")
    close = np.array([1.1000 + 0.0005 * (i % 5) for i in range(count)])
    open_ = close - 0.0002
    return pd.DataFrame({
        "time": ts,
        "open": open_,
        "high": np.maximum(open_, close) + 0.0008,
        "low": np.minimum(open_, close) - 0.0008,
        "close": close,
        "volume": 100.0,
    })


# ============================================================
# Fix 1: feed chain (MT5 primary guarded, yfinance, no AV)
# ============================================================

class TestFeedChain:
    def _engine(self, feeds):
        from data.market_data_engine import MarketDataEngine
        return MarketDataEngine(feeds=feeds, failover_deadline_sec=5.0)

    def test_no_alpha_vantage_feed_in_chain(self):
        import data.market_data_engine as mde
        assert not hasattr(mde, "AlphaVantageFeed"), "AlphaVantageFeed must be removed"
        assert not hasattr(mde, "AV_SYMBOLS")

    def test_default_chain_order_mt5_first(self, temp_db):
        from data.market_data_engine import MarketDataEngine
        eng = MarketDataEngine()
        names = [f.name for f in eng.feeds]
        assert names[0] == "mt5"
        assert names == ["mt5", "yfinance", "twelve_data"]

    def test_failover_to_secondary_when_primary_fails(self, temp_db):
        primary = MagicMock(); primary.name = "mt5"; primary.configured = True
        primary.fetch.side_effect = Exception("terminal dead")
        secondary = MagicMock(); secondary.name = "yfinance"; secondary.configured = True
        secondary.fetch.return_value = _candles()

        eng = self._engine([primary, secondary])
        candle = eng.get_candles("EURUSD", 15, 60)
        assert candle.source == "yfinance"
        assert eng.active_feed == "yfinance"
        assert eng.consecutive_failures == 0  # recovered by failover

    def test_validation_rejects_bad_payload_and_fails_over(self, temp_db):
        bad = MagicMock(); bad.name = "mt5"; bad.configured = True
        bad.fetch.return_value = _candles()
        bad.fetch.return_value.loc[bad.fetch.return_value.index[-1], "high"] = 0.0  # OHLC violation
        good = MagicMock(); good.name = "yfinance"; good.configured = True
        good.fetch.return_value = _candles()

        eng = self._engine([bad, good])
        candle = eng.get_candles("EURUSD", 15, 60)
        assert candle.source == "yfinance"

    def test_feed_status_shape(self, temp_db):
        from data.market_data_engine import MarketDataEngine
        eng = MarketDataEngine(feeds=[])
        status = eng.get_feed_status()
        assert status["active_feed"] == "none"
        assert status["last_update"] is None
        assert status["consecutive_failures"] == 0
        assert status["mt5_connected"] is False
        assert status["feeds_configured"] == []

    def test_feed_status_after_success(self, temp_db):
        feed = MagicMock(); feed.name = "yfinance"; feed.configured = True
        feed.fetch.return_value = _candles()
        eng = self._engine([feed])
        eng.get_candles("EURUSD", 15, 60)
        status = eng.get_feed_status()
        assert status["active_feed"] == "yfinance"
        assert status["last_update"] is not None

    def test_feed_switch_audited(self, temp_db):
        from core import db
        primary = MagicMock(); primary.name = "mt5"; primary.configured = True
        primary.fetch.side_effect = Exception("down")
        secondary = MagicMock(); secondary.name = "yfinance"; secondary.configured = True
        secondary.fetch.return_value = _candles()
        eng = self._engine([primary, secondary])
        eng.get_candles("EURUSD", 15, 60)
        with db.engine.begin() as conn:
            from sqlalchemy import text
            rows = conn.execute(text(
                "SELECT action FROM audit_log WHERE category='data' "
                "AND action='feed_switch'")).all()
        assert rows, "feed switch must be audited"

    def test_unconfigured_feed_skipped(self, temp_db):
        dead = MagicMock(); dead.name = "mt5"; dead.configured = False
        good = MagicMock(); good.name = "yfinance"; good.configured = True
        good.fetch.return_value = _candles()
        eng = self._engine([dead, good])
        assert eng.get_candles("EURUSD", 15, 60).source == "yfinance"

    def test_all_feeds_dead_raises(self, temp_db):
        dead = MagicMock(); dead.name = "mt5"; dead.configured = True
        dead.fetch.side_effect = Exception("x")
        eng = self._engine([dead])
        with pytest.raises(Exception):
            eng.get_candles("EURUSD", 15, 60)

    def test_stale_payload_rejected_outside_weekend(self, temp_db):
        feed = MagicMock(); feed.name = "yfinance"; feed.configured = True
        feed.fetch.return_value = _candles(age_min=600)  # 10h old
        eng = self._engine([feed])
        with pytest.raises(Exception):
            eng.get_candles("EURUSD", 15, 60)

    def test_weekend_staleness_tolerated(self, temp_db):
        import data.market_data_engine as mde
        fake_now = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)  # Saturday
        feed = MagicMock(); feed.name = "yfinance"; feed.configured = True
        feed.fetch.return_value = _candles(age_min=600)
        eng = self._engine([feed])
        with patch.object(mde, "fx_market_closed", return_value=True):
            candle = eng.get_candles("EURUSD", 15, 60)
            assert candle.source == "yfinance"

    def test_mt5_feed_guarded_import(self, temp_db):
        """On non-Windows hosts (this test env) MT5Feed.configured is False."""
        from data.market_data_engine import MT5Feed
        feed = MT5Feed()
        try:
            import MetaTrader5  # noqa: F401
            installed = True
        except ImportError:
            installed = False
        assert feed.configured == installed

    def test_mt5_fetch_sets_volume_column(self, temp_db):
        """MT5 raw rates use tick_volume; ensure the rename stays intact."""
        from data.market_data_engine import MT5Feed
        feed = MT5Feed()
        mt5_mod = MagicMock()
        mt5_mod.TIMEFRAME_M15 = 15
        rates = pd.DataFrame({
            "time": [1, 2], "open": [1.0, 1.0], "high": [1.1, 1.1],
            "low": [0.9, 0.9], "close": [1.05, 1.05], "tick_volume": [10, 20],
            "spread": [1, 1], "real_volume": [0, 0],
        })
        mt5_mod.copy_rates_from_pos.return_value = rates.to_records()
        feed.mt5 = mt5_mod
        feed._connected = True
        df = feed.fetch("EURUSD", 15, 30)
        assert "volume" in df.columns
        assert list(df["volume"]) == [10.0, 20.0]


# ============================================================
# Fix 2: circuit breaker escalation / dedup
# ============================================================

class TestBreakerEscalation:
    def _brk(self, sent):
        from risk.circuit_breakers import CircuitBreakers
        return CircuitBreakers(notifier=sent.append)

    def test_first_trigger_alerts_once(self, temp_db):
        sent = []
        brk = self._brk(sent)
        assert brk.trigger("data_stale", "feed age 999s") is True
        assert len(sent) == 1
        assert "data_stale" in sent[0]
        # re-trigger within seconds: silent, returns False
        assert brk.trigger("data_stale", "feed age 999s") is False
        assert len(sent) == 1

    def test_escalation_schedule(self, temp_db):
        sent = []
        brk = self._brk(sent)
        brk.trigger("data_stale", "stale")
        entry = brk.active_breakers["data_stale"]
        # age the breaker artificially: 1h -> alert
        entry["first_trigger"] -= 3600 + 1
        assert brk.trigger("data_stale", "stale") is True
        assert len(sent) == 2
        # 30 min later (total 1.5h < 3h threshold): silent
        brk.active_breakers["data_stale"]["first_trigger"] -= 1800
        assert brk.trigger("data_stale", "stale") is False
        assert len(sent) == 2
        # push past 3h total: alert
        brk.active_breakers["data_stale"]["first_trigger"] -= 7200
        assert brk.trigger("data_stale", "stale") is True
        assert len(sent) == 3
        assert "STILL ACTIVE" in sent[2]

    def test_max_six_alerts_in_24h(self, temp_db):
        sent = []
        brk = self._brk(sent)
        for hours in (0, 1, 3, 6, 12, 24, 25, 30, 47):
            brk.trigger("data_stale", "stale")
            brk.active_breakers["data_stale"]["first_trigger"] = time.time() - hours * 3600
        # 0h + 1h + 3h + 6h + 12h + 24h = 6 alerts in the first 24h...
        assert brk.active_breakers["data_stale"]["alert_count"] == 6
        # ...and the 25h+ re-evaluations within the same day stay silent
        assert len(sent) == 6
        # a day after the last alert, exactly one more is allowed
        brk.active_breakers["data_stale"]["last_alert"] -= 86400 + 1
        assert brk.trigger("data_stale", "stale") is True
        assert len(sent) == 7

    def test_resolve_sends_one_confirmation(self, temp_db):
        sent = []
        brk = self._brk(sent)
        brk.trigger("data_stale", "stale")
        brk.resolve("data_stale")
        assert len(sent) == 2
        assert "RESOLVED" in sent[1] or "resolved" in sent[1].lower()
        # resolving again is a no-op (no duplicate confirmation)
        brk.resolve("data_stale")
        assert len(sent) == 2
        # resolving a breaker that never fired alerts nothing
        brk.resolve("groq_down")
        assert len(sent) == 2

    def test_refire_after_resolve_realerts(self, temp_db):
        sent = []
        brk = self._brk(sent)
        brk.trigger("data_stale", "stale")
        brk.resolve("data_stale")
        assert brk.trigger("data_stale", "stale again") is True
        assert len(sent) == 3
        assert "CIRCUIT BREAKER" in sent[2]

    def test_is_active_and_has_active_breakers(self, temp_db):
        brk = self._brk([])
        assert brk.is_active("data_stale") is False
        assert brk.has_active_breakers() is False
        brk.trigger("data_stale", "stale")
        assert brk.is_active("data_stale") is True
        assert brk.has_active_breakers() is True
        brk.resolve("data_stale")
        assert brk.is_active("data_stale") is False

    def test_has_active_breakers_reads_db(self, temp_db):
        from core import db
        brk = self._brk([])
        db.log_breaker("daily_loss", "test", "halt")
        # not in the in-memory map but present as unresolved in DB
        assert brk.has_active_breakers() is True

    def test_state_persists_across_restart(self, temp_db):
        sent = []
        brk = self._brk(sent)
        brk.trigger("data_stale", "stale")
        # "restart": fresh instance loads persisted schedule state
        from risk.circuit_breakers import CircuitBreakers
        brk2 = CircuitBreakers(notifier=sent.append)
        assert brk2.is_active("data_stale") is True
        assert brk2.trigger("data_stale", "stale") is False  # no re-spam
        assert len(sent) == 1
        # escalation timing survives too
        brk2.active_breakers["data_stale"]["first_trigger"] -= 3600 + 1
        assert brk2.trigger("data_stale", "stale") is True
        assert len(sent) == 2

    def test_evaluate_resolves_and_confirms(self, temp_db):
        from core import db
        sent = []
        brk = self._brk(sent)
        db.log_breaker("data_stale", "feed dead", "halt")
        brk.active_breakers["data_stale"] = {
            "first_trigger": time.time() - 120, "last_alert": time.time(),
            "alert_count": 1, "resolved": False, "reason": "feed dead",
        }
        state = brk.evaluate(feed_age_sec=10, mt5_ok=True, db_ok=True)
        # the same pass that clears the breaker aggregates first, so verify
        # the DB row is resolved and the confirmation went out
        assert "data_stale" not in db.unresolved_breakers()
        assert any("RESOLVED" in s or "resolved" in s.lower() for s in sent)
        state2 = brk.evaluate(feed_age_sec=10, mt5_ok=True, db_ok=True)
        assert not state2.halted

    def test_weekend_does_not_fire_data_stale(self, temp_db):
        brk = self._brk([])
        brk.evaluate(feed_age_sec=99999, mt5_ok=True, db_ok=True,
                     now_utc_weekday=5, now_utc_hour=12)
        assert brk.is_active("data_stale") is False

    def test_resume_all_clears_everything(self, temp_db):
        from core import db
        brk = self._brk([])
        brk.trigger("data_stale", "a")
        brk.trigger("groq_down", "b")
        brk.resume_all()
        assert db.unresolved_breakers() == []
        assert brk.has_active_breakers() is False


# ============================================================
# Fix 3: AutoRecovery
# ============================================================

class TestAutoRecovery:
    def _rec(self, data, brk, sent):
        from core.autorecovery import AutoRecovery
        return AutoRecovery(data_engine=data, breakers=brk, notifier=sent.append)

    def _data_ok(self):
        d = MagicMock()
        d.get_feed_status.return_value = {"active_feed": "yfinance",
                                          "consecutive_failures": 0,
                                          "mt5_connected": False}
        d.feeds = []
        return d

    def test_resolves_data_stale_when_feed_recovers(self, temp_db):
        from core import db
        sent = []
        brk = MagicMock()
        brk.is_active.return_value = True
        brk.has_active_breakers.return_value = True
        rec = self._rec(self._data_ok(), brk, sent)
        rec.check_and_recover()
        brk.resolve.assert_called_once_with("data_stale")

    def test_noop_when_feed_down(self, temp_db):
        sent = []
        brk = MagicMock()
        brk.is_active.return_value = True
        data = MagicMock()
        data.get_feed_status.return_value = {"active_feed": "none",
                                             "consecutive_failures": 3,
                                             "mt5_connected": False}
        data.feeds = []
        rec = self._rec(data, brk, sent)
        rec.check_and_recover()
        brk.resolve.assert_not_called()

    def test_does_not_resume_over_operator_pause(self, temp_db):
        from core import db
        db.set_state("trading_paused", "1")
        db.set_state("running", "0")
        brk = MagicMock()
        brk.has_active_breakers.return_value = False
        rec = self._rec(self._data_ok(), brk, sent=[])
        rec.check_and_recover()
        assert db.get_state("running") == "0"  # pause respected
        db.set_state("trading_paused", "0")

    def test_auto_resumes_when_no_breakers(self, temp_db):
        from core import db
        db.set_state("running", "0")
        db.set_state("trading_paused", "0")
        brk = MagicMock()
        brk.has_active_breakers.return_value = False
        rec = self._rec(self._data_ok(), brk, [])
        rec.check_and_recover()
        assert db.get_state("running") == "1"
        with db.engine.begin() as conn:
            from sqlalchemy import text
            rows = conn.execute(text(
                "SELECT action FROM audit_log WHERE action='auto_resume'")).all()
        assert rows, "auto-resume must be audited"

    def test_no_resume_when_breakers_active(self, temp_db):
        from core import db
        db.set_state("running", "0")
        db.set_state("trading_paused", "0")
        brk = MagicMock()
        brk.has_active_breakers.return_value = True
        rec = self._rec(self._data_ok(), brk, [])
        rec.check_and_recover()
        assert db.get_state("running") == "0"

    def test_mt5_reconnect_attempted_when_dead(self, temp_db):
        rec = self._rec(self._data_ok(), MagicMock(), [])
        feed = MagicMock(); feed.name = "mt5"; feed.configured = True
        feed.connected.return_value = False
        feed.reconnect.return_value = True
        rec.data.feeds = [feed]
        rec.check_and_recover()
        feed.reconnect.assert_called_once()

    def test_mt5_reconnect_throttled(self, temp_db):
        rec = self._rec(self._data_ok(), MagicMock(), [])
        feed = MagicMock(); feed.name = "mt5"; feed.configured = True
        feed.connected.return_value = False
        feed.reconnect.return_value = False
        rec.data.feeds = [feed]
        rec.check_and_recover()
        rec.check_and_recover()  # within throttle window
        assert feed.reconnect.call_count == 1

    def test_skips_mt5_when_package_absent(self, temp_db):
        rec = self._rec(self._data_ok(), MagicMock(), [])
        feed = MagicMock(); feed.name = "mt5"; feed.configured = False
        rec.data.feeds = [feed]
        rec.check_and_recover()  # must not raise
        feed.reconnect.assert_not_called()

    def test_feed_boot_failure_healed_on_recovery(self, temp_db):
        """A boot that started with all feeds dead re-enables trading once
        any feed produces valid candles (feed-only failure)."""
        from core import db
        db.set_state("feed_boot_failure", "1")
        db.set_state("observation_mode", "1")
        db.set_state("running", "0")
        sent = []
        brk = MagicMock()
        brk.has_active_breakers.return_value = False
        rec = self._rec(self._data_ok(), brk, sent)
        rec.check_and_recover()
        assert db.get_state("feed_boot_failure") == "0"
        assert db.get_state("observation_mode") == "0"
        assert db.get_state("running") == "1"
        assert any("auto-resumed" in s.lower() or "resumed" in s.lower()
                   for s in sent)

    def test_feed_outage_edge_logging(self, temp_db):
        from core import db
        rec = self._rec(self._data_ok(), MagicMock(), [])
        rec.check_and_recover()  # ok
        data = MagicMock()
        data.get_feed_status.return_value = {"active_feed": "none",
                                             "consecutive_failures": 2,
                                             "mt5_connected": False}
        data.feeds = []
        rec.data = data
        rec.check_and_recover()  # outage edge
        rec.check_and_recover()  # steady state: no new edge
        with db.engine.begin() as conn:
            from sqlalchemy import text
            rows = conn.execute(text(
                "SELECT action FROM audit_log WHERE action='feed_outage'")).all()
        assert len(rows) == 1

    def test_monitor_loop_runs(self, temp_db):
        import asyncio
        from core.autorecovery import AutoRecovery
        rec = AutoRecovery(self._data_ok(), MagicMock(), None, interval_sec=1)
        rec.check_and_recover = MagicMock()
        async def run():
            task = asyncio.create_task(rec.monitor())
            await asyncio.sleep(1.6)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        asyncio.run(run())
        assert rec.check_and_recover.call_count >= 1
