"""Regression tests for the GBPJPY trade #5 wrong-side SL (2026-10-07).

Trade #5: BUY GBPJPY 0.49 lots @ 208.984 with SL 209.203 — 21.9 pips ABOVE
entry. Three layers failed:
  1. ema_trend_rider derived SL from the M15 EMA50, which sits ABOVE price in
     a deep pullback inside an H1 uptrend;
  2. the research fallback reinstated that structural stop with no sanity
     check (Groq's sane SL 208.94 was correctly rejected as TIGHTER, because
     _sl_at_least_as_wide must never widen, but the fallback itself was never
     validated);
  3. the risk gate computed sl_pips with abs() and the paper broker then
     "stopped out" the BUY at the level above entry, booking +$73.09 of fake
     PnL (rr_achieved recorded as +1 on a stop-out).

Also covers the lost pending limit fill: NAS100 #428 (confluence 9,
conviction 90, 3/3 consensus) was approved, deferred to a limit fill, and a
512MB OOM restart silently dropped it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from config import settings
from execution.pip_math import pip_size

RNG = np.random.default_rng(7)


def _ohlc(closes, pip=0.0001):
    closes = np.asarray(closes, dtype=float)
    highs = closes + RNG.uniform(0.5, 1.5, len(closes)) * pip * 2.0
    lows = closes - RNG.uniform(0.5, 1.5, len(closes)) * pip * 2.0
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "time": pd.date_range("2026-10-07", periods=len(closes), freq="15min"),
    })


def _deep_pullback_context():
    """H1 uptrend + deep M15 pullback so M15 EMA50 sits ABOVE price.

    Reproduces the geometry that produced trade #5: the old stop formula
    (e50 * 0.9995 for a buy) lands on the WRONG side of entry, and RSI is
    still in the healthy 40-60 pullback band.
    """
    from strategies.base_strategy import MarketContext

    depth, n_grind, bounce = 0.0025, 55, 0.0004
    h1 = _ohlc(np.linspace(1.10, 1.135, 300))
    base = np.full(210, 1.1300)
    grind = np.linspace(1.1300, 1.1300 - depth, n_grind)
    last_close = 1.1300 - depth + bounce
    last_open = last_close - 2 * bounce  # bullish body = 2*bounce
    m15 = _ohlc(np.r_[base, grind, [last_close]])
    m15.iloc[-1, m15.columns.get_loc("open")] = last_open
    m15.iloc[-1, m15.columns.get_loc("high")] = last_close + 0.3 * bounce
    m15.iloc[-1, m15.columns.get_loc("low")] = last_open - 0.3 * bounce
    ctx = MarketContext(pair="EURUSD", now=pd.Timestamp("2026-10-07 14:00", tz="UTC"),
                        m1=m15, m15=m15, h1=h1, h4=h1, daily=h1)
    return ctx


class TestEmaTrendRiderStopSide:
    """The strategy stop must ALWAYS be on the correct side of entry."""

    def test_buy_stop_below_price_in_deep_pullback(self):
        ctx = _deep_pullback_context()
        from strategies.ema_trend_rider import EMATrendRiderStrategy

        sig = EMATrendRiderStrategy().evaluate(ctx)
        assert sig is not None, "fixture must produce a signal"
        assert sig.direction == "buy"
        assert sig.sl < sig.entry, (
            f"buy SL {sig.sl} must be below entry {sig.entry}")

    def test_sell_stop_above_price_in_deep_rally(self):
        ctx = _deep_pullback_context()
        # mirror every price around the entry zone to make a downtrend rally
        for name in ("m1", "m15", "h1", "h4", "daily"):
            df = getattr(ctx, name)
            for col in ("open", "high", "low", "close"):
                df[col] = 2.26 - df[col]
        from strategies.ema_trend_rider import EMATrendRiderStrategy

        sig = EMATrendRiderStrategy().evaluate(ctx)
        if sig is not None:  # mirrored geometry may flip trend label
            assert sig.direction == "sell"
            assert sig.sl > sig.entry

    def test_reproduces_trade5_geometry_with_old_formula(self):
        """Sanity: with the OLD formula (e50*0.9995) this fixture produced a
        wrong-side stop — i.e. the fixture really covers the bug."""
        from strategies.base_strategy import ema

        ctx = _deep_pullback_context()
        e50 = float(ema(ctx.m15["close"], 50).iloc[-1])
        price = ctx.price()
        assert e50 > price, "fixture precondition: M15 EMA50 above price"
        assert e50 * 0.9995 > price, "old formula would place SL above price"


class TestResearchFallbackSanitized:
    """The research fallback stop must never be wrong-sided or zero."""

    def test_wrong_side_signal_sl_re_anchored_buy(self):
        from research.research_engine import ResearchEngine

        frames = {"m15": _ohlc(np.linspace(1.13, 1.133, 60))}
        sl = ResearchEngine._sanitize_structural_sl(
            "EURUSD", 1.1299, 1.12841, "buy", frames)
        assert sl < 1.12841, f"buy stop {sl} must be below entry"
        dist_pips = (1.12841 - sl) / pip_size("EURUSD")
        assert dist_pips >= 3 * 1.0, "re-anchored stop must clear risk-gate floor"

    def test_wrong_side_signal_sl_re_anchored_sell(self):
        from research.research_engine import ResearchEngine

        frames = {"m15": _ohlc(np.linspace(1.13, 1.127, 60))}
        sl = ResearchEngine._sanitize_structural_sl(
            "EURUSD", 1.1280, 1.12841, "sell", frames)
        assert sl > 1.12841

    def test_correct_side_passes_through(self):
        from research.research_engine import ResearchEngine

        frames = {"m15": _ohlc(np.linspace(1.13, 1.127, 60))}
        assert ResearchEngine._sanitize_structural_sl(
            "EURUSD", 1.1278, 1.12841, "buy", frames) == 1.1278

    def test_zero_stop_re_anchored(self):
        from research.research_engine import ResearchEngine

        frames = {"m15": _ohlc(np.linspace(1.13, 1.127, 60))}
        sl = ResearchEngine._sanitize_structural_sl(
            "EURUSD", 0.0, 1.12841, "buy", frames)
        assert 0 < sl < 1.12841

    def test_groq_sane_sl_still_wins_over_fallback(self):
        """Trade #5 exact numbers: Groq SL 208.94 (correct side) beats any
        structural fallback for a buy at 208.984."""
        from research.research_engine import ResearchEngine

        entry = 208.983994
        groq_sl, signal_sl = 208.94, 209.203343734853
        assert ResearchEngine._sl_at_least_as_wide(
            groq_sl, signal_sl, entry, "buy") is False, \
            "groq stop must be rejected as tighter (signal stop wrong-sided)"
        frames = {"m15": _ohlc(np.linspace(208.5, 209.2, 60),
                               pip=pip_size("GBPJPY"))}
        fallback = ResearchEngine._sanitize_structural_sl(
            "GBPJPY", signal_sl, entry, "buy", frames)
        assert fallback < entry, "fallback must be re-anchored below entry"


class TestRiskGateWrongSideSl:
    """Last defense before sizing: abs() must not bless a wrong-side stop."""

    @staticmethod
    def _rm():
        from risk.risk_manager import RiskManager
        return RiskManager()

    def test_trade5_geometry_rejected(self, temp_db):
        d = self._rm().evaluate("GBPJPY", "buy", "ema_trend_rider",
                                208.983994, 209.203343734853, 209.058,
                                balance=10400.0, equity=10400.0,
                                used_margin=0.0, confluence=8, mode="demo")
        assert d.approved is False
        assert "wrong-side sl" in d.reason

    def test_sell_wrong_side_rejected(self, temp_db):
        d = self._rm().evaluate("EURUSD", "sell", "ema_trend_rider",
                                1.12841, 1.1280, 1.1320,
                                balance=10000.0, equity=10000.0,
                                used_margin=0.0, confluence=8, mode="demo")
        assert d.approved is False

    def test_correct_side_still_approved(self, temp_db):
        d = self._rm().evaluate("GBPJPY", "buy", "ema_trend_rider",
                                208.984, 208.75, 209.20,
                                balance=10400.0, equity=10400.0,
                                used_margin=0.0, confluence=8, mode="demo")
        assert d.approved is True


class TestPaperBrokerRejectsBadStop:
    """The broker must refuse orders whose stop would fire immediately."""

    def test_buy_with_sl_above_market_rejected(self, temp_db):
        from execution.paper_broker import PaperBroker

        class FakeData:
            def get_candles(self, pair, tf, count):
                class DF:
                    df = pd.DataFrame({"close": [208.95, 208.96, 208.98]})
                return DF()

        broker = PaperBroker(data_engine=FakeData())
        with pytest.raises(ValueError):
            broker.market_order("GBPJPY", "buy", 0.49, 209.2033, 209.058, "t")

    def test_sell_with_sl_below_market_rejected(self, temp_db):
        from execution.paper_broker import PaperBroker

        class FakeData:
            def get_candles(self, pair, tf, count):
                class DF:
                    df = pd.DataFrame({"close": [1.1285, 1.1284, 1.1283]})
                return DF()

        broker = PaperBroker(data_engine=FakeData())
        with pytest.raises(ValueError):
            broker.market_order("EURUSD", "sell", 0.10, 1.1280, 1.1320, "t")

    def test_valid_order_still_fills(self, temp_db):
        from execution.paper_broker import PaperBroker

        class FakeData:
            def get_candles(self, pair, tf, count):
                class DF:
                    df = pd.DataFrame({"close": [208.95, 208.96, 208.98]})
                return DF()

        broker = PaperBroker(data_engine=FakeData())
        ticket = broker.market_order("GBPJPY", "buy", 0.49, 208.80, 209.20, "t")
        assert ticket > 0


class TestDefensiveStopClose:
    """If a wrong-side stop somehow exists on an open trade, closing it must
    cap the exit at breakeven instead of booking profit."""

    @staticmethod
    def _trade_row(temp_db):
        from core import db
        trade_id = db.record_trade(
            pair="GBPJPY", direction="buy", lots=0.49, entry=208.983994,
            sl=209.203343734853, tp=209.058, strategy="ema_trend_rider",
            confluence=8, conviction=80, mode="demo")
        rows = [t for t in db.open_trades(mode="demo") if t["id"] == trade_id]
        assert rows
        return rows[0]

    @staticmethod
    def _manager(bid=209.2033):
        from execution.trade_manager import TradeManager

        class FakeBroker:
            def __init__(self):
                self.closed = []

            def close_matching(self, pair, direction):
                self.closed.append((pair, direction))
                return 4242

            def close_position(self, ticket, lots=None):
                return True

            def modify_sl_tp(self, ticket, sl, tp):
                return True

            def tick(self, pair):
                return {"bid": bid, "ask": bid + 0.03, "spread_pips": 2.5}

        return TradeManager(broker=FakeBroker(), notifier=None)

    def test_wrong_side_stop_closes_at_breakeven_cap(self, temp_db):
        from core import db

        trade = self._trade_row(temp_db)
        tm = self._manager()
        # price 209.2033: "hits" the bogus stop; old code closed AT the stop
        # => +21.9 pips. New code must cap at ~breakeven.
        tm.manage_trade(trade)
        closed = [t for t in db.closed_trades(limit=5, mode="demo")
                  if t["id"] == trade["id"]]
        assert closed, "trade must be closed"
        row = closed[0]
        assert abs(row["pips"]) <= 1.0, (
            f"exit must be capped near breakeven, got {row['pips']} pips")
        audits = db.recent_audit(limit=20)
        assert any(a.get("action") == "wrong_side_stop_close" for a in audits)

    def test_correct_side_stop_still_closes_at_stop(self, temp_db):
        from core import db

        trade = self._trade_row(temp_db)
        trade = dict(trade)
        trade["sl"] = 208.75  # sane stop below entry
        tm = self._manager(bid=208.70)  # below stop, below tp
        # price below stop -> normal SL close at the stop price
        tm.manage_trade(trade)
        closed = [t for t in db.closed_trades(limit=5, mode="demo")
                  if t["id"] == trade["id"]]
        assert closed
        assert closed[0]["pips"] < 0, "a real stop-out should be a loss"


class TestPendingFillPersistence:
    """Deferred limit fills must survive a restart and expire on TTL."""

    @staticmethod
    def _verdict(entry=31472.75, sl=31420.0, tp1=31520.0):
        from research.research_engine import ResearchVerdict
        return ResearchVerdict(
            pair="NAS100", direction="buy", strategy="ema_trend_rider",
            approved=True, reason="approved: confluence 9, conviction 90, "
            "consensus 3/3", entry=entry, sl=sl, tp1=tp1,
            tp2=entry + 2 * (entry - sl), tp3=entry + 3 * (entry - sl),
            confluence=9, conviction=90, consensus_size=1.0,
            regime="trending")

    @staticmethod
    def _engine():
        from execution.execution_engine import ExecutionEngine

        eng = ExecutionEngine(mode="demo", notifier=None)
        return eng

    def test_deferred_fill_persisted(self, temp_db):
        from core import db

        eng = self._engine()
        eng._save_pending_fill(self._verdict(), lots=0.5)
        keys = [r["key"] for r in db.all_state()
                if r["key"].startswith("pending_fill_")]
        assert len(keys) == 1

    def test_service_fills_when_price_returns_to_zone(self, temp_db):
        from core import db

        eng = self._engine()
        v = self._verdict()
        eng._save_pending_fill(v, lots=0.5)

        class FakeBroker:
            def market_order(self, pair, direction, lots, sl, tp, comment):
                return 9001

        eng._market_fill = lambda broker, verdict, lots: {
            "ticket": 9001, "price": verdict.entry, "session": "NewYork"}
        eng.active_broker = lambda: FakeBroker()
        eng._current_price = lambda pair: v.entry - 1.0  # at/below zone
        eng._notify = lambda text: None

        executed = eng.service_pending_fills()
        assert executed == 1
        trades = db.closed_trades(limit=10, mode="demo") + db.open_trades(mode="demo")
        assert any(t["pair"] == "NAS100" for t in trades), \
            "re-armed fill must create a trade row"
        keys = [r["key"] for r in db.all_state()
                if r["key"].startswith("pending_fill_")]
        assert not keys, "consumed pending fill must be deleted"

    def test_expired_fill_dropped(self, temp_db):
        from core import db

        eng = self._engine()
        eng._save_pending_fill(self._verdict(), lots=0.5)
        # backdate beyond the TTL
        old = (datetime.now(timezone.utc)
               - timedelta(minutes=settings.PENDING_LIMIT_TTL_MIN + 10)
               ).isoformat()
        keys = [r["key"] for r in db.all_state()
                if r["key"].startswith("pending_fill_")]
        assert keys
        import json
        payload = json.loads(db.get_state(keys[0]))
        payload["created_at"] = old
        db.set_state(keys[0], json.dumps(payload))

        eng._current_price = lambda pair: 30000.0  # never at zone anyway
        executed = eng.service_pending_fills()
        assert executed == 0
        keys_after = [r["key"] for r in db.all_state()
                      if r["key"].startswith("pending_fill_")]
        assert not keys_after, "expired pending fill must be deleted"

    def test_far_price_stays_armed(self, temp_db):
        from core import db

        eng = self._engine()
        eng._save_pending_fill(self._verdict(), lots=0.5)
        eng._current_price = lambda pair: 32000.0  # far above the buy zone
        executed = eng.service_pending_fills()
        assert executed == 0
        keys = [r["key"] for r in db.all_state()
                if r["key"].startswith("pending_fill_")]
        assert keys, "pending fill must stay armed while inside TTL"


class TestHealthMemory:
    def test_mem_snapshot_shape(self):
        import health
        snap = health._mem_snapshot()
        assert "rss_mb" in snap and "limit_mb" in snap and "pct" in snap
        assert 0 < snap["pct"] < 200
