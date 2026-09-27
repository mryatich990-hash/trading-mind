"""Tests for execution engine and trade manager (fake broker)."""

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from execution.execution_engine import ExecutionEngine, _hash_signal, _pip
from execution.trade_manager import TradeManager
from research.research_engine import ResearchVerdict


def _verdict(pair="EURUSD", direction="buy", approved=True, entry=1.1000,
             sl=1.0980, tp=1.1060, strategy="london_breakout"):
    v = ResearchVerdict(pair=pair, direction=direction, strategy=strategy,
                        approved=approved, reason="test", entry=entry, sl=sl,
                        tp1=tp, tp2=entry + 2 * (entry - sl),
                        tp3=entry + 3 * (entry - sl), conviction=80,
                        consensus_size=1.0, confluence=8)
    return v


class FakeBroker:
    """In-memory broker stub satisfying every call the engines make."""

    name = "fake"

    def __init__(self):
        self.orders = []
        self.closes = []
        self.sl_moves = []
        self._price = 1.1000

    def available(self):
        return True

    def healthy(self):
        return True

    def account(self):
        return {"balance": 10000.0, "equity": 10000.0, "margin_used": 0.0,
                "margin_free": 10000.0, "margin_level": 1000.0, "currency": "USD"}

    def tick(self, pair):
        return {"bid": self._price - 0.00005, "ask": self._price + 0.00005,
                "time": 0, "pair": pair}

    def swap_rates(self, pair):
        return -0.5, -0.8

    def market_order(self, pair, direction, lots, sl, tp, comment=""):
        self.orders.append((pair, direction, lots, sl, tp))
        return 424242

    def close_position(self, ticket, lots=None):
        self.closes.append((ticket, lots))
        return True

    def modify_sl_tp(self, ticket, sl, tp):
        self.sl_moves.append((ticket, sl, tp))
        return True


@pytest.fixture()
def engine(temp_db):
    broker = FakeBroker()
    exe = ExecutionEngine(mt5=broker, oanda=MagicMock(), notifier=None)
    exe.oanda.available = MagicMock(return_value=False)
    return exe, broker


class TestExecutionEngine:
    def test_execute_records_trade(self, engine):
        exe, broker = engine
        trade_id = exe.execute(_verdict(), Decimal("0.10"), 10000.0, 8)
        assert trade_id is not None
        assert len(broker.orders) == 1
        pair, direction, lots, sl, tp = broker.orders[0]
        assert pair == "EURUSD" and direction == "buy"
        assert lots == pytest.approx(0.10)

    def test_duplicate_signal_suppressed(self, engine):
        exe, _ = engine
        v = _verdict()
        exe.execute(v, Decimal("0.10"), 10000.0, 8)
        assert exe.execute(v, Decimal("0.10"), 10000.0, 8) is None

    def test_unapproved_verdict_ignored(self, engine):
        exe, broker = engine
        assert exe.execute(_verdict(approved=False), Decimal("0.10"),
                           10000.0, 8) is None
        assert not broker.orders

    def test_no_broker_no_execution(self, temp_db):
        exe = ExecutionEngine(mt5=MagicMock(), oanda=MagicMock())
        exe.mt5.available = MagicMock(return_value=False)
        exe.oanda.available = MagicMock(return_value=False)
        assert exe.execute(_verdict(), Decimal("0.10"), 10000.0, 8) is None

    def test_pip_table(self):
        assert _pip("EURUSD") == 0.0001
        assert _pip("USDJPY") == 0.01
        assert _pip("XAUUSD") == 0.1

    def test_hash_signal_stable(self):
        v = _verdict()
        assert _hash_signal(v) == _hash_signal(_verdict())


class TestTradeManager:
    def _tm(self, broker, temp_db):
        return TradeManager(broker=broker, notifier=None)

    def _open_trade(self, **overrides):
        from core import db
        db.record_trade("EURUSD", "buy", 0.10, 1.1000, 1.0980, 1.1060,
                        strategy="london_breakout", session="London",
                        signal_hash=overrides.get("hash", "tm1"), mode="demo")
        return db.open_trades("demo")[0]

    def test_tp1_partial_and_breakeven(self, temp_db):
        broker = FakeBroker()
        broker._price = 1.1000 + 1.0010 * 1.0  # ~1R above entry
        tm = self._tm(broker, temp_db)
        trade = self._open_trade()
        trade = dict(trade)
        trade["entry_price"] = 1.1000
        trade["sl"] = 1.0980
        tm.manage_trade(trade)
        assert any(c[1] is not None for c in broker.closes)   # partial close
        assert broker.sl_moves and broker.sl_moves[0][1] >= 1.1000  # BE+2pips

    def test_sl_hit_closes(self, temp_db):
        broker = FakeBroker()
        broker._price = 1.0970  # below SL
        tm = self._tm(broker, temp_db)
        trade = self._open_trade()
        tm.manage_trade(dict(trade))
        assert len(broker.closes) == 1

    def test_close_records_pnl(self, temp_db):
        from core import db
        broker = FakeBroker()
        tm = self._tm(broker, temp_db)
        trade = self._open_trade()
        tm.close(dict(trade), 1.1040, "TP1")
        rows = db.closed_trades(limit=1)
        assert rows and rows[0]["status"] == "closed"
        assert float(rows[0]["pnl_usd"]) > 0

    def test_trail_never_moves_against(self, temp_db):
        broker = FakeBroker()
        tm = self._tm(broker, temp_db)
        trade = dict(self._open_trade())
        trade["sl"] = 1.0995  # already trailed up
        tm._apply_trail(trade, 1.1010, 15.0)  # would trail to 1.1005 < 1.0995? no
        # 15 pips below price 1.1010 = 1.0995, not above current -> no move
        moves = [m for m in broker.sl_moves if m[0] == int(trade["id"])]
        assert all(m[1] >= trade["sl"] for m in moves)

    def test_invalidation_closes_immediately(self, temp_db):
        from core import db
        broker = FakeBroker()
        broker._price = 1.0985
        tm = self._tm(broker, temp_db)
        trade = self._open_trade()
        db._WRITE_LOCK and None
        import json as _json
        from sqlalchemy import text as sqltext
        with db.engine.begin() as conn:
            conn.execute(sqltext("UPDATE trades SET features_json = :f WHERE id = :i"),
                         {"f": _json.dumps({"invalidation": 1.0990}),
                          "i": int(trade["id"])})
        tm.manage_trade(dict(trade))
        assert broker.closes  # closed on invalidation touch


class TestGracefulShutdown:
    def test_request_idempotent(self):
        from core.event_bus import bus
        from core.graceful_shutdown import GracefulShutdown

        async def scenario():
            gs = GracefulShutdown(bus)
            await asyncio.sleep(0)
            gs.request_shutdown()
            gs.request_shutdown()
            await asyncio.wait_for(gs.wait(), timeout=1)
            return True

        assert asyncio.run(scenario())

    def test_shutdown_closes_trades(self, temp_db):
        from core import db
        from core.event_bus import bus
        from core.graceful_shutdown import GracefulShutdown

        db.record_trade("EURUSD", "buy", 0.1, 1.1000, 1.0980, 1.1060,
                        strategy="s", signal_hash="sd1", mode="demo")
        open_trade = db.open_trades("demo")[0]
        alerts = []

        async def scenario():
            gs = GracefulShutdown(bus, notifier=alerts.append)

            async def close_all():
                for t in db.open_trades():
                    db.close_trade(int(t["id"]), 1.1040, 40.0, 40.0)
                    return 1
                return 0

            await gs.run(close_all)
            return db.open_trades()

        remaining = asyncio.run(scenario())
        assert len(remaining) == 0
        assert alerts and "shutting down" in alerts[0].lower()
        assert float(open_trade["lots"]) == 0.1  # sanity: original row existed
