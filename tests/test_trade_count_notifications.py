"""Trade-count lines in fill/close notifications (db.trade_counts + alert text)."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from core import db
from execution.execution_engine import ExecutionEngine
from execution.trade_manager import TradeManager
from research.research_engine import ResearchVerdict


class FakeBroker:
    name = "fake"

    def available(self):
        return True

    def healthy(self):
        return True

    def account(self):
        return {"balance": 10000.0, "equity": 10000.0, "margin_used": 0.0,
                "margin_free": 10000.0, "margin_level": 1000.0, "currency": "USD"}

    def tick(self, pair):
        return {"bid": 1.09995, "ask": 1.10005, "time": 0, "pair": pair}

    def market_order(self, pair, direction, lots, sl, tp, comment=""):
        return 424242

    def close_position(self, ticket, lots=None):
        return True

    def modify_sl_tp(self, ticket, sl, tp):
        return True


def _verdict(pair="EURUSD", direction="buy", entry=1.1000, sl=1.0980, tp=1.1060,
             strategy="london_breakout", signal_hash_suffix="a"):
    return ResearchVerdict(pair=pair, direction=direction, strategy=strategy,
                           approved=True, reason="test", entry=entry, sl=sl,
                           tp1=tp, tp2=entry + 2 * (entry - sl),
                           tp3=entry + 3 * (entry - sl), conviction=80,
                           consensus_size=1.0, confluence=8)


class TestTradeCounts:
    def test_trade_counts_shape(self, temp_db):
        db.record_trade("EURUSD", "buy", 0.10, 1.1000, 1.0980, 1.1060,
                        strategy="s", signal_hash="tc1", mode="demo")
        counts = db.trade_counts("demo")
        assert counts["today"] >= 1
        assert counts["all_time"] >= counts["today"]

    def test_trade_counts_no_mode_filter(self, temp_db):
        db.record_trade("EURUSD", "buy", 0.10, 1.1000, 1.0980, 1.1060,
                        strategy="s", signal_hash="tc2", mode="demo")
        counts = db.trade_counts()
        assert counts["today"] >= 1

    def test_fill_alert_includes_trade_number(self, temp_db):
        alerts = []
        exe = ExecutionEngine(mt5=FakeBroker(), oanda=MagicMock(),
                              notifier=alerts.append, mode="demo")
        exe.oanda.available = MagicMock(return_value=False)
        trade_id = exe.execute(_verdict(), Decimal("0.10"), 10000.0, 8)
        assert trade_id is not None
        assert len(alerts) == 1
        assert f"Trade #{trade_id}" in alerts[0]
        assert "today)" in alerts[0]

    def test_close_alert_includes_trade_number(self, temp_db):
        alerts = []
        broker = FakeBroker()
        tm = TradeManager(broker=broker, notifier=alerts.append)
        db.record_trade("EURUSD", "buy", 0.10, 1.1000, 1.0980, 1.1060,
                        strategy="s", signal_hash="tc3", mode="demo")
        trade = db.open_trades("demo")[0]
        trade["entry_price"] = 1.1000
        tm.close(dict(trade), 1.1040, "TP1")
        assert len(alerts) == 1
        assert f"Trade #{int(trade['id'])}" in alerts[0]
        assert "today)" in alerts[0]
        assert "Running today:" in alerts[0]

    def test_trade_counts_fail_safe_returns_last_known(self, temp_db, monkeypatch):
        db.record_trade("EURUSD", "buy", 0.10, 1.1000, 1.0980, 1.1060,
                        strategy="s", signal_hash="tc4", mode="demo")
        good = db.trade_counts("demo")
        assert good["today"] >= 1

        def boom(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(db, "_session", boom)
        fallback = db.trade_counts("demo")
        assert fallback == good
