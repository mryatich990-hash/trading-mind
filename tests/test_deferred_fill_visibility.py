"""Deferred limit fills must be visible: audit entry + Telegram notice.

Three approved setups on 2026-10-01 vanished silently because the limit
zone was never reached — no trade row, no audit line, no message. These
tests pin the new fill_deferred audit + notification behavior.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from core import db
from execution.execution_engine import ExecutionEngine
from research.research_engine import ResearchVerdict


class _NoZoneBroker:
    """Broker whose price never reaches the limit zone -> deferred fill."""

    name = "fake"
    configured = True

    def __init__(self):
        self._price = 1.2000  # far from the verdict entry below

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

    def market_order(self, pair, direction, lots, sl, tp, comment=""):
        raise AssertionError("market_order must not be called on deferred fill")


def _verdict():
    return ResearchVerdict(
        pair="EURUSD", direction="buy", strategy="rsi_divergence",
        approved=True, reason="test", entry=1.1000, sl=1.0980,
        tp1=1.1060, tp2=1.1120, tp3=1.1180, conviction=80,
        consensus_size=1.0, confluence=8)


class TestDeferredFillVisibility:
    def test_deferred_fill_audits_and_notifies_without_trade_row(self, temp_db):
        alerts = []
        exe = ExecutionEngine(mt5=_NoZoneBroker(), oanda=MagicMock(),
                              notifier=alerts.append, mode="demo")
        exe.oanda.available = MagicMock(return_value=False)

        trade_id = exe.execute(_verdict(), Decimal("0.10"), 10000.0, 8)

        assert trade_id is None                      # no trade row
        assert db.trades_today("demo") == []         # nothing recorded
        # audit trail now shows the deferred fill
        with db.engine.begin() as conn:
            from sqlalchemy import text as sqltext
            row = conn.execute(sqltext(
                "SELECT category, action, detail FROM audit_log "
                "ORDER BY id DESC LIMIT 1")).first()
        assert row is not None and row[1] == "fill_deferred"
        assert "EURUSD" in row[2] and "not reached" in row[2]
        # operator got a Telegram notice instead of silence
        assert alerts and "deferred" in alerts[0].lower()
        assert "rsi_divergence" in alerts[0]

    def test_immediate_fill_still_notifies_fill(self, temp_db):
        """Non-deferred path unchanged: market-order strategies fill and alert."""
        class _ImmediateBroker(_NoZoneBroker):
            def __init__(self):
                super().__init__()
                self.orders = []

            def market_order(self, pair, direction, lots, sl, tp, comment=""):
                self.orders.append((pair, direction))
                return 777

        broker = _ImmediateBroker()
        alerts = []
        exe = ExecutionEngine(mt5=broker, oanda=MagicMock(),
                              notifier=alerts.append, mode="demo")
        exe.oanda.available = MagicMock(return_value=False)
        v = _verdict()
        v.strategy = "forced_test_trade"   # market-order strategy
        v.entry = broker._price            # at-market: fills immediately

        trade_id = exe.execute(v, Decimal("0.10"), 10000.0, 8)

        assert trade_id is not None
        assert broker.orders == [("EURUSD", "buy")]
        assert alerts and "Trade #" in alerts[0]
