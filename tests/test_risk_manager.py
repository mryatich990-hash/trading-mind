"""Tests for the unified risk manager gate and Kelly sizing."""

import pytest

from risk.kelly_criterion import KellyCriterion
from risk.risk_manager import RiskManager, lots_for_risk, signal_hash


class TestLotsForRisk:
    def test_basic_eurusd(self):
        # 1% of 10000 = 100 USD; SL 20 pips * 10 USD = 200 -> 0.50 lots
        lots = lots_for_risk(10000.0, 1.0, 20.0, "EURUSD")
        assert float(lots) == pytest.approx(0.50, abs=0.01)

    def test_minimum_lot(self):
        lots = lots_for_risk(500.0, 0.25, 500.0, "EURUSD")
        assert float(lots) >= 0.01

    def test_usdjpy_dynamic_pip_value(self):
        # pip value 1000/150 = 6.67 USD/lot/pip; 150 USD risk / (20*6.67) ~ 1.12
        lots = lots_for_risk(15000.0, 1.0, 20.0, "USDJPY", usdjpy=150.0)
        assert 0.5 < float(lots) < 1.5

    def test_decimal_quantized(self):
        from decimal import Decimal
        lots = lots_for_risk(10000.0, 1.0, 20.0, "EURUSD")
        assert isinstance(lots, Decimal)
        assert str(lots) == str(lots.quantize(Decimal("0.01")))

    def test_zero_sl_floors(self):
        assert float(lots_for_risk(10000.0, 1.0, 0.0, "EURUSD")) >= 0.01


class TestSignalHash:
    def test_stable(self):
        a = signal_hash("EURUSD", "buy", "london_breakout", 1.08456)
        b = signal_hash("EURUSD", "buy", "london_breakout", 1.08456)
        assert a == b and len(a) == 32

    def test_differs_on_entry(self):
        assert signal_hash("EURUSD", "buy", "s", 1.08456) != \
            signal_hash("EURUSD", "buy", "s", 1.08457)


class TestKelly:
    def test_no_trades_default_risk(self):
        result = KellyCriterion().compute([])
        assert result.applied_pct == pytest.approx(1.0)  # RISK_PER_TRADE_PCT default

    def test_winning_history_scales(self, seeded_db):
        from core import db
        result = KellyCriterion().compute(db.closed_trades(limit=60))
        assert 0 < result.applied_pct <= 1.5
        assert result.win_rate > 40

    def test_floor_and_cap(self):
        k = KellyCriterion()
        # heavy losers -> full kelly 0 -> floor applies
        losers = [{"pnl_usd": -50} for _ in range(10)]
        assert k.compute(losers).applied_pct >= 0.25  # KELLY_FLOOR
        # big winners -> cap at 1.5%
        winners = [{"pnl_usd": 200} for _ in range(10)]
        assert k.compute(winners).applied_pct <= 1.5


class TestRiskGate:
    def _mgr(self):
        return RiskManager()

    def test_rejects_when_breaker_active(self, temp_db):
        from core import db
        db.log_breaker("daily_loss", "test", "halt")
        decision = self._mgr().evaluate("EURUSD", "buy", "london_breakout",
                                        1.1000, 1.0980, 1.1040,
                                        10000.0, 10000.0)
        assert not decision.approved
        assert "halted" in decision.reason

    def test_rejects_max_open_trades(self, temp_db):
        from config import settings
        from core import db
        for i in range(settings.MAX_OPEN_TRADES):
            db.record_trade("GBPUSD", "buy", 0.1, 1.2500, 1.2450, 1.2600,
                            strategy="ema_trend_rider", signal_hash=f"open{i}",
                            mode="demo")
        decision = self._mgr().evaluate("EURUSD", "buy", "london_breakout",
                                        1.1000, 1.0980, 1.1040,
                                        10000.0, 10000.0, mode="demo")
        assert not decision.approved
        assert "max open" in decision.reason

    def test_rejects_max_trades_per_day(self, temp_db):
        from config import settings
        from core import db
        for i in range(settings.MAX_TRADES_PER_DAY):
            db.record_trade("GBPUSD", "buy", 0.1, 1.2500, 1.2450, 1.2600,
                            strategy="ema_trend_rider", signal_hash=f"t{i}",
                            mode="demo")
            db.close_trade(db.open_trades("demo")[0]["id"], 1.2550, 50.0, 10.0)
        decision = self._mgr().evaluate("EURUSD", "buy", "london_breakout",
                                        1.1000, 1.0980, 1.1040,
                                        10000.0, 10000.0, mode="demo")
        assert not decision.approved
        assert "max daily trades" in decision.reason

    def test_approves_clean_state(self, temp_db):
        from config import settings
        decision = self._mgr().evaluate(
            "EURUSD", "buy", "london_breakout", 1.1000, 1.0980, 1.1040,
            balance=10000.0, equity=10000.0, confluence=settings.MIN_CONFLUENCE,
            mode="demo")
        assert decision.approved, decision.reason
        assert float(decision.lots) > 0

    def test_low_confluence_rejected(self, temp_db):
        decision = self._mgr().evaluate("EURUSD", "buy", "london_breakout",
                                        1.1000, 1.0980, 1.1040,
                                        10000.0, 10000.0, confluence=5, mode="demo")
        assert not decision.approved
        assert "confluence" in decision.reason

    def test_daily_loss_halts(self, temp_db):
        from core import db
        for i in range(4):
            db.record_trade("GBPUSD", "sell", 0.1, 1.2500, 1.2550, 1.2400,
                            strategy="ema_trend_rider", signal_hash=f"l{i}",
                            mode="demo")
            db.close_trade(db.open_trades("demo")[0]["id"], 1.2520, -20.0, -350.0)
        decision = self._mgr().evaluate("EURUSD", "buy", "london_breakout",
                                        1.1000, 1.0980, 1.1040,
                                        10000.0, 10000.0, mode="demo")
        assert not decision.approved

    def test_identical_pair_blocked(self, temp_db):
        from core import db
        db.record_trade("EURUSD", "buy", 0.3, 1.0800, 1.0750, 1.0900,
                        strategy="ema_trend_rider", signal_hash="corr1", mode="demo")
        decision = self._mgr().evaluate("EURUSD", "buy", "ema_trend_rider",
                                        1.1000, 1.0950, 1.1100,
                                        10000.0, 10000.0, mode="demo",
                                        open_positions=[{"pair": "EURUSD",
                                                         "direction": "buy",
                                                         "lots": 0.3}])
        assert not decision.approved
        assert "correlat" in decision.reason.lower()

    def test_portfolio_lot_guard_blocks_family(self, temp_db):
        """EURUSD + GBPUSD same-direction combined lots over 0.5 must be blocked."""
        from risk.correlation_filter import CorrelationFilter
        ok, why = CorrelationFilter.portfolio_lot_guard(
            "GBPUSD", "buy", 0.4,
            [{"pair": "EURUSD", "direction": "buy", "lots": 0.3}])
        assert not ok and "0.5" in why

    def test_anti_martingale_multiplier_bounds(self, temp_db):
        m = self._mgr()
        mult = m.anti_martingale_multiplier()
        assert 0.55 <= mult <= 1.30
