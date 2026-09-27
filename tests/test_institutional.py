"""Tests for institutional data plumbing: intermarket, COT, calendar, ML, backtests."""

import numpy as np
import pandas as pd
import pytest

from institutional.intermarket_analyzer import IntermarketSnapshot
from risk.monte_carlo import MonteCarloEngine


class TestIntermarketSnapshot:
    def test_bias_for(self):
        snap = IntermarketSnapshot(usd_bias=20.0, eur_bias=-10.0)
        assert snap.bias_for("USD") == 20.0
        assert snap.bias_for("eur") == -10.0
        assert snap.bias_for("XXX") == 0.0

    def test_pair_bias(self):
        snap = IntermarketSnapshot(usd_bias=10.0, eur_bias=30.0)
        assert snap.pair_bias("EURUSD") == pytest.approx(20.0)
        assert snap.pair_bias("XAUUSD") == snap.bias_for("XAU") - 10.0

    def test_vix_scaling(self):
        snap = IntermarketSnapshot(vix=45.0, vix_halt=True, vix_position_scale=0.0)
        assert snap.vix_halt and snap.vix_position_scale == 0.0


class TestMonteCarlo:
    def test_empty_history(self):
        result = MonteCarloEngine(simulations=100).run([])
        assert result.simulations == 0

    def test_simulation_shape(self, seeded_db):
        from core import db
        result = MonteCarloEngine(simulations=200, horizon=50).run(
            db.closed_trades(limit=200))
        assert result.simulations == 200
        assert 0 <= result.prob_ruin <= 100
        assert result.p5_outcome <= result.p50_outcome <= result.p95_outcome


class TestBacktestEngine:
    def _frames(self):
        rng = np.random.default_rng(9)
        closes = 1.08 + np.cumsum(rng.normal(0.0001, 0.0009, 700))
        times = pd.date_range("2025-11-03", periods=700, freq="15min", tz="UTC")
        closes = np.asarray(closes)
        df = pd.DataFrame({
            "time": times, "open": np.concatenate([[closes[0]], closes[:-1]]),
            "high": closes + 0.0005, "low": closes - 0.0005, "close": closes,
            "volume": np.random.randint(100, 1000, 700).astype(float)})
        return {"m15": df}

    def test_insufficient_data_note(self, temp_db):
        from backtesting.backtest_engine import BacktestEngine
        from strategies.library import LondonBreakoutStrategy

        df = self._frames()["m15"].head(100)
        result = BacktestEngine().run(LondonBreakoutStrategy(), {"m15": df}, "EURUSD")
        assert result.note == "insufficient data"
        assert result.trades == 0

    def test_result_serialization(self, temp_db):
        from backtesting.backtest_engine import BacktestEngine
        from strategies.library import EMATrendRiderStrategy

        result = BacktestEngine().run(EMATrendRiderStrategy(), self._frames(), "EURUSD")
        d = result.as_dict()
        for key in ("strategy", "pair", "trades", "win_rate", "profit_factor", "passed"):
            assert key in d

    def test_startup_gate_no_data_does_not_lockout(self, temp_db):
        from backtesting.backtest_engine import BacktestEngine, run_startup_gate
        from strategies.library import EMATrendRiderStrategy

        passing, disabled = run_startup_gate(BacktestEngine(),
                                             [EMATrendRiderStrategy()], {})
        assert passing == 0 and disabled == []
        from core import db
        assert db.get_state("backtest_gate_passed") == "1"


class TestMLModel:
    def test_feature_vector_length(self, temp_db):
        from ml.ml_model import FEATURE_NAMES, MLModel

        row = {"features_json": '{"rsi14": 42, "vix": 18}', "opened_at":
               "2026-01-05T08:00:00+00:00", "session": "London",
               "confluence_score": 8, "groq_conviction": 80}
        vec = MLModel.features_from_row(row)
        assert vec is not None and len(vec) == len(FEATURE_NAMES)

    def test_predict_without_model_is_neutral(self, temp_db):
        from ml.ml_model import MLModel

        model = MLModel(model_dir="/tmp/nonexistent_models")
        assert model.predict_loss_prob([0.0] * 18) == 0.5
        assert not model.should_skip(0.5)


class TestSessionOptimizer:
    def test_heatmap_empty_ok(self, temp_db):
        from ml.session_optimizer import SessionOptimizer

        assert SessionOptimizer().heatmap() == []

    def test_hour_allowed_default_true(self, temp_db):
        from ml.session_optimizer import SessionOptimizer

        assert SessionOptimizer().hour_allowed("london_breakout", 8, 0)


class TestShadowTrader:
    def test_close_shadow_without_rows(self, temp_db):
        from backtesting.shadow_trader import ShadowTrader

        ShadowTrader().close_shadow(9999, 1.10)  # must not raise

    def test_weekly_report_no_trades(self, temp_db):
        from backtesting.shadow_trader import ShadowTrader

        assert "no trades" in ShadowTrader().weekly_report().lower()


class TestDashboard:
    def test_routes_respond(self, temp_db):
        from dashboard.app import app

        app.config["TESTING"] = True
        client = app.test_client()
        assert client.get("/health").status_code == 200
        assert client.get("/api/status").status_code == 200
        assert client.get("/api/open_trades").status_code == 200
        assert client.get("/api/research").status_code == 200
        assert client.get("/api/strategies").status_code == 200
        assert client.get("/api/heatmap").status_code == 200
        assert client.get("/api/ml").status_code == 200
        assert client.get("/api/history").status_code == 200
        assert client.get("/api/equity").status_code == 200

    def test_control_pause_resume(self, temp_db):
        from core import db
        from dashboard.app import app

        app.config["TESTING"] = True
        client = app.test_client()
        assert client.post("/api/control", json={"action": "pause"}).status_code == 200
        assert db.get_state("trading_paused") == "1"
        assert client.post("/api/control", json={"action": "resume"}).status_code == 200
        assert db.get_state("trading_paused") == "0"


class TestTelegramCommands:
    def test_status_and_stats(self, temp_db):
        from notifications.telegram_cmds import cmd_stats, cmd_status

        assert "Bot:" in cmd_status([])
        assert "No trades" in cmd_stats([]) or "Trades" in cmd_stats([])

    def test_pause_resume(self, temp_db):
        from core import db
        from notifications.telegram_cmds import cmd_pause, cmd_resume

        cmd_pause([])
        assert db.get_state("trading_paused") == "1"
        cmd_resume([])
        assert db.get_state("trading_paused") == "0"

    def test_register_all(self):
        from notifications.telegram_bot import TelegramBot
        from notifications.telegram_cmds import COMMANDS, register_all

        bot = TelegramBot()
        register_all(bot)
        for name in COMMANDS:
            assert bot._handlers.get(name) is not None
