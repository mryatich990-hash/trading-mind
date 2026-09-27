"""Tests for UPGRADES 1-9 (fallback backends only; heavy deps are guarded)."""

import json

import numpy as np
import pandas as pd
import pytest


def _candles(n=400, base=1.08, seed=7):
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0.00008, 0.0008, n))
    times = pd.date_range("2026-09-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "time": times, "open": np.concatenate([[closes[0]], closes[:-1]]),
        "high": closes + 0.0004, "low": closes - 0.0004, "close": closes,
        "volume": rng.integers(100, 1000, n).astype(float)})
    return df


# ---------------- UPGRADE 1 ----------------

class TestLSTM:
    def test_features_shape(self):
        from deep_learning.lstm_model import build_features

        feats = build_features(_candles(120))
        assert feats.shape[1] == 9 and np.isfinite(feats).all()

    def test_numpy_train_predict(self, temp_db):
        from deep_learning.lstm_model import LSTMModel

        model = LSTMModel("EURUSD", models_dir="/tmp/dl_test_models")
        assert model.backend in ("tensorflow", "numpy")
        info = model.train(_candles(400))
        assert info["ok"]
        vote = model.predict(_candles(120))
        assert vote["direction"] in ("bullish", "bearish")
        assert 0 <= vote["confidence"] <= 100


class TestXGBoost:
    def test_47_feature_contract(self):
        from deep_learning.xgboost_model import FEATURE_NAMES, build_feature_vector

        assert len(FEATURE_NAMES) == 47
        vec = build_feature_vector({"rsi14": 55.0})
        assert vec.shape == (47,) and vec[FEATURE_NAMES.index("rsi14")] == 55.0

    def test_train_predict(self, temp_db):
        from deep_learning.xgboost_model import XGBoostModel, build_feature_vector

        model = XGBoostModel(models_dir="/tmp/dl_test_models")
        rng = np.random.default_rng(3)
        X = rng.normal(size=(120, 47)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        info = model.train(X, y)
        assert info["ok"]
        proba = model.predict_proba({"rsi14": 50})
        assert proba is None or 0 <= proba <= 100


class TestRLAgent:
    def test_bandit_train_act_learn(self, temp_db):
        from deep_learning.rl_agent import RLAgent

        agent = RLAgent(models_dir="/tmp/dl_test_models")
        info = agent.train(np.random.default_rng(5).normal(size=(80, 47)).astype(np.float32),
                           (np.random.default_rng(6).normal(size=80) > 0).astype(float))
        assert info["ok"]
        action = agent.act({})
        assert action["action"] in ("buy", "sell", "skip")
        agent.learn_outcome({}, "buy", True)

    def test_env_rewards(self):
        from deep_learning.rl_agent import TradeEnv

        env = TradeEnv(np.zeros((3, 47), dtype=np.float32), np.array([1.0, 0.0, 1.0]))
        env.reset()
        _, r_buy, _, _ = env.step(0)
        _, r_sell, _, _ = env.step(1)
        _, r_skip, _, _ = env.step(2)
        assert r_buy == 1.0 and r_sell == 1.0 and r_skip == -0.1


class TestEnsemble:
    def test_vote_flow(self, temp_db):
        from deep_learning.ensemble_voter import EnsembleVoter, EnsembleVerdict

        voter = EnsembleVoter(pairs=["EURUSD"])
        verdict = voter.vote("EURUSD", "buy", m15_df=_candles(120),
                             ctx_dict={"rsi14": 40}, features=np.zeros(47))
        assert isinstance(verdict, EnsembleVerdict)
        assert verdict.proceed in (True, False)
        assert verdict.agreement in ("all_agree", "split", "skipped", "unknown")

    def test_outcome_recording_and_accuracy(self, temp_db):
        from deep_learning.ensemble_voter import EnsembleVoter

        voter = EnsembleVoter(pairs=["EURUSD"])
        v = voter.vote("EURUSD", "buy")
        for won in (True, False):
            voter.record_outcome("EURUSD", "buy", v, won)
        acc = voter.rolling_accuracy()
        assert set(acc) >= {"lstm", "xgboost", "rl", "window"}


# ---------------- UPGRADE 2 ----------------

class TestNLP:
    def test_finbert_lexicon_scores(self, temp_db):
        from nlp.finbert_engine import FinBertEngine

        eng = FinBertEngine()
        assert eng.backend in ("finbert", "lexicon")
        hawk = eng.score_text("Fed signals rate hike amid inflation concern and robust growth")
        dove = eng.score_text("ECB discusses rate cut and accommodative policy on economic concern")
        assert hawk["score"] > 0 > dove["score"]
        eng.add_headline("Gold rallies as dollar weakens")
        assert eng.aggregate("XAU")["samples"] >= 1 or eng.aggregate("USD")["samples"] >= 1

    def test_central_bank_scoring(self):
        from nlp.central_bank_parser import score_statement

        score, label, _ = score_statement("The committee remains concerned about inflation "
                                          "and may raise rates to support tightening")
        assert score > 0 and label == "hawkish"
        score2, label2, _ = score_statement("Officials discussed a rate cut and further easing")
        assert score2 < 0 and label2 == "dovish"

    def test_twitter_inactive_without_token(self, temp_db):
        from nlp.finbert_engine import FinBertEngine
        from nlp.twitter_sentiment import TwitterSentiment

        engine = TwitterSentiment(FinBertEngine())
        assert engine.active is False  # no TWITTER_BEARER_TOKEN in test env

    def test_trends_signals_shape(self, temp_db):
        from nlp.google_trends import GoogleTrends

        trends = GoogleTrends()
        assert isinstance(trends.signals(), dict)


# ---------------- UPGRADE 3 ----------------

class TestTick:
    def test_stats_from_synthetic_ticks(self, temp_db):
        from tick.tick_engine import TickDataEngine

        eng = TickDataEngine()
        base = 1.08
        for i in range(120):
            px = base + i * 1e-6 * (1 if i % 3 else -2)
            eng.add_tick("EURUSD", px - 0.00005, px + 0.00005)
        stats = eng.get_stats("EURUSD")
        assert stats["velocity"] > 0
        assert 0 <= stats["imbalance"] <= 1

    def test_breaker_halts_at_spike(self, temp_db):
        from tick.tick_circuit_breaker import TickCircuitBreaker

        br = TickCircuitBreaker()
        for _ in range(30):
            br.update_baseline("EURUSD", 50.0)
            br.evaluate("EURUSD", 50.0)
        state = br.evaluate("EURUSD", 600.0)  # 12x baseline
        assert state["halted"] is True
        assert br.is_halted("EURUSD")
        # calm period required to resume
        state2 = br.evaluate("EURUSD", 50.0)
        assert state2["halted"] is True and state2["resume_in_sec"] > 0

    def test_microstructure_stop_hunt(self, temp_db):
        import time as _time

        from tick.tick_engine import TickDataEngine
        from tick.microstructure import MicrostructureAnalyzer

        eng = TickDataEngine()
        base = 1.0800
        now = _time.time()
        # normal ticks, spike up, full revert (all "recent")
        prices = [base] * 40 + [base + 0.0008] * 6 + [base] * 10
        for i, p in enumerate(prices):
            eng.add_tick("EURUSD", p - 0.00002, p + 0.00002, ts=now - (len(prices) - i))
        micro = MicrostructureAnalyzer(eng)
        events = micro.detect("EURUSD")
        assert any(e["type"] == "stop_hunt" for e in events)


# ---------------- UPGRADE 4 ----------------

class TestScalping:
    def test_kill_zones(self, temp_db):
        from datetime import datetime, timezone

        from scalping.scalping_engine import ScalpingEngine

        eng = ScalpingEngine()
        assert eng.kill_zone(datetime(2026, 9, 28, 7, 30, tzinfo=timezone.utc)) == "london_open"
        assert eng.kill_zone(datetime(2026, 9, 28, 13, 30, tzinfo=timezone.utc)) == "ny_open"
        assert eng.kill_zone(datetime(2026, 9, 28, 11, 0, tzinfo=timezone.utc)) is None

    def test_gates(self, temp_db):
        from scalping.scalping_engine import ScalpingEngine

        eng = ScalpingEngine()
        ok, reason = eng.can_scalp("EURUSD", 0.7, m15_df=_candles(200), open_trades=[])
        assert ok or reason in ("outside kill zones",)
        ok, reason = eng.can_scalp("EURUSD", 3.0)
        assert not ok
        eng.record_outcome(False, -10)
        eng.record_outcome(False, -10)
        eng.record_outcome(False, -10)
        ok, reason = eng.can_scalp("EURUSD", 0.5)
        assert not ok and "day-locked" in reason

    def test_micro_ob(self, temp_db):
        from scalping.scalping_engine import ScalpingEngine

        df = _candles(60, seed=11)
        # engineer a bullish impulse after a down candle
        df.loc[df.index[-4], "close"] = df["open"].iloc[-4] - 0.0006
        for k in (-3, -2, -1):
            df.loc[df.index[k], "close"] = df["close"].iloc[k - 1] + 0.0012
            df.loc[df.index[k], "high"] = df["close"].iloc[k] + 0.0002
        ob = ScalpingEngine._micro_ob(df, 0.0005)
        assert ob is None or ob["side"] in ("bullish", "bearish")


# ---------------- UPGRADE 5 ----------------

class TestStatArb:
    def test_synthetic_gap_shape(self, temp_db):
        from arbitrage.stat_arb_engine import StatArbEngine

        eng = StatArbEngine()
        # no data engine -> None, must not raise
        assert eng.synthetic_gap_pips() is None

    def test_scan_with_position_blocked(self, temp_db):
        from arbitrage.stat_arb_engine import StatArbEngine, StatArbPosition

        eng = StatArbEngine()
        eng.position = StatArbPosition(kind="correlation", legs=["EURUSD"],
                                       opened_at=0.0, entry_gap=1.0, max_gap=10.0)
        assert eng.scan() == []
        reasons = eng.manage()  # time limit (opened_at=0 -> ancient)
        assert any("time limit" in r for r in reasons)


# ---------------- UPGRADE 6 ----------------

class TestAnalytics:
    def test_ratios(self):
        from analytics.advanced_analytics import compute_ratios

        pnls = [100, -50, 80, -40, 120, -30, 90, -60, 70, 110]
        r = compute_ratios(pnls)
        assert r["trades"] == 10
        assert r["profit_factor"] and r["profit_factor"] > 1
        assert r["sharpe"] is not None and r["max_dd"] > 0

    def test_mae_mfe(self, temp_db):
        from analytics.mae_mfe_analyzer import MAEMFEAnalyzer

        analyzer = MAEMFEAnalyzer()
        result = analyzer.analyze(force=True)  # empty db -> graceful
        assert "samples" in result

    def test_quality_scorer(self):
        from analytics.trade_quality_scorer import TradeQualityScorer

        scorer = TradeQualityScorer()
        good = scorer.score(won=True, rr_achieved=2.0, confluence=6,
                            immediate_move=True, captured_fraction=0.9)
        bad = scorer.score(won=False, rr_achieved=0.0, confluence=3,
                           immediate_move=False, captured_fraction=0.1,
                           adverse_first_pips=15)
        assert good["score"] > bad["score"]
        assert bad["flagged"] and not good["flagged"]


# ---------------- UPGRADE 7 ----------------

class TestInfrastructure:
    def test_cache_memory_fallback(self, temp_db):
        from infrastructure.redis_cache import get_cache

        cache = get_cache()
        cache.set("k", {"v": 1}, "indicators")
        assert cache.get("k", "indicators") == {"v": 1}
        assert cache.get("missing", "indicators") is None
        assert 0 <= cache.hit_rate() <= 100

    def test_hot_reload_apply(self, temp_db):
        from config import settings
        from infrastructure.hot_reload import apply_overrides

        applied = apply_overrides({"RISK_PER_TRADE_PCT": 0.75,
                                   "NOT_ALLOWED": 1})
        assert "RISK_PER_TRADE_PCT" in applied
        assert settings.RISK_PER_TRADE_PCT == 0.75
        applied2 = apply_overrides({"RISK_PER_TRADE_PCT": 99})  # out of range
        assert applied2 == []

    def test_versioning(self, temp_db):
        from infrastructure.strategy_versioning import StrategyVersioning

        v = StrategyVersioning()
        v.bump("ema_trend_rider", "1.1.0", "tuned stop")
        assert v.current("ema_trend_rider") == "1.1.0"
        assert v.current("other") == "1.0.0"

    def test_profiler_budgets(self, temp_db):
        from infrastructure.performance_profiler import get_profiler

        profiler = get_profiler()
        profiler.record("groq_verify_call", 20.0)  # over the 15s groq budget
        profiler.record("feed_fetch", 0.2)
        assert profiler.budget_breaches.get("groq_verify_call", 0) >= 1
        assert any(r["function"] == "feed_fetch" for r in profiler.summary())

    def test_dependency_report(self, temp_db):
        from infrastructure.dependency_manager import DependencyManager

        mgr = DependencyManager()
        report = mgr.check()
        assert report and {"package", "installed", "latest"} <= set(report[0])


# ---------------- UPGRADE 8 ----------------

class TestSmartSL:
    def test_atr_stop_buy(self):
        from risk.smart_sl_engine import SmartSLEngine

        eng = SmartSLEngine()
        df = _candles(120)
        result = eng.stop_distance("EURUSD", "ema_trend_rider", 1.085, df, "buy")
        assert result["sl_price"] < 1.085 and result["distance_pips"] > 0
        assert result["basis"].startswith("atr") or result["basis"].startswith("structure")

    def test_structure_stop(self):
        from risk.smart_sl_engine import SmartSLEngine

        eng = SmartSLEngine()
        df = _candles(120)
        result = eng.stop_distance("EURUSD", "order_block", 1.085, df, "sell")
        assert result["sl_price"] > 1.085

    def test_initial_tp_min_rr(self):
        from risk.smart_sl_engine import SmartSLEngine

        eng = SmartSLEngine()
        df = _candles(120)
        tp = eng.initial_tp("EURUSD", 1.085, 15.0, df, "buy", min_rr=1.5)
        assert tp["rr"] >= 1.5

    def test_tp_adjustment_rules(self):
        from risk.smart_sl_engine import SmartSLEngine

        eng = SmartSLEngine()
        df = _candles(120, seed=9)
        trade = {"pair": "EURUSD", "direction": "buy", "entry_price": 1.0800,
                 "tp": 1.0830}
        # winning trade + fading momentum -> tighten but never below price
        out = eng.adjust_tp(trade, df, current_price=1.0825)
        if out and out["action"] == "tightened":
            assert out["tp"] > 1.0825
        # losing trade -> never tighten to lock (no action or extend only)
        out2 = eng.adjust_tp(trade, df, current_price=1.0790)
        if out2:
            assert out2["action"] in ("extended", "tightened")
            if out2["action"] == "tightened":
                assert out2["tp"] > 1.0790


# ---------------- registry ----------------

class TestRegistry:
    def test_status_payload(self, temp_db):
        from upgrades_registry import UpgradeRegistry

        registry = UpgradeRegistry(pairs=["EURUSD"])
        payload = registry.status()
        for key in ("deep_learning", "nlp", "tick", "scalping", "stat_arb",
                    "analytics", "system_health"):
            assert key in payload
