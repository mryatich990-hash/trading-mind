"""Tests for the unified strategy base, library and selector."""

import numpy as np
import pandas as pd
import pytest

from strategies.base_strategy import MarketContext, StrategySignal, atr, ema, pip_size, rsi
from strategies.strategy_selector import StrategySelector, session_of


def make_df(closes, timeframe_minutes=15, start=None):
    """Synthetic OHLCV frame from closes."""
    start = start or pd.Timestamp("2026-01-05 00:00", tz="UTC")  # Monday
    n = len(closes)
    times = pd.date_range(start, periods=n, freq=f"{timeframe_minutes}min")
    closes = np.asarray(closes, dtype=float)
    highs = closes + 0.0004
    lows = closes - 0.0004
    opens = np.concatenate([[closes[0] - 0.0002], closes[:-1]])
    return pd.DataFrame({"time": times, "open": opens, "high": highs, "low": lows,
                         "close": closes,
                         "volume": np.random.randint(100, 1000, n).astype(float)})


def make_ctx(pair="EURUSD", now=None, m15_bars=250):
    """Context with 250 m15 bars (uptrend), plus resampled h1/h4/daily."""
    now = now or pd.Timestamp("2026-01-05 08:30", tz="UTC")  # London
    rng = np.random.default_rng(11)
    closes = 1.08 + np.cumsum(rng.normal(0.0001, 0.0008, m15_bars))
    m15 = make_df(closes, 15)
    ctx = MarketContext(pair=pair, now=now, m1=m15.tail(60), m15=m15,
                        h1=make_df(closes[::4] + 0.0001, 60),
                        h4=make_df(closes[::16] + 0.0002, 240),
                        daily=make_df(closes[::64] + 0.0003, 1440))
    return ctx


class TestHelpers:
    def test_pip_size_table(self):
        assert pip_size("EURUSD") == 0.0001
        assert pip_size("USDJPY") == 0.01
        assert pip_size("XAUUSD") == 0.1
        assert pip_size("NAS100") == 1.0

    def test_ema_constant(self):
        assert abs(ema(pd.Series([100.0] * 100), 50).iloc[-1] - 100.0) < 1e-6

    def test_rsi_bounds(self):
        rng = np.random.default_rng(3)
        r = rsi(pd.Series(1.0 + np.cumsum(rng.normal(0, 0.001, 200))))
        assert r.between(0, 100).all()

    def test_atr_positive(self):
        assert atr(make_df([1.0, 1.001, 0.999, 1.0005, 1.002])).iloc[-1] > 0


class TestMarketContext:
    def test_price_and_atr(self):
        ctx = make_ctx()
        assert ctx.price() == pytest.approx(float(ctx.m15["close"].iloc[-1]))
        assert ctx.atr_m15() > 0

    def test_daily_trend_labels(self):
        ctx = make_ctx()
        assert ctx.daily_trend() in ("up", "down", "range")
        assert ctx.h4_bias() in ("bullish", "bearish", "ranging")
        assert ctx.h1_trend() in ("bullish", "bearish", "ranging")

    def test_minutes_since(self):
        ctx = make_ctx(now=pd.Timestamp("2026-01-05 08:30", tz="UTC"))
        assert ctx.minutes_since(7) == pytest.approx(90.0)

    def test_signal_rr(self):
        sig = StrategySignal(strategy="s", pair="EURUSD", direction="buy",
                             entry=1.1000, sl=1.0980, tp=1.1060, session="London")
        assert sig.sl_pips == pytest.approx(20.0)
        assert sig.rr == pytest.approx(3.0)


class TestSelector:
    def test_pip_bug_fixed(self):
        """Regression: _pip must return pip_size, not '' on empty pair."""
        ctx = make_ctx(pair="EURUSD")
        from strategies.library import LondonBreakoutStrategy
        strat = LondonBreakoutStrategy()
        assert StrategySelector._pip(strat, ctx) == 0.0001

    def test_scores_all_strategies(self, temp_db):
        ctx = make_ctx()
        selector = StrategySelector(min_score=0)
        result = selector.run(ctx)
        assert len(result.scores) == 15
        assert all(isinstance(v, (int, float)) for v in result.scores.values())
        assert result.winner is None or result.winner.direction in ("buy", "sell")

    def test_no_winner_below_threshold(self, temp_db):
        ctx = make_ctx()
        selector = StrategySelector(min_score=100)
        result = selector.run(ctx)
        assert result.winner is None
        assert "no setup above" in result.reason

    def test_disabled_strategy_scores_zero(self, temp_db):
        from core import db
        db.set_strategy_weight("london_breakout", 0.0, False, "test")
        ctx = make_ctx()
        selector = StrategySelector(min_score=0)
        result = selector.run(ctx)
        assert result.scores["london_breakout"] == 0.0

    def test_session_of(self):
        assert session_of(pd.Timestamp("2026-01-05 08:00", tz="UTC")) == "London"
        assert session_of(pd.Timestamp("2026-01-05 13:00", tz="UTC")) == "Overlap"
        assert session_of(pd.Timestamp("2026-01-05 02:00", tz="UTC")) == "Asian"

    def test_strategy_exception_degrades_gracefully(self, temp_db):
        class Boom:
            name = "boom"
            sessions = ("London", "NewYork", "Overlap")
            atr_range_pips = (3.0, 40.0)

            def __init__(self):
                self.logger = None

            def session_match(self, ctx):
                return True

            def evaluate(self, ctx):
                raise RuntimeError("boom")

        selector = StrategySelector(strategies=[Boom()], min_score=0)
        ctx = make_ctx()
        result = selector.run(ctx)
        assert result.winner is None  # failed strategy produced no signal
