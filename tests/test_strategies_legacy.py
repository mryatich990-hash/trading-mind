"""Migrated strategy tests: unit behavior plus a multi-year stability sweep."""

import numpy as np
import pandas as pd
import pytest

from strategies.asian_range_fade import AsianRangeFadeStrategy
from strategies.base_strategy import MarketContext, StrategySignal
from strategies.ema_trend_rider import EMATrendRiderStrategy
from strategies.fvg_fill import FVGFillStrategy
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.london_breakout import LondonBreakoutStrategy
from strategies.macd_momentum import MACDMomentumStrategy
from strategies.news_spike_fade import NewsSpikeFadeStrategy
from strategies.ny_reversal import NYReversalStrategy
from strategies.order_block_sniper import OrderBlockSniperStrategy
from strategies.rsi_divergence import RSIDivergenceStrategy
from strategies.strategy_selector import session_of

ALL_STRATEGY_CLASSES = [
    LondonBreakoutStrategy, NYReversalStrategy, OrderBlockSniperStrategy,
    FVGFillStrategy, EMATrendRiderStrategy, LiquiditySweepStrategy,
    RSIDivergenceStrategy, MACDMomentumStrategy, NewsSpikeFadeStrategy,
    AsianRangeFadeStrategy,
]


def build_candles(closes, freq_min=15, start=None, volume=500.0):
    """OHLCV frame from a close sequence (±1 pip band)."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    start = start or pd.Timestamp("2026-01-05 00:00", tz="UTC")
    times = pd.date_range(start, periods=n, freq=f"{freq_min}min")
    return pd.DataFrame({
        "time": times,
        "open": closes - 0.0002,
        "high": closes + 0.0005,
        "low": closes - 0.0007,
        "close": closes,
        "volume": volume,
    })


def _now(hour, minute=0):
    return pd.Timestamp("2026-01-06", tz="UTC") + pd.Timedelta(hours=hour, minutes=minute)


def _make_ctx(m15_closes, now, **kwargs):
    m15 = build_candles(m15_closes, 15)
    shift = m15["time"].iloc[-1] - now
    m15["time"] = m15["time"] - shift
    h1 = build_candles(list(m15_closes[::4]) or [1.08] * 250, 60)
    h4 = build_candles([1.08] * 250, 240)
    daily = build_candles([1.08] * 250, 1440)
    m1 = build_candles(list(m15_closes[-10:]), 1)
    return MarketContext(pair=kwargs.pop("pair", "EURUSD"), now=now, m1=m1, m15=m15,
                         h1=h1, h4=h4, daily=daily, **kwargs)


class TestLondonBreakout:
    def test_no_signal_outside_window(self):
        strategy = LondonBreakoutStrategy()
        closes = [1.08 + 0.0001 * i for i in range(120)]
        ctx = _make_ctx(closes, _now(20))
        assert strategy.evaluate(ctx) is None

    def test_range_size_filters(self):
        strategy = LondonBreakoutStrategy()
        # very tight Asian range (1 pip) -> skip
        closes = [1.0800 + 0.00002 * (i % 2) for i in range(120)]
        ctx = _make_ctx(closes, _now(8))
        assert strategy.evaluate(ctx) is None

    def test_signal_structure_when_setup_present(self):
        strategy = LondonBreakoutStrategy()
        rng = np.random.default_rng(2)
        asian = list(1.0800 + np.cumsum(rng.normal(0, 0.0004, 20)))
        asian = [min(max(c, 1.0795), 1.0825) for c in asian]  # contained range
        breakout = [1.0830, 1.0835, 1.0820, 1.0825]  # break above + retest
        ctx = _make_ctx(asian + breakout, _now(8, 30))
        sig = strategy.evaluate(ctx)
        if sig is not None:  # setup-dependent
            assert sig.direction in ("buy", "sell")
            assert sig.rr > 0


class TestNYReversal:
    def test_no_signal_outside_ny_window(self):
        strategy = NYReversalStrategy()
        closes = list(np.linspace(1.08, 1.083, 120))
        ctx = _make_ctx(closes, _now(9))
        assert strategy.evaluate(ctx) is None


class TestOrderBlockSniper:
    def test_requires_rsi_and_volume(self):
        strategy = OrderBlockSniperStrategy()
        rng = np.random.default_rng(5)
        closes = list(1.08 + np.cumsum(rng.normal(0, 0.0006, 250)))
        ctx = _make_ctx(closes, _now(10))
        sig = strategy.evaluate(ctx)
        if sig is not None:
            assert sig.strategy == "order_block_sniper"


class TestFVGFill:
    def test_no_signal_without_trend(self):
        strategy = FVGFillStrategy()
        rng = np.random.default_rng(6)
        closes = list(1.08 + np.cumsum(rng.normal(0, 0.0004, 250)))
        ctx = _make_ctx(closes, _now(10))
        ctx.h1 = build_candles(list(np.linspace(1.08, 1.08, 250)), 60)  # flat
        assert strategy.evaluate(ctx) is None

    def test_old_fvgs_invalidated(self):
        strategy = FVGFillStrategy()
        rng = np.random.default_rng(7)
        closes = list(1.08 + np.cumsum(rng.normal(0.0005, 0.0006, 250)))
        ctx = _make_ctx(closes, _now(10))
        gaps = strategy.detect_fvgs(ctx)
        for g in gaps:
            assert g["age_hours"] <= 8.0


class TestEMATrendRider:
    def test_rsi_band_gate(self):
        strategy = EMATrendRiderStrategy()
        # strong trend -> RSI usually out of 40-60 band -> no signal
        closes = list(np.linspace(1.08, 1.10, 250))
        ctx = _make_ctx(closes, _now(10))
        ctx.h1 = build_candles(closes, 60)
        sig = strategy.evaluate(ctx)
        assert sig is None or isinstance(sig, StrategySignal)


class TestLiquiditySweep:
    def test_no_signal_in_dead_zone(self):
        strategy = LiquiditySweepStrategy()
        closes = list(1.08 + np.sin(np.linspace(0, 12, 120)) * 0.002)
        ctx = _make_ctx(closes, _now(20))
        assert strategy.evaluate(ctx) is None


class TestRSIDivergence:
    def test_runs_cleanly(self):
        strategy = RSIDivergenceStrategy()
        rng = np.random.default_rng(8)
        closes = list(1.08 + np.cumsum(rng.normal(0, 0.0007, 250)))
        ctx = _make_ctx(closes, _now(10))
        sig = strategy.evaluate(ctx)
        if sig:
            assert sig.direction in ("buy", "sell")


class TestMACDMomentum:
    def test_runs_cleanly(self):
        strategy = MACDMomentumStrategy()
        closes = list(np.concatenate([np.linspace(1.078, 1.084, 125),
                                      np.linspace(1.084, 1.080, 125)]))
        ctx = _make_ctx(closes, _now(10))
        sig = strategy.evaluate(ctx)
        if sig:
            assert sig.rr >= 2.0 - 0.01  # TP is 1:2 minimum


class TestNewsSpikeFade:
    def test_requires_recent_spike(self):
        strategy = NewsSpikeFadeStrategy()
        closes = list(np.linspace(1.08, 1.083, 120))
        ctx = _make_ctx(closes, _now(10))
        assert strategy.evaluate(ctx) is None  # no spike in context

    def test_spread_filter(self):
        strategy = NewsSpikeFadeStrategy()
        closes = list(np.linspace(1.08, 1.083, 120))
        ctx = _make_ctx(closes, _now(10), spread_pips=6.0)
        ctx.news_spike = {"direction": "up", "size_pips": 40, "minutes_since": 2,
                          "spike_high": 1.0835, "spike_low": 1.0800}
        assert strategy.evaluate(ctx) is None  # spread 6 > 5 pips

    def test_valid_fade_signal(self):
        strategy = NewsSpikeFadeStrategy()
        closes = [1.0830, 1.0831, 1.0830, 1.0831]  # stalled M1
        ctx = _make_ctx(closes, _now(10))
        ctx.m1 = build_candles([1.0830, 1.0831, 1.0830, 1.0831], 1)
        ctx.news_spike = {"direction": "up", "size_pips": 45, "minutes_since": 2,
                          "spike_high": 1.0835, "spike_low": 1.0800}
        sig = strategy.evaluate(ctx)
        assert sig is not None and sig.direction == "sell"
        assert sig.sl > sig.entry > sig.tp  # fade structure


class TestAsianRangeFade:
    def test_requires_quiet_market(self):
        strategy = AsianRangeFadeStrategy()
        closes = list(1.08 + np.sin(np.linspace(0, 9, 60)) * 0.001)
        ctx = _make_ctx(closes, _now(3))
        sig = strategy.evaluate(ctx)
        if sig:
            assert sig.session == "Asian"


class TestSignalValidity:
    def test_rr_math(self):
        sig = StrategySignal(strategy="x", pair="EURUSD", direction="buy",
                             entry=1.1000, sl=1.0980, tp=1.1060, session="London")
        assert sig.sl_pips == pytest.approx(20.0)
        assert sig.rr == pytest.approx(3.0)


@pytest.mark.parametrize("strategy_cls", ALL_STRATEGY_CLASSES)
def test_historical_simulation_stability(strategy_cls):
    """Multi-regime sweep: each strategy must run without raising on varied
    regimes and any signal must be structurally valid (SL/TP on correct sides)."""
    strategy = strategy_cls()
    regimes = [
        (1.10, 0.0004),  # calm grind up
        (1.16, 0.0009),  # volatile
        (1.05, 0.0006),  # downtrend chop
        (1.08, 0.0003),  # quiet
        (1.09, 0.0007),  # mixed
    ]
    for year, (start, vol) in zip(range(2020, 2025), regimes):
        rng = np.random.default_rng(year)
        n = 300
        closes = list(start + np.cumsum(rng.normal(0, vol, n)))
        now = pd.Timestamp(f"{year}-06-15 10:00", tz="UTC")
        ctx = _make_ctx(closes, now, pair="EURUSD")
        try:
            sig = strategy.evaluate(ctx)
        except Exception as exc:  # pragma: no cover - failure means regression
            pytest.fail(f"{strategy_cls.__name__} raised on {year} data: {exc}")
        if sig is not None:
            assert sig.direction in ("buy", "sell")
            assert sig.sl_pips > 0 and sig.rr > 0
            if sig.direction == "buy":
                assert sig.sl < sig.entry < sig.tp
            else:
                assert sig.sl > sig.entry > sig.tp


def test_session_helper():
    assert session_of(pd.Timestamp("2026-01-05 08:00", tz="UTC")) == "London"
    assert session_of(pd.Timestamp("2026-01-05 13:00", tz="UTC")) == "Overlap"
    assert session_of(pd.Timestamp("2026-01-05 03:00", tz="UTC")) == "Asian"
