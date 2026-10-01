"""Regression tests: feed resilience (cache TTLs, 1m freeze fallback, probe
bypass) and the step3 mirror-retry (direction alignment with HTF trend)."""

from types import SimpleNamespace

import pandas as pd
import pytest

from data.market_data_engine import (CandleData, MarketDataEngine,
                                     freshness_limit_sec, validate_ohlcv)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mk_df(timeframe_min: int, n: int, vary: bool = True) -> pd.DataFrame:
    """Fresh, validated-looking OHLCV frame ending 'now'."""
    end = pd.Timestamp.now(tz="UTC").floor("min")
    times = pd.date_range(end=end, periods=n, freq=f"{timeframe_min}min")
    step = 0.0002 if vary else 0.0
    closes = [1.1000 + i * step for i in range(n)]
    return pd.DataFrame({
        "time": times,
        "open": [c - 0.0001 for c in closes],
        "high": [c + 0.0002 for c in closes],
        "low": [c - 0.0002 for c in closes],
        "close": closes,
        "volume": [1000.0 + i for i in range(n)],
    })


class FakeFeed:
    name = "yfinance"
    configured = True

    def __init__(self, frames: dict[tuple[int, int], pd.DataFrame] = None):
        self.frames = frames or {}
        self.calls: list[tuple[str, int, int]] = []

    def fetch(self, pair: str, timeframe_min: int, count: int) -> pd.DataFrame:
        self.calls.append((pair, timeframe_min, count))
        df = self.frames.get((timeframe_min, count))
        if df is None:
            df = _mk_df(timeframe_min, count)
        return df.copy()


def _engine(feed: FakeFeed) -> MarketDataEngine:
    return MarketDataEngine(feeds=[feed])


# ---------------------------------------------------------------------------
# cache TTLs
# ---------------------------------------------------------------------------

class TestCacheTtlByTimeframe:
    def test_d1_cached_within_ttl(self, temp_db):
        feed = FakeFeed()
        eng = _engine(feed)
        eng.get_candles("EURUSD", 1440, 260)
        eng.get_candles("EURUSD", 1440, 260)
        assert len(feed.calls) == 1  # second call served from cache

    def test_m15_ttl_shorter_than_d1(self, temp_db):
        feed = FakeFeed()
        eng = _engine(feed)
        eng.get_candles("EURUSD", 15, 300)
        eng.get_candles("EURUSD", 15, 300)
        assert len(feed.calls) == 1

    def test_different_bundles_do_not_share_cache(self, temp_db):
        feed = FakeFeed()
        eng = _engine(feed)
        eng.get_candles("EURUSD", 15, 300)
        eng.get_candles("EURUSD", 60, 300)
        assert len(feed.calls) == 2

    def test_expired_entry_refetched(self, temp_db, monkeypatch):
        feed = FakeFeed()
        eng = _engine(feed)
        eng.get_candles("EURUSD", 1440, 260)
        # age the cache entry past the d1 TTL
        first = feed.calls and 1
        cached = eng._cache["EURUSD_1440_260"]
        cached.fetched_at -= 3601.0
        eng.get_candles("EURUSD", 1440, 260)
        assert len(feed.calls) == 2


class TestProbeBypass:
    def test_probe_skips_cache(self, temp_db):
        feed = FakeFeed()
        eng = _engine(feed)
        eng.get_candles("EURUSD", 15, 60, probe=True)
        eng.get_candles("EURUSD", 15, 60, probe=True)
        assert len(feed.calls) == 2  # probes always hit the live feed

    def test_probe_does_not_read_normal_cache(self, temp_db):
        feed = FakeFeed()
        eng = _engine(feed)
        eng.get_candles("EURUSD", 15, 60)          # populates cache
        eng.get_candles("EURUSD", 15, 60, probe=True)
        assert len(feed.calls) == 2


# ---------------------------------------------------------------------------
# validator: quiet-session flat 15m candles are NOT a frozen feed
# ---------------------------------------------------------------------------

class TestFreezeDetectionScope:
    @pytest.fixture(autouse=True)
    def _force_market_open(self, monkeypatch):
        import data.market_data_engine as mde
        monkeypatch.setattr(mde, "fx_market_closed", lambda now=None: False)

    def test_quiet_15m_flat_candles_pass(self, temp_db):
        """Flat 15m closes in Yahoo's lag zone must not log freeze failures."""
        flat = _mk_df(15, 300, vary=False)
        assert validate_ohlcv(flat, "EURUSD", 1800, 15) is None

    def test_frozen_1m_still_flagged(self, temp_db):
        """Genuine sub-15m freeze (stuck closes, fresh timestamps) still errors."""
        frozen_1m = _mk_df(1, 60, vary=False)
        assert validate_ohlcv(frozen_1m, "EURUSD", 960, 1) == "data freeze detected"

    def test_stale_15m_still_flagged(self, temp_db):
        """Fresh-but-flat is fine; genuinely old data still errors."""
        old = _mk_df(15, 300, vary=True)
        old["time"] = old["time"] - pd.Timedelta(hours=2)
        result = validate_ohlcv(old, "EURUSD", 1800, 15)
        assert isinstance(result, str) and result.startswith("stale:")


class TestFreshnessLimit:
    def test_sub_15m_gets_16min_floor(self, temp_db, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "STALENESS_LIMIT_SEC", 600)
        assert freshness_limit_sec(1) == 960

    def test_15m_gets_two_bars(self, temp_db, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "STALENESS_LIMIT_SEC", 600)
        assert freshness_limit_sec(15) == 1800

    def test_setting_can_raise_floor(self, temp_db, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "STALENESS_LIMIT_SEC", 2000)
        assert freshness_limit_sec(15) == 2000


# ---------------------------------------------------------------------------
# 1m freeze fallback
# ---------------------------------------------------------------------------

class TestOneMinuteFreezeFallback:
    def test_frozen_1m_serves_resampled_15m(self, temp_db):
        frozen_1m = _mk_df(1, 60, vary=False)          # all identical closes
        good_15m = _mk_df(15, 300, vary=True)
        feed = FakeFeed({(1, 60): frozen_1m, (15, 300): good_15m})
        eng = _engine(feed)

        candle = eng.get_candles("EURUSD", 1, 60)
        assert candle.timeframe_min == 1
        assert candle.source.endswith("+resample")
        assert len(candle.df) > 0
        assert {"time", "open", "high", "low", "close", "volume"} <= set(candle.df.columns)

    def test_xauusd_never_gets_resampled_1m(self, temp_db):
        """XAUUSD 1m uses futures with real volume — freeze there must fail hard."""
        frozen_1m = _mk_df(1, 60, vary=False)
        feed = FakeFeed({(1, 60): frozen_1m})
        eng = _engine(feed)
        from data.market_data_engine import FeedError
        with pytest.raises(FeedError):
            eng.get_candles("XAUUSD", 1, 60)

    def test_stale_1m_not_resampled(self, temp_db):
        """Only freeze falls back; genuinely stale data must still error."""
        old = _mk_df(1, 60, vary=True)
        old["time"] = old["time"] - pd.Timedelta(hours=3)
        feed = FakeFeed({(1, 60): old})
        eng = _engine(feed)
        from data.market_data_engine import FeedError
        with pytest.raises(FeedError):
            eng.get_candles("EURUSD", 1, 60)


# ---------------------------------------------------------------------------
# step3 mirror retry
# ---------------------------------------------------------------------------

class FakeMacro:
    def __init__(self):
        self.calls: list[str] = []

    def analyze(self, pair, direction):
        # called once per evaluate() (step 1) -> perfect recursion tracker
        self.calls.append(direction)
        from research.macro_analyzer import MacroResult
        return MacroResult(news_gate="clear")


class FakeMatcher:
    def match(self, *a, **k):
        return SimpleNamespace(required_confluence=8, win_rate=60.0, samples=30)


class FakeValidator:
    def __init__(self):
        self.directions: list[str] = []

    def validate(self, pair, direction, frames, htf, macro, confluence_bonus=0):
        self.directions.append(direction)
        from research.entry_validator import EntryChecklist
        return EntryChecklist(score=9, required=8)


class FakeBuilder:
    def build(self, research_data):
        return "test prompt"


class FakeVerifier:
    def __init__(self, validator: FakeValidator):
        self._validator = validator

    def verified_decision(self, pair, prompt, max_attempts=1):
        # conviction 80 clears the real GROQ_MIN_CONVICTION=74 threshold
        return {"decision": self._validator.directions[-1],
                "conviction": 80, "_consensus": 3}


class FakeHTFAnalyzer:
    def __init__(self, ok_dir: str = None, agree: bool = True):
        self.ok_dir = ok_dir
        self.agree = agree

    def analyze(self, pair, frames, direction):
        from research.htf_analyzer import HTFResult
        ok = self.agree and (self.ok_dir is None or direction == self.ok_dir)
        return HTFResult(daily_trend="up", h4_bias="bullish",
                         htf_agree=self.agree, direction_ok=ok)


def _engine_for_mirror(validator: FakeValidator, htf: FakeHTFAnalyzer):
    from research.research_engine import ResearchEngine

    eng = ResearchEngine.__new__(ResearchEngine)
    # minimal attrs used by evaluate(); Groq consensus fully faked
    eng.data = SimpleNamespace(get_frames=lambda pair: {"stub": True})
    eng.macro = FakeMacro()
    eng._test_macro = eng.macro
    eng.htf_analyzer = htf
    eng.validator = validator
    eng.verifier = FakeVerifier(validator)
    eng.matcher = FakeMatcher()
    eng.builder = FakeBuilder()
    # _research_data builds the real Groq prompt from raw frames; stub it
    eng._research_data = lambda *a, **k: SimpleNamespace()
    return eng


class TestMirrorRetry:
    def test_flipped_direction_gets_full_evaluation(self, temp_db):
        validator = FakeValidator()
        eng = _engine_for_mirror(validator, FakeHTFAnalyzer(ok_dir="buy"))
        verdict = eng.evaluate("EURJPY", "sell", "trend_follow", 130.0,
                               130.5, 129.5)
        # doomed sell dies at step3 (never reaches the checklist); the
        # mirrored buy gets the full evaluation and is approved
        assert eng._test_macro.calls == ["sell", "buy"]
        assert validator.directions == ["buy"]
        assert verdict.approved is True
        assert verdict.direction == "buy"

    def test_htf_conflict_no_retry(self, temp_db):
        """daily=up h4=bearish: both directions blocked by design."""
        validator = FakeValidator()
        eng = _engine_for_mirror(validator, FakeHTFAnalyzer(agree=False))
        verdict = eng.evaluate("EURJPY", "sell", "trend_follow", 130.0,
                               130.5, 129.5)
        # exactly one evaluate() — no mirror attempted on a conflicted regime
        assert eng._test_macro.calls == ["sell"]
        assert verdict.approved is False
        assert verdict.reason.startswith("step3 HTF disagree")

    def test_no_infinite_recursion(self, temp_db):
        """A mirrored evaluation must never mirror again."""
        validator = FakeValidator()
        # HTF agrees with neither direction: the mirror's own step3 also fails
        eng = _engine_for_mirror(validator, FakeHTFAnalyzer(ok_dir="impossible"))
        verdict = eng.evaluate("EURJPY", "sell", "trend_follow", 130.0,
                               130.5, 129.5)
        # exactly 2 evaluations (original + one mirror), then the guard stops it
        assert eng._test_macro.calls == ["sell", "buy"]
        assert verdict.approved is False

    def test_mirror_prices_flip_levels(self):
        from research.research_engine import ResearchEngine
        # sell setup: entry 100, sl 110, tp 80 -> buy mirror: sl 90, tp 120
        assert ResearchEngine._mirror_prices(100.0, 110.0, 80.0, "buy") == \
            (100.0, 90.0, 120.0)
        # and back again
        assert ResearchEngine._mirror_prices(100.0, 90.0, 120.0, "sell") == \
            (100.0, 110.0, 80.0)
        # degenerate levels pass through untouched
        assert ResearchEngine._mirror_prices(0.0, 0.0, 0.0, "buy") == (0.0, 0.0, 0.0)
