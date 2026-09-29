"""Tests for the first-trade pilot: while the trades table is empty, a setup
short of the static confluence floor may proceed, compensated by a raised Groq
conviction floor. Expiry is the DB state itself; DB errors fail closed."""

from types import SimpleNamespace

import pytest

from config import settings
from core import db as core_db
from research.research_engine import ResearchEngine


# ---------------------------------------------------------------------------
# stubs (same harness as test_feed_resilience_and_mirror.py)
# ---------------------------------------------------------------------------

class FakeMacro:
    def __init__(self):
        self.calls = []

    def analyze(self, pair, direction):
        self.calls.append(direction)
        return SimpleNamespace(blocked=False, size_multipliers={},
                               news_gate="clear", pattern_name="none",
                               harmonic_name="none", notes=[])


class FakeMatcher:
    def match(self, *a, **k):
        return SimpleNamespace(required_confluence=settings.MIN_CONFLUENCE,
                               win_rate=60.0, samples=30)


class FakeValidator:
    """Reports a fixed checklist score."""

    def __init__(self, score: int):
        self.score = score
        self.directions = []

    def validate(self, pair, direction, frames, htf, macro, confluence_bonus=0):
        self.directions.append(direction)
        from research.entry_validator import EntryChecklist
        return EntryChecklist(score=self.score, required=8)


class FakeBuilder:
    def build(self, research_data):
        return "test prompt"


class FakeVerifier:
    def __init__(self, validator: FakeValidator, conviction: int = 76):
        self._validator = validator
        self.conviction = conviction
        self.calls = 0

    def verified_decision(self, pair, prompt, max_attempts=1):
        self.calls += 1
        return {"decision": self._validator.directions[-1],
                "conviction": self.conviction, "_consensus": 3}


class FakeHTFAnalyzer:
    def analyze(self, pair, frames, direction):
        from research.htf_analyzer import HTFResult
        return HTFResult(daily_trend="up", h4_bias="bullish",
                         htf_agree=True, direction_ok=True)


def _engine(score: int, conviction: int):
    from research.research_engine import ResearchEngine

    validator = FakeValidator(score)
    eng = ResearchEngine.__new__(ResearchEngine)
    eng.data = SimpleNamespace(get_frames=lambda pair: {"stub": True})
    eng.macro = FakeMacro()
    eng._test_macro = eng.macro
    eng.htf_analyzer = FakeHTFAnalyzer()
    eng.validator = validator
    eng.verifier = FakeVerifier(validator, conviction)
    eng.matcher = FakeMatcher()
    eng.builder = FakeBuilder()
    eng._research_data = lambda *a, **k: SimpleNamespace()
    return eng


@pytest.fixture()
def pilot_on(monkeypatch):
    monkeypatch.setattr(settings, "FIRST_TRADE_PILOT_ENABLED", True)


# ---------------------------------------------------------------------------
# pilot behavior
# ---------------------------------------------------------------------------

class TestPilotGate:
    def test_pilot_lets_7_of_8_reach_groq_and_pass(self, temp_db, pilot_on):
        """7/8 + conviction 80 clears the raised pilot floor and approves."""
        eng = _engine(score=7, conviction=80)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is True

    def test_below_pilot_confluence_still_rejected(self, temp_db, pilot_on):
        eng = _engine(score=6, conviction=80)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is False
        assert verdict.reason.startswith("step6 confluence 6/8")

    def test_pilot_off_seven_of_eight_rejected(self, temp_db):
        eng = _engine(score=7, conviction=80)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is False
        assert verdict.reason.startswith("step6 confluence 7/8")

    def test_pilot_conviction_floor_binds_only_for_pilot_setups(
            self, temp_db, pilot_on, monkeypatch):
        """The raised floor binds ONLY when the setup needed the pilot. An
        8/8 setup with conviction 76 must still approve (steady-state floor
        74), while a 7/8 setup with the same conviction must not."""
        monkeypatch.setattr(settings, "GROQ_MIN_CONVICTION", 74)
        eng8 = _engine(score=8, conviction=76)
        v8 = eng8.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                           1.0980, 1.1060)
        assert v8.approved is True

        eng7 = _engine(score=7, conviction=76)
        v7 = eng7.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                           1.0980, 1.1060)
        assert v7.approved is False  # 76 < 80 pilot floor
        assert "conviction" in v7.reason or v7.reason == "step9 groq: no consensus"

    def test_pilot_void_after_first_trade(self, temp_db, pilot_on):
        """One trade row voids the pilot forever."""
        core_db.record_trade("EURUSD", "buy", 0.01, 1.1000, 1.0980, 1.1060,
                             strategy="trend_follow", signal_hash="t1")
        eng = _engine(score=7, conviction=80)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is False
        assert verdict.reason.startswith("step6 confluence 7/8")

    def test_db_failure_fails_closed(self, temp_db, pilot_on, monkeypatch):
        """If the trades lookup explodes, the pilot must NOT open the gate."""
        def boom():
            raise RuntimeError("db down")
        monkeypatch.setattr(core_db, "has_any_trades", boom)
        eng = _engine(score=7, conviction=80)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is False

    def test_history_driven_floor_never_relaxed(self, temp_db, pilot_on):
        """hist.required_confluence=9 (poor history) must stay a hard floor."""
        class StrictMatcher(FakeMatcher):
            def match(self, *a, **k):
                return SimpleNamespace(required_confluence=9,
                                       win_rate=40.0, samples=30)

        eng = _engine(score=7, conviction=80)
        eng.matcher = StrictMatcher()
        verdict = eng.evaluate("EURUSD", "buy", "reluctant", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is False
        assert verdict.reason.startswith("step6 confluence 7/9")


class TestHasAnyTrades:
    def test_empty_then_nonempty(self, temp_db):
        assert core_db.has_any_trades() is False
        core_db.record_trade("EURUSD", "sell", 0.01, 1.1000, 1.1020, 1.0940,
                             strategy="t", signal_hash="h1")
        assert core_db.has_any_trades() is True

    def test_never_raises(self, temp_db, monkeypatch):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        broken = create_engine("sqlite:///")  # in-memory, no schema
        monkeypatch.setattr(core_db, "SessionLocal",
                            sessionmaker(bind=broken, future=True))
        assert core_db.has_any_trades() is True  # fail-closed
