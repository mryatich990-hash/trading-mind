"""Tests for the operator TEMP unblock window (confluence cap, Groq
conviction floor, verifier floor, breaker allowlist) and the forced
end-to-end test trade. The window auto-expires via TEMP_WINDOW_STARTED_AT."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from config import settings
from core import db as core_db


# ---------------------------------------------------------------------------
# window activation helper
# ---------------------------------------------------------------------------

def _arm_window(monkeypatch, hours=24.0, started=None):
    started = started or datetime.now(timezone.utc)
    monkeypatch.setattr(settings, "TEMP_WINDOW_STARTED_AT", started.isoformat())
    monkeypatch.setattr(settings, "TEMP_WINDOW_HOURS", hours)
    monkeypatch.setattr(settings, "TEMP_MIN_CONFLUENCE", 5)
    monkeypatch.setattr(settings, "TEMP_GROQ_MIN_CONVICTION", 60)
    monkeypatch.setattr(settings, "TEMP_GROQ_VERIFY_THRESHOLD", 65)
    monkeypatch.setattr(settings, "TEMP_BREAKERS_ALLOWLIST",
                        "daily_loss,drawdown_halt,margin_halt")


# ---------------------------------------------------------------------------
# window lifecycle
# ---------------------------------------------------------------------------

class TestWindowLifecycle:
    def test_inactive_without_start(self):
        assert settings.temp_min_confluence() == 0
        assert settings.temp_groq_min_conviction() == 0
        assert settings.temp_breaker_allowlist() == []

    def test_active_when_armed(self, monkeypatch):
        _arm_window(monkeypatch)
        assert settings.temp_min_confluence() == 5
        assert settings.temp_groq_min_conviction() == 60
        assert settings.temp_groq_verify_threshold() == 65
        assert settings.temp_breaker_allowlist() == \
            ["daily_loss", "drawdown_halt", "margin_halt"]

    def test_expires_after_window(self, monkeypatch):
        stale = datetime.now(timezone.utc) - timedelta(hours=25)
        _arm_window(monkeypatch, started=stale)
        assert settings.temp_min_confluence() == 0
        assert settings.temp_groq_min_conviction() == 0
        assert settings.temp_breaker_allowlist() == []

    def test_malformed_start_fails_closed(self, monkeypatch):
        monkeypatch.setattr(settings, "TEMP_WINDOW_STARTED_AT", "not-a-date")
        assert settings.temp_min_confluence() == 0
        assert settings.temp_breaker_allowlist() == []

    def test_effective_verify_threshold(self, monkeypatch):
        assert settings.effective_groq_verify_threshold() == \
            settings.GROQ_VERIFY_THRESHOLD
        _arm_window(monkeypatch)
        assert settings.effective_groq_verify_threshold() == 65


# ---------------------------------------------------------------------------
# breaker allowlist behavior
# ---------------------------------------------------------------------------

class TestBreakerAllowlist:
    def test_capital_breakers_survive_window(self, temp_db, monkeypatch):
        _arm_window(monkeypatch)
        core_db.log_breaker("vix_halt", "test", "halt")
        core_db.log_breaker("daily_loss", "-3.1% today", "halt")
        active = core_db.unresolved_breakers()
        assert "daily_loss" in active
        assert "vix_halt" not in active

    def test_all_suppressed_outside_window(self, temp_db):
        core_db.log_breaker("vix_halt", "test", "halt")
        assert "vix_halt" in core_db.unresolved_breakers()

    def test_observe_severity_suppressed_in_window(self, temp_db, monkeypatch):
        _arm_window(monkeypatch)
        core_db.log_breaker("data_stale", "test", "observe")
        assert core_db.unresolved_breakers("observe") == []


# ---------------------------------------------------------------------------
# confluence cap + conviction floor in the engine
# ---------------------------------------------------------------------------

class _FakeMacro:
    def __init__(self):
        self.calls = []

    def analyze(self, pair, direction):
        self.calls.append(direction)
        return SimpleNamespace(blocked=False, size_multipliers={},
                               news_gate="clear", pattern_name="none",
                               harmonic_name="none", notes=[])


class _FakeMatcher:
    def match(self, *a, **k):
        return SimpleNamespace(required_confluence=settings.MIN_CONFLUENCE,
                               win_rate=60.0, samples=30)


class _FakeValidator:
    def __init__(self, score):
        self.score = score
        self.directions = []

    def validate(self, pair, direction, frames, htf, macro, confluence_bonus=0):
        self.directions.append(direction)
        from research.entry_validator import EntryChecklist
        return EntryChecklist(score=self.score, required=8)


class _FakeBuilder:
    def build(self, research_data):
        return "prompt"


class _FakeVerifier:
    def __init__(self, validator, conviction):
        self._v = validator
        self.conviction = conviction

    def verified_decision(self, pair, prompt, max_attempts=1):
        return {"decision": self._v.directions[-1],
                "conviction": self.conviction, "_consensus": 3}


class _FakeHTF:
    def analyze(self, pair, frames, direction):
        from research.htf_analyzer import HTFResult
        return HTFResult(daily_trend="up", h4_bias="bullish",
                         htf_agree=True, direction_ok=True)


def _engine(score, conviction):
    from research.research_engine import ResearchEngine

    validator = _FakeValidator(score)
    eng = ResearchEngine.__new__(ResearchEngine)
    eng.data = SimpleNamespace(get_frames=lambda pair: {"stub": True})
    eng.macro = _FakeMacro()
    eng._test_macro = eng.macro
    eng.htf_analyzer = _FakeHTF()
    eng.validator = validator
    eng.verifier = _FakeVerifier(validator, conviction)
    eng.matcher = _FakeMatcher()
    eng.builder = _FakeBuilder()
    eng._research_data = lambda *a, **k: SimpleNamespace()
    return eng


class TestWindowInEngine:
    def test_window_lets_6_of_8_trade_on_conviction_60(self, temp_db, monkeypatch):
        """6/8 + conviction 60: impossible at steady state, approved in window."""
        _arm_window(monkeypatch)
        eng = _engine(score=6, conviction=60)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is True
        assert verdict.confluence == 6  # risk gate reads this; must not be 0

    def test_confluence_always_populated(self, temp_db):
        """Regression: verdict.confluence stayed 0 forever, so the risk gate
        rejected every organic approval with 'confluence 0 < required 8'."""
        eng = _engine(score=7, conviction=90)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.confluence == 7  # set even on the rejection path
        assert verdict.approved is False  # 7 < 8 floor out of window

    def test_below_window_floor_still_rejected(self, temp_db, monkeypatch):
        _arm_window(monkeypatch)
        eng = _engine(score=4, conviction=90)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is False
        assert verdict.reason.startswith("step6 confluence 4/5")  # floor capped to 5

    def test_low_conviction_rejected_in_window(self, temp_db, monkeypatch):
        _arm_window(monkeypatch)
        eng = _engine(score=6, conviction=55)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is False

    def test_history_floor_not_capped(self, temp_db, monkeypatch):
        """hist.required_confluence=9 survives the cap (cap only relaxes the
        operator floor, not data-driven requirements)."""
        _arm_window(monkeypatch)

        class Strict(_FakeMatcher):
            def match(self, *a, **k):
                return SimpleNamespace(required_confluence=9,
                                       win_rate=40.0, samples=30)

        eng = _engine(score=6, conviction=90)
        eng.matcher = Strict()
        verdict = eng.evaluate("EURUSD", "buy", "reluctant", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is False
        assert verdict.reason.startswith("step6 confluence 6/9")

    def test_window_closed_restores_steady_state(self, temp_db, monkeypatch):
        stale = datetime.now(timezone.utc) - timedelta(hours=25)
        _arm_window(monkeypatch, started=stale)
        eng = _engine(score=6, conviction=90)
        verdict = eng.evaluate("EURUSD", "buy", "trend_follow", 1.1000,
                               1.0980, 1.1060)
        assert verdict.approved is False


# ---------------------------------------------------------------------------
# verifier floor in window
# ---------------------------------------------------------------------------

class TestRiskGateWindow:
    def test_risk_gate_honors_cap(self, temp_db, monkeypatch):
        """Risk gate check 5 applies the same TEMP cap (5) as research."""
        from risk.risk_manager import RiskManager
        _arm_window(monkeypatch)
        rm = RiskManager()
        kwargs = dict(entry=1.1000, sl=1.0980, tp=1.1060, balance=10000.0,
                      equity=10000.0, used_margin=0.0, usdjpy=0.0,
                      open_positions=[], vix=0.0, spread_pips=1.0,
                      mode="demo", size_multipliers={})
        decision = rm.evaluate("EURUSD", "buy", "trend_follow", confluence=5,
                               **kwargs)
        assert "confluence" not in decision.reason

    def test_risk_gate_full_floor_outside_window(self, temp_db):
        from risk.risk_manager import RiskManager
        rm = RiskManager()
        decision = rm.evaluate("EURUSD", "buy", "trend_follow", confluence=5,
                               entry=1.1000, sl=1.0980, tp=1.1060,
                               balance=10000.0, equity=10000.0,
                               mode="demo")
        assert decision.approved is False
        assert "confluence 5 < required" in decision.reason


class TestVerifierWindow:
    def test_low_score_accepted_in_window(self, monkeypatch):
        from ai.groq_verifier import GroqVerifier
        _arm_window(monkeypatch)
        v = GroqVerifier(SimpleNamespace(), threshold=82)  # boot threshold 82
        prompt = "price 1.1000 sl 1.0980 tp 1.1060"
        response = {"decision": "buy", "conviction": 60,
                    "reasoning": {"macro_case": "spread near 1.1000 favors bids",
                                  "structure_case": "support holds above 1.0980",
                                  "entry_case": "enter 1.1000 stop 1.0980",
                                  "institutional_case": "flow favors bids below 1.1060",
                                  "against_case": "risk if 1.0980 breaks"}}
        result = v.verify(response, prompt)
        assert result.accepted is True

    def test_low_score_rejected_outside_window(self, monkeypatch):
        from ai.groq_verifier import GroqVerifier
        v = GroqVerifier(SimpleNamespace(), threshold=82)
        prompt = "price 1.1000 sl 1.0980 tp 1.1060"
        response = {"decision": "buy", "conviction": 60,
                    "reasoning": {"for_case": "vibes look great today",
                                  "institutional_case": None,
                                  "against_case": None}}
        result = v.verify(response, prompt)
        assert result.accepted is False


# ---------------------------------------------------------------------------
# forced test trade (handler-level, execution fully stubbed)
# ---------------------------------------------------------------------------

class TestForcedTrade:
    def _system(self, temp_db, monkeypatch, fill=True):
        from main import TradingSystem

        ts = TradingSystem.__new__(TradingSystem)
        ts.data = SimpleNamespace(get_candles=lambda p, t, c: SimpleNamespace(
            last_close=1.1000))
        ts.risk = SimpleNamespace(evaluate=lambda *a, **k: SimpleNamespace(
            approved=True, lots=Decimal("0.01"), reason=""))
        captured = {}

        def fake_execute(verdict, lots, balance, confluence):
            captured["verdict"] = verdict
            captured["lots"] = lots
            return 4242 if fill else None

        ts.execution = SimpleNamespace(execute=fake_execute,
                                       active_broker=lambda: object())
        ts._active_broker_or_none = lambda: None
        ts._account_snapshot = lambda broker: (10000.0, 10000.0, 0.0)
        ts._captured = captured
        return ts

    def test_forced_trade_executes_eurusd_001(self, temp_db, monkeypatch):
        ts = self._system(temp_db, monkeypatch)
        core_db.set_state("force_trade_requested", "1")
        ts._handle_forced_trade()
        assert core_db.get_state("force_trade_requested", "0") == "0"
        v = ts._captured["verdict"]
        assert v.pair == "EURUSD" and v.direction == "buy"
        assert v.strategy == "forced_test_trade"
        assert float(ts._captured["lots"]) == 0.01
        assert v.sl < v.entry < v.tp1

    def test_risk_denial_logged_and_cleared(self, temp_db, monkeypatch):
        ts = self._system(temp_db, monkeypatch)
        ts.risk = SimpleNamespace(evaluate=lambda *a, **k: SimpleNamespace(
            approved=False, lots=Decimal("0"), reason="halted by breakers: x"))
        core_db.set_state("force_trade_requested", "1")
        ts._handle_forced_trade()
        assert core_db.get_state("force_trade_requested", "0") == "0"
        assert "verdict" not in ts._captured  # never reached execution

    def test_execute_failure_clears_trigger(self, temp_db, monkeypatch):
        ts = self._system(temp_db, monkeypatch, fill=False)
        core_db.set_state("force_trade_requested", "1")
        ts._handle_forced_trade()
        assert core_db.get_state("force_trade_requested", "0") == "0"

    def test_exception_still_clears_trigger(self, temp_db, monkeypatch):
        ts = self._system(temp_db, monkeypatch)
        ts.data = SimpleNamespace(get_candles=lambda *a: (_ for _ in ()).throw(
            RuntimeError("feed down")))
        core_db.set_state("force_trade_requested", "1")
        ts._handle_forced_trade()  # must not raise
        assert core_db.get_state("force_trade_requested", "0") == "0"
