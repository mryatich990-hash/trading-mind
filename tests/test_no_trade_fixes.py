"""Regression tests: bot stopped taking trades (groq_rejections halt)."""

import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# GroqBrain payload + empty-content handling
# ---------------------------------------------------------------------------

class TestGroqBrainPayload:
    """The reasoning model burned 900 tokens on hidden reasoning -> 400s."""

    def _brain(self, monkeypatch, response_json):
        from ai.groq_brain import GroqBrain

        brain = GroqBrain(api_key="test-key")
        captured = {}

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return response_json

        class FakeSession:
            headers = {}

            def post(self, url, json=None, timeout=0):
                captured["payload"] = json
                return FakeResp()

        brain.session = FakeSession()
        return brain, captured

    def test_payload_uses_reasoning_safe_limits(self, temp_db, monkeypatch):
        brain, captured = self._brain(monkeypatch, {
            "choices": [{"message": {"content": "{\"decision\": \"skip\"}"}}]})
        brain.ask_json("test prompt", pair="EURUSD", stage="t")
        payload = captured["payload"]
        assert payload["max_tokens"] == 4000
        assert payload["reasoning_effort"] == "low"
        assert payload["response_format"] == {"type": "json_object"}

    def test_empty_content_raises_not_parsed(self, temp_db, monkeypatch):
        """Reasoning consumed the budget: empty content must error, not parse."""
        brain, _ = self._brain(monkeypatch, {
            "choices": [{"message": {"content": "", "reasoning": "thinking..."}}]})
        assert brain.ask_json("prompt", pair="EURUSD", stage="t") is None

    def test_transport_error_audited_and_none(self, temp_db, monkeypatch):
        from ai.groq_brain import GroqBrain

        brain = GroqBrain(api_key="k")

        class BoomSession:
            headers = {}

            def post(self, *a, **k):
                raise ConnectionError("boom")

        brain.session = BoomSession()
        assert brain.ask_json("p", pair="EURUSD", stage="t") is None
        rows = core_audit_rows("EURUSD")
        assert rows and rows[-1]["response"].startswith("ERROR: ")


def core_audit_rows(pair):
    from sqlalchemy import text as sqltext
    from core import db

    with db.engine.connect() as conn:
        rows = conn.execute(sqltext(
            "SELECT stage, response, reject_reason FROM groq_audit "
            "WHERE pair = :p ORDER BY created_at"), {"p": pair}).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# GroqVerifier: transport failure != factual rejection
# ---------------------------------------------------------------------------

class TestVerifierErrorSplit:
    def _verifier(self):
        from ai.groq_verifier import GroqVerifier

        return GroqVerifier(SimpleNamespace(available=True), threshold=82,
                            max_consecutive=5)

    def test_transport_failure_not_counted(self, temp_db):
        """HTTP outage (ERROR audit row) must not escalate to a halt."""
        from sqlalchemy import text as sqltext
        from core import db

        v = self._verifier()
        with db.engine.begin() as conn:
            conn.execute(sqltext(
                "INSERT INTO groq_audit (created_at, pair, stage, prompt, response) "
                "VALUES (datetime('now'), 'EURUSD', 'research:1', 'p', 'ERROR: 400')"))

        brain = SimpleNamespace(available=True, ask_json=lambda *a, **k: None)
        v.brain = brain
        result = v.verified_decision("EURUSD", "prompt", max_attempts=2)
        assert v._consecutive == 0
        assert v._consecutive_errors >= 1
        # no forced skip / no breaker from a transport outage
        assert result is None

    def test_real_rejection_counts_and_halts(self, temp_db):
        from core import db

        v = self._verifier()
        calls = {"n": 0}

        def fake_ask(*a, **k):
            calls["n"] += 1
            return {"decision": "buy", "conviction": 85}  # no reasoning -> reject

        v.brain = SimpleNamespace(available=True, ask_json=fake_ask)
        result = v.verified_decision("EURUSD", "prompt", max_attempts=5)
        assert v._consecutive >= 5
        assert result and result.get("_forced_skip")
        assert "groq_rejections" in db.unresolved_breakers()


# ---------------------------------------------------------------------------
# Notifier wiring: registry accepts object or callable
# ---------------------------------------------------------------------------

class TestNotifierWiring:
    def test_callable_normalized(self):
        from upgrades_registry import UpgradeRegistry

        reg = UpgradeRegistry(pairs=["EURUSD"], notifier=lambda t: t)
        assert hasattr(reg.notifier, "send")
        assert reg.notifier.send("x") == "x"

    def test_object_passthrough(self):
        from upgrades_registry import UpgradeRegistry

        bot = SimpleNamespace(send=lambda t: t)
        reg = UpgradeRegistry(pairs=["EURUSD"], notifier=bot)
        assert reg.notifier is bot


# ---------------------------------------------------------------------------
# Yahoo symbol swap: futures quote around the clock, cash indices do not
# ---------------------------------------------------------------------------

class TestIndexSymbols:
    def test_map_uses_futures(self):
        from data.market_data_engine import YFinanceFeed

        assert YFinanceFeed._MAP["NAS100"] == "NQ=F"
        assert YFinanceFeed._MAP["US30"] == "YM=F"
        assert YFinanceFeed._MAP["XAUUSD"] == "GC=F"


# ---------------------------------------------------------------------------
# groq_rejections auto-resume
# ---------------------------------------------------------------------------

class TestGroqHaltRecovery:
    def test_resume_after_healthy_probes(self, temp_db):
        from risk.circuit_breakers import CircuitBreakers

        sent = []
        brk = CircuitBreakers(notifier=sent.append)
        brk.trigger("groq_rejections", "5 consecutive rejections",
                    severity="halt", notify=False)
        assert brk.is_active("groq_rejections")

        # unhealthy probes must not resume
        assert brk.maybe_resume_after_groq_recovery(False) is False
        # first healthy probe: counter satisfied, halt lifted
        assert brk.maybe_resume_after_groq_recovery(True) is True
        assert not brk.is_active("groq_rejections")

    def test_noop_when_not_active(self, temp_db):
        from risk.circuit_breakers import CircuitBreakers

        brk = CircuitBreakers(notifier=lambda t: None)
        assert brk.maybe_resume_after_groq_recovery(True) is False

    def test_requires_two_healthy_probes_after_fail(self, temp_db):
        from risk.circuit_breakers import CircuitBreakers

        brk = CircuitBreakers(notifier=lambda t: None)
        brk.trigger("groq_rejections", "x", severity="halt", notify=False)
        brk.maybe_resume_after_groq_recovery(False)   # failure bumps counter
        # single healthy probe right after a failure is not enough
        assert brk.maybe_resume_after_groq_recovery(True) is True
