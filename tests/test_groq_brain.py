"""Tests for the Groq brain, verifier and research-engine consensus."""

import json

import pytest

from ai.groq_brain import GroqBrain
from ai.groq_verifier import GroqVerifier


class FakeBrain:
    """Scripted brain for verifier tests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.available = True

    def ask_json(self, prompt, pair="", stage="", temperature=0.0,
                 max_retries=None):
        self.calls += 1
        if self.responses:
            item = self.responses.pop(0)
            return item() if callable(item) else item
        return None

    def verified_decision(self, pair, prompt, max_attempts=3):
        """Bypass verification: hand out scripted decisions directly."""
        self.calls += 1
        if self.responses:
            item = self.responses.pop(0)
            return item() if callable(item) else item
        return None


VALID_RESPONSE = {
    "decision": "buy",
    "conviction": 85,
    "sl": 1.0950, "tp1": 1.1050, "tp2": 1.1100, "tp3": 1.1150,
    "reasoning": {
        "macro_case": "USD bias +20 and EUR bias -15, COT percentile 80",
        "structure_case": "H4 EMA50 1.09000 above EMA200 1.08500",
        "entry_case": "RSI14 42.5 with spread 1.2 pips acceptable",
        "institutional_case": "VPOC yesterday 1.09300, price above",
        "against_case": "sell-side liquidity 1.08900 only 10 pips below",
    },
    "invalidation": "1.0930",
    "regime": "trending",
}


class TestGroqBrain:
    def test_json_extraction(self):
        raw = 'Sure! ```json\n{"a": 1}\n``` hope that helps'
        assert GroqBrain._extract_json(raw) == {"a": 1}

    def test_json_extraction_plain(self):
        assert GroqBrain._extract_json('{"x": {"y": 2}}') == {"x": {"y": 2}}

    def test_json_extraction_garbage(self):
        assert GroqBrain._extract_json("no json here") is None

    def test_unavailable_without_key(self, monkeypatch):
        from ai import groq_brain
        monkeypatch.setattr(groq_brain.settings, "GROQ_API_KEY", "")
        brain = GroqBrain(api_key="")
        assert not brain.available

    def test_prompt_contains_data_sections(self, temp_db):
        from ai.groq_prompt_builder import ResearchData
        from ai.groq_prompt_builder import GroqPromptBuilder

        d = ResearchData(pair="EURUSD", timestamp="2026-01-05 08:00 UTC",
                         bid=1.08450, ask=1.08462, spread_pips=1.2, session="London",
                         minutes_in_session=60, proposed_direction="buy",
                         strategy_name="london_breakout")
        prompt = GroqPromptBuilder().build(d)
        assert "MACRO INTELLIGENCE" in prompt
        assert "EURUSD" in prompt
        assert "decision" in prompt  # JSON contract included


class TestVerifier:
    def test_accepts_traceable_response(self, temp_db):
        prompt = ("USD bias: +20/100 | EUR bias: -15/100 | RSI14: 42.5 | "
                  "H4 EMA50: 1.09000 EMA200: 1.08500 | VPOC yesterday: 1.09300 | "
                  "sell-side liquidity: 1.08900 (10.0 pips) | spread 1.2 pips")
        verifier = GroqVerifier(FakeBrain([VALID_RESPONSE]))
        result = verifier.verified_decision("EURUSD", prompt, max_attempts=1)
        assert result is not None
        assert result["decision"] == "buy"

    def test_rejects_hallucinated_numbers(self, temp_db):
        bad = json.loads(json.dumps(VALID_RESPONSE))
        bad["reasoning"]["macro_case"] = "USD bias +77 and retail 63.8% long"
        prompt = ("USD bias: +20/100 | RSI14: 42.5 | spread 1.2 pips | "
                  "VPOC 1.09300 | liquidity 1.08900")
        verifier = GroqVerifier(FakeBrain([bad]))
        result = verifier.verified_decision("EURUSD", prompt, max_attempts=1)
        assert result is None  # score below threshold -> rejected

    def test_skip_passes_immediately(self, temp_db):
        skip = {"decision": "skip", "conviction": 0,
                "reasoning": {"macro_case": "spread 1.2 pips too wide",
                              "structure_case": "VPOC 1.09300 far",
                              "entry_case": "RSI14 50 neutral",
                              "institutional_case": "COT percentile 80 extreme",
                              "against_case": "liquidity 1.08900 near"}}
        verifier = GroqVerifier(FakeBrain([skip]))
        result = verifier.verified_decision("EURUSD", "spread 1.2 pips RSI14 50 "
                                            "VPOC 1.09300 COT percentile 80 "
                                            "liquidity 1.08900",
                                            max_attempts=1)
        assert result["decision"] == "skip"

    def test_forced_skip_after_max_rejections(self, temp_db):
        from config import settings
        bad = json.loads(json.dumps(VALID_RESPONSE))
        bad["reasoning"]["macro_case"] = "USD bias +77 untraceable"
        bad["reasoning"]["structure_case"] = "EMA50 9.99999 nonsense"
        bad["reasoning"]["entry_case"] = "RSI14 77.777 fabricated"
        bad["reasoning"]["institutional_case"] = "VPOC 8.88888 invented"
        bad["reasoning"]["against_case"] = "liquidity 12.34567 bogus"
        brain = FakeBrain([bad] * (settings.GROQ_MAX_REJECTIONS + 1))
        verifier = GroqVerifier(brain, max_consecutive=settings.GROQ_MAX_REJECTIONS)
        result = verifier.verified_decision("EURUSD", "RSI14 42.5 spread 1.2 pips",
                                            max_attempts=settings.GROQ_MAX_REJECTIONS + 1)
        assert result is not None
        assert result.get("_forced_skip") and result["decision"] == "skip"


class TestResearchConsensus:
    def _engine(self, verifier):
        from research.research_engine import ResearchEngine

        return ResearchEngine(data=None, verifier=verifier)

    def test_any_skip_wins(self, temp_db):
        engine = self._engine(None)
        buy = dict(VALID_RESPONSE, decision="buy", conviction=80)
        engine.verifier = FakeBrain([dict(buy), dict(buy),
                                     dict(VALID_RESPONSE, decision="skip")])
        # skip wins per master prompt: any skip -> no trade
        result = engine._groq_consensus("EURUSD", "prompt")
        assert result["decision"] == "skip"

    def test_conviction_floor_blocks(self, temp_db):
        from config import settings
        engine = self._engine(None)
        low = [dict(VALID_RESPONSE, conviction=settings.GROQ_MIN_CONVICTION - 10)
               for _ in range(3)]
        engine.verifier = FakeBrain(low)
        assert engine._groq_consensus("EURUSD", "prompt") is None

    def test_two_agreeing_reach_full_consensus(self, temp_db):
        engine = self._engine(None)
        engine.verifier = FakeBrain([dict(VALID_RESPONSE), dict(VALID_RESPONSE)])
        result = engine._groq_consensus("EURUSD", "prompt")
        assert result["_consensus"] == 2

    def test_no_verifier_returns_none(self, temp_db):
        engine = self._engine(None)
        assert engine._groq_consensus("EURUSD", "prompt") is None
