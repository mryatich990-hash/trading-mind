"""GroqVerifier: fact-checks Groq responses against the exact prompt numbers.

Rounding-aware numeric trace-back (prompts show rounded values, so 1.0845
matches a data value of 1.084523). Score = share of traceable claims; below
threshold -> reject, log, retry. 5 consecutive rejections -> hard breaker.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["VerificationResult", "GroqVerifier"]


@dataclass
class VerificationResult:
    """Verification outcome."""

    accepted: bool
    score: float
    failed_claims: list[str] = field(default_factory=list)
    reason: str = ""


class GroqVerifier:
    """Numeric trace-back verification."""

    EVIDENCE_FIELDS = ("macro_case", "structure_case", "entry_case",
                       "institutional_case", "against_case")
    IGNORED = {0.0, 100.0, 1.0, 2.0, 3.0}
    PARAM_FIELDS = ("sl", "tp1", "tp2", "tp3", "invalidation")

    def __init__(self, brain, threshold: int = 0, max_consecutive: int = 0) -> None:
        self.brain = brain
        self.threshold = threshold or settings.GROQ_VERIFY_THRESHOLD
        self.max_consecutive = max_consecutive or settings.GROQ_MAX_REJECTIONS
        self._consecutive = 0

    @staticmethod
    def _numbers_with_decimals(text: str) -> list[tuple[float, int]]:
        """(value, decimal places) pairs."""
        out = []
        for m in re.finditer(r"-?\d+(?:\.(\d+))?", text):
            try:
                out.append((float(m.group(0)), len(m.group(1) or "")))
            except ValueError:
                continue
        return out

    @classmethod
    def _traceable(cls, value: float, decimals: int, universe: set[float]) -> bool:
        """Match at the value's own printed precision (half-ULP)."""
        tol = 0.5 * (10 ** -decimals) + 1e-12
        return any(abs(value - u) <= tol for u in universe)

    def _derived(self, prompt: str, response: dict) -> set[float]:
        """Differences in pip units + the response's own price parameters."""
        nums = [v for v, _ in self._numbers_with_decimals(prompt)]
        derived: set[float] = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                diff = abs(nums[i] - nums[j])
                for pip in (0.0001, 0.01, 0.1, 1.0):
                    derived.add(diff / pip)
        for key in self.PARAM_FIELDS:
            if response.get(key) is not None:
                derived.update(v for v, _ in self._numbers_with_decimals(str(response[key])))
        return derived

    def verify(self, response: dict, prompt: str) -> VerificationResult:
        """Score factual accuracy of a decision response."""
        decision = str(response.get("decision", "")).lower()
        if decision not in ("buy", "sell", "skip"):
            return VerificationResult(False, 0.0, [], "invalid decision")
        try:
            conv = int(float(response.get("conviction", response.get("confidence", -1))))
            if not 0 <= conv <= 100:
                return VerificationResult(False, 0.0, [], "conviction out of range")
        except (TypeError, ValueError):
            return VerificationResult(False, 0.0, [], "conviction not numeric")

        universe = {v for v, _ in self._numbers_with_decimals(prompt)} | self._derived(prompt, response)
        reasoning = response.get("reasoning") or {}
        claims: list[str] = []
        if isinstance(reasoning, dict):
            for key in self.EVIDENCE_FIELDS:
                val = reasoning.get(key)
                if isinstance(val, str) and val.strip():
                    claims.append(val.strip())
                elif val is None:
                    claims.append("__MISSING__")
        failed: list[str] = []
        for claim in claims:
            if claim == "__MISSING__":
                failed.append("missing evidence field")
                continue
            numbers = [(v, d) for v, d in self._numbers_with_decimals(claim)
                       if v not in self.IGNORED]
            if not numbers:
                failed.append(f"no numeric evidence: {claim[:70]}")
                continue
            untraceable = [v for v, d in numbers if not self._traceable(v, d, universe)]
            if untraceable:
                failed.append(f"untraceable {[round(v, 5) for v in untraceable[:3]]} in: {claim[:70]}")

        if decision == "skip" and not failed:
            return VerificationResult(True, 100.0, [], "")
        if any("missing evidence" in f for f in failed):
            return VerificationResult(False, 0.0, failed, "missing evidence field")
        total = len(claims) or 1
        score = round((total - len(failed)) / total * 100.0, 1)
        if not failed:
            return VerificationResult(True, score, [], "")
        if score >= self.threshold:
            return VerificationResult(True, score, failed, "")
        return VerificationResult(False, score, failed, f"score {score:.0f} < {self.threshold}")

    def verified_decision(self, pair: str, prompt: str, max_attempts: int = 3) -> Optional[dict]:
        """ask -> verify -> retry loop; forced skip after consecutive failures."""
        for attempt in range(1, max_attempts + 1):
            response = self.brain.ask_json(pair=pair, stage=f"research:{attempt}",
                                           prompt=prompt, temperature=0.0)
            if response is None:
                self._register(pair, "json_unparseable", 0.0, [], "")
                continue
            result = self.verify(response, prompt)
            if result.accepted:
                self._consecutive = 0
                return response
            self._register(pair, result.reason, result.score, result.failed_claims,
                           json.dumps(response)[:2000])
        if self._consecutive >= self.max_consecutive:
            db.log_breaker("groq_rejections",
                           f"{self._consecutive} consecutive factual rejections", "halt")
            return {"decision": "skip", "conviction": 0,
                    "reasoning": {"against_case": "unverifiable AI output"},
                    "_forced_skip": True}
        return None

    def _register(self, pair: str, reason: str, score: float,
                  claims: list[str], raw: str) -> None:
        """Persist rejection and bump counter."""
        self._consecutive += 1
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext(
                "INSERT INTO groq_rejections (created_at, pair, reason, score, failed_claims, "
                "raw_response) VALUES (:t, :p, :r, :s, :c, :raw)"
            ), {"t": db._utcnow(), "p": pair, "r": reason[:255], "s": score,
                "c": json.dumps(claims)[:2000], "raw": raw})
        logger.warning("groq rejected for %s: %s (score %.0f)", pair, reason, score)
