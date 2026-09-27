"""Groq brain: strict JSON-mode calls with retry, verbatim audit and health check.

Consolidates the prompt-2 client. Every request/response pair is stored in
groq_audit; parse failures are retried up to 3 times then return None.
"""

import json
import time
from typing import Optional

import requests

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_INSTRUCTION = (
    "You are a quantitative Forex analyst operating on verified market data only. "
    "Do not use training data. Do not make assumptions. Cite every claim with a "
    "specific number from the data below. If data is insufficient return skip."
)


class GroqBrain:
    """Groq chat wrapper with JSON enforcement."""

    def __init__(self, api_key: str = "", model: str = "", max_json_retries: int = 3) -> None:
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.max_json_retries = max_json_retries
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update(
                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    @property
    def available(self) -> bool:
        """True when configured."""
        return bool(self.api_key)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Pull the first JSON object from model output."""
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    def _raw_call(self, prompt: str, temperature: float, stage: str, pair: str) -> tuple[Optional[str], int]:
        """One HTTP call with audit persistence; (raw, latency_ms)."""
        if not self.available:
            return None, 0
        payload = {
            "model": self.model, "temperature": temperature, "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_INSTRUCTION},
                         {"role": "user", "content": prompt}],
        }
        started = time.monotonic()
        try:
            resp = self.session.post(GROQ_URL, json=payload, timeout=30)
            resp.raise_for_status()
            latency = int((time.monotonic() - started) * 1000)
            content = resp.json()["choices"][0]["message"]["content"]
            self._audit(pair, stage, prompt, content, attempt=1, temperature=temperature,
                        latency=latency)
            return content, latency
        except Exception as exc:
            latency = int((time.monotonic() - started) * 1000)
            logger.error("groq call failed (%s): %s", stage, exc)
            self._audit(pair, stage, prompt, f"ERROR: {exc}", 1, temperature, latency)
            return None, latency

    @staticmethod
    def _audit(pair: str, stage: str, prompt: str, response: str, attempt: int,
               temperature: float, latency: int, accepted: bool = False,
               parsed: str = "", reject_reason: str = "", score: Optional[float] = None) -> None:
        """Persist verbatim to groq_audit."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext(
                "INSERT INTO groq_audit (created_at, pair, stage, prompt, response, parsed_json, "
                "attempt, temperature, accepted, reject_reason, verify_score, latency_ms) "
                "VALUES (:t, :p, :st, :pr, :re, :pa, :at, :te, :ac, :rr, :sc, :la)"
            ), {"t": db._utcnow(), "p": pair, "st": stage, "pr": prompt[:8000],
                "re": response[:8000], "pa": parsed[:8000], "at": attempt, "te": temperature,
                "ac": accepted, "rr": reject_reason[:255], "sc": score, "la": latency})

    def ask_json(self, prompt: str, pair: str = "", stage: str = "call",
                 temperature: float = 0.0, max_retries: Optional[int] = None) -> Optional[dict]:
        """Ask Groq for JSON; retries on parse failure; None when exhausted."""
        retries = self.max_json_retries if max_retries is None else max_retries
        for attempt in range(1, retries + 1):
            raw, _ = self._raw_call(prompt, temperature, stage, pair)
            if raw is None:
                continue
            parsed = self._extract_json(raw)
            if parsed is not None:
                self._audit(pair, f"{stage}:parsed", "", "", attempt, temperature, 0,
                            accepted=True, parsed=json.dumps(parsed))
                return parsed
            logger.warning("groq json parse failed (attempt %d/%d) %s", attempt, retries, stage)
            self._audit(pair, stage, prompt, raw, attempt, temperature, 0,
                        reject_reason="json_parse_failed")
        return None

    def health_check(self) -> bool:
        """Cheap availability probe."""
        if not self.available:
            return False
        try:
            resp = self.session.post(
                GROQ_URL,
                json={"model": self.model, "messages": [{"role": "user", "content": "ping"}],
                      "max_tokens": 1},
                timeout=10,
            )
            ok = resp.status_code == 200
            db.log_feed_health("groq", ok, f"status={resp.status_code}")
            return ok
        except Exception as exc:
            logger.error("groq health check failed: %s", exc)
            db.log_feed_health("groq", False, str(exc)[:200])
            return False
