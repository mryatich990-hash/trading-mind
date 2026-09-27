"""Prompt evolution: A/B tests Groq prompt wording variants on real outcomes."""

import json
import random
import threading
from typing import Optional

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["PromptEvolver"]

BASE_PROMPT_HEADER = "You are a quantitative Forex analyst operating on verified market data only."


class PromptEvolver:
    """Tracks prompt-version performance and promotes winners after 30 trades."""

    def __init__(self, groq_ask_json: Optional[callable] = None) -> None:
        self.groq = groq_ask_json
        self._lock = threading.Lock()
        self._seed_base_version()

    @staticmethod
    def _seed_base_version() -> None:
        """Ensure version A exists and is active."""
        if db.get_state("prompt_active_version", "") == "":
            from sqlalchemy import text as sqltext

            from core.db import engine, _WRITE_LOCK
            with engine.begin() as conn, _WRITE_LOCK:
                conn.execute(sqltext(
                    "INSERT INTO prompt_versions (created_at, version, prompt, trades, wins, active) "
                    "VALUES (CURRENT_TIMESTAMP, 'A', :p, 0, 0, TRUE)"
                ), {"p": BASE_PROMPT_HEADER})
            db.set_state("prompt_active_version", "A")

    def active_variant(self) -> str:
        """Which prompt version should serve the next call ('A' or 'B')."""
        if db.get_state("ab_prompt_running", "0") != "1":
            return "A"
        return random.choice(("A", "B"))

    def record_outcome(self, version: str, won: bool) -> None:
        """Record a trade outcome for the prompt version used."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext(
                "UPDATE prompt_versions SET trades = trades + 1, wins = wins + :w "
                "WHERE version = :v"
            ), {"w": 1 if won else 0, "v": version})

    def maybe_evolve(self) -> Optional[dict]:
        """Every 30 trades: ask Groq for wording changes, create variant B.

        Returns the analysis dict when a B variant was created.
        """
        with self._lock:
            if db.get_state("ab_prompt_running", "0") == "1":
                return None
            from sqlalchemy import text as sqltext

            from core.db import engine
            with engine.begin() as conn:
                row = conn.execute(sqltext(
                    "SELECT version, trades, wins FROM prompt_versions WHERE version='A'"
                )).first()
            if row is None or row[1] < 30 or row[1] == 0:
                return None
            win_rate = row[2] / row[1] * 100.0
            if self.groq is None:
                return None
            prompt = (
                f"The current analysis prompt produced {row[2]} wins in {row[1]} trades "
                f"({win_rate:.0f}% win rate). Based on these results, how should the analysis "
                "prompt be improved? Return JSON with specific suggested changes to prompt "
                'wording: {"changes": ["..."], "rationale": "..."}'
            )
            analysis = self.groq(prompt, pair="PROMPT", stage="evolve")
            if not analysis or not analysis.get("changes"):
                return None
            variant_b = BASE_PROMPT_HEADER + "\nAdditional instructions:\n" + "\n".join(
                f"- {c}" for c in analysis["changes"][:5])
            with engine.begin() as conn, _WRITE_LOCK:
                conn.execute(sqltext("UPDATE prompt_versions SET active = FALSE"))
                conn.execute(sqltext(
                    "INSERT INTO prompt_versions (created_at, version, prompt, active) "
                    "VALUES (CURRENT_TIMESTAMP, 'B', :p, TRUE)"
                ), {"p": variant_b[:8000]})
            db.set_state("ab_prompt_running", "1")
            db.set_state("ab_prompt_started", str(row[1]))
            db.audit("ai", "prompt_evolution", json.dumps(analysis)[:2000])
            logger.info("prompt variant B created from Groq suggestions")
            return analysis

    def maybe_promote_winner(self) -> Optional[str]:
        """After 50 trades each: promote the significant winner, reset the test."""
        if db.get_state("ab_prompt_running", "0") != "1":
            return None
        from sqlalchemy import text as sqltext

        from core.db import engine
        with engine.begin() as conn:
            rows = conn.execute(sqltext(
                "SELECT version, trades, wins FROM prompt_versions WHERE version IN ('A','B')"
            )).all()
        stats = {r[0]: (r[1], r[2]) for r in rows}
        a, b = stats.get("A", (0, 0)), stats.get("B", (0, 0))
        if a[0] < 50 or b[0] < 50:
            return None
        # two-proportion z-test approximation
        import math
        p1, p2 = a[1] / a[0], b[1] / b[0]
        pooled = (a[1] + b[1]) / (a[0] + b[0])
        se = math.sqrt(pooled * (1 - pooled) * (1 / a[0] + 1 / b[0])) or 1e-9
        z = (p2 - p1) / se
        winner = None
        if z > 1.645:  # p < 0.05 one-sided
            winner = "B"
        elif z < -1.645:
            winner = "A"
        if winner is None:
            return None
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext("DELETE FROM prompt_versions WHERE version = 'B'"))
            conn.execute(sqltext("UPDATE prompt_versions SET active = TRUE WHERE version = :v"),
                         {"v": winner})
        db.set_state("ab_prompt_running", "0")
        db.audit("ai", "prompt_promoted", f"winner={winner} z={z:.2f}")
        logger.info("prompt %s promoted (z=%.2f)", winner, z)
        return winner
