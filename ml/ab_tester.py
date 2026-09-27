"""A/B tester: variant B = ±10% parameter perturbation of the selector.

After 50 trades per variant a two-proportion z-test decides the winner
(p < 0.05); the winner is promoted and a new B variant is generated.
"""

from __future__ import annotations

import json
import math
import random
import threading
from typing import Optional

from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["ABTester", "BASE_PARAMS"]

BASE_PARAMS = {
    "selector_min_score": 70,
    "min_confluence": 8,
    "trail_pips": 15,
    "max_open_trades": 2,
}
TRADES_PER_VARIANT = 50
Z_CRITICAL = 1.645  # p < 0.05 one-sided


class ABTester:
    """Runs A/B experiments over live trading parameters."""

    def __init__(self, base_params: Optional[dict] = None) -> None:
        self.base = dict(base_params or BASE_PARAMS)
        self._lock = threading.Lock()

    # ---- state helpers ----

    @staticmethod
    def running() -> bool:
        """True while an A/B test is active."""
        return db.get_state("ab_running", "0") == "1"

    def start(self) -> Optional[dict]:
        """Create variant B and start the test."""
        with self._lock:
            if self.running():
                return None
            variant_b = {k: v * random.choice((0.9, 1.1)) for k, v in self.base.items()}
            self._record("A", self.base)
            self._record("B", variant_b)
            db.set_state("ab_running", "1")
            db.audit("ml", "ab_started", json.dumps(variant_b))
            logger.info("A/B test started with variant B: %s", variant_b)
            return variant_b

    @staticmethod
    def _record(variant: str, params: dict) -> None:
        """Insert an ab_tests row."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(
                sqltext("INSERT INTO ab_tests (created_at, name, variant, params_json, "
                        "trades, wins, promoted) VALUES (:t, 'selector', :v, :p, 0, 0, FALSE)"),
                {"t": db._utcnow(), "v": variant, "p": json.dumps(params)},
            )

    def record_outcome(self, variant: Optional[str], won: bool) -> None:
        """Increment the variant's trade counters (A when no test running)."""
        variant = variant or "A"
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        try:
            with engine.begin() as conn, _WRITE_LOCK:
                row = conn.execute(sqltext(
                    "SELECT id FROM ab_tests WHERE variant = :v "
                    "ORDER BY id DESC LIMIT 1"), {"v": variant}).first()
                if row is None:
                    return
                conn.execute(sqltext(
                    "UPDATE ab_tests SET trades = trades + 1, wins = wins + :w "
                    "WHERE id = :i"), {"w": 1 if won else 0, "i": row[0]})
        except Exception as exc:
            logger.warning("ab outcome record failed: %s", exc)

    # ---- evaluation ----

    def maybe_evaluate(self) -> Optional[str]:
        """Promote a winner when both variants reach 50 trades."""
        if not self.running():
            return None
        from sqlalchemy import text as sqltext

        from core.db import engine
        with engine.begin() as conn:
            rows = conn.execute(sqltext(
                "SELECT variant, trades, wins FROM ab_tests WHERE name='selector' "
                "ORDER BY id DESC LIMIT 2")).all()
        stats = {r[0]: (r[1], r[2]) for r in rows}
        a, b = stats.get("A", (0, 0)), stats.get("B", (0, 0))
        if a[0] < TRADES_PER_VARIANT or b[0] < TRADES_PER_VARIANT:
            return None
        p1, p2 = a[1] / a[0], b[1] / b[0]
        pooled = (a[1] + b[1]) / (a[0] + b[0])
        se = math.sqrt(pooled * (1 - pooled) * (1 / a[0] + 1 / b[0])) or 1e-9
        z = (p2 - p1) / se
        winner = "B" if z > Z_CRITICAL else "A" if z < -Z_CRITICAL else None
        if winner is None:
            logger.info("A/B inconclusive (z=%.2f); keeping A", z)
            self._reset()
            return None
        params = self._winner_params(winner)
        self._apply_params(params)
        self._mark_promoted(winner)
        self._reset()
        db.audit("ml", "ab_promoted", f"winner={winner} z={z:.2f}")
        logger.info("A/B winner: %s (z=%.2f) params=%s", winner, z, params)
        return winner

    def _winner_params(self, winner: str) -> dict:
        """Stored params of the winning variant."""
        from sqlalchemy import text as sqltext

        from core.db import engine
        with engine.begin() as conn:
            row = conn.execute(sqltext(
                "SELECT params_json FROM ab_tests WHERE variant = :v "
                "ORDER BY id DESC LIMIT 1"), {"v": winner}).first()
        return json.loads(row[0]) if row else self.base

    @staticmethod
    def _apply_params(params: dict) -> None:
        """Promote params into system_state for the selector to read."""
        for key, value in params.items():
            db.set_state(f"param_{key}", str(round(value, 3)))

    @staticmethod
    def _mark_promoted(winner: str) -> None:
        """Flag the winning ab_tests row."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext(
                "UPDATE ab_tests SET promoted = TRUE WHERE variant = :v"),
                {"v": winner})

    @staticmethod
    def _reset() -> None:
        """End the test and clear variant counters."""
        db.set_state("ab_running", "0")
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext("DELETE FROM ab_tests WHERE name='selector' "
                                 "AND promoted = FALSE"))
