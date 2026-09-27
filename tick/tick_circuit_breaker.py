"""Tick-based circuit breaker (UPGRADE 3): news-spike halt.

Halts new entries when tick velocity exceeds 5x the pair's rolling normal;
resumes only after velocity stays normal for 3 consecutive minutes.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

VELOCITY_MULT = 5.0
CALMMinutes_REQUIRED = 3


class TickCircuitBreaker:
    """Velocity spike guard per pair."""

    def __init__(self, baseline_min: float = 30.0) -> None:
        self.baseline_velocity: dict[str, float] = {}
        self.baseline_min = baseline_min
        self.halted_since: dict[str, float] = {}
        self.calm_since: dict[str, Optional[float]] = {}

    def update_baseline(self, pair: str, velocity: float) -> None:
        """Track rolling normal velocity (EMA)."""
        pair = pair.upper()
        prev = self.baseline_velocity.get(pair)
        self.baseline_velocity[pair] = (0.9 * prev + 0.1 * velocity) if prev else velocity

    def evaluate(self, pair: str, velocity: float) -> dict:
        """Return {halted, reason, resume_in_sec} for the pair."""
        pair = pair.upper()
        base = self.baseline_velocity.get(pair)
        if base is None or base <= 0:
            self.update_baseline(pair, velocity)
            return {"halted": False, "reason": "", "resume_in_sec": 0}
        now = time.time()
        if velocity > base * VELOCITY_MULT:
            if pair not in self.halted_since:
                logger.warning("tick breaker: %s velocity %.0f > %.0f x%.1f - HALT",
                               pair, velocity, base, VELOCITY_MULT)
                self.halted_since[pair] = now
                self.calm_since[pair] = None
            return {"halted": True, "reason": f"tick velocity {velocity:.0f}/min "
                                             f"({velocity / max(base, 1):.1f}x normal)",
                    "resume_in_sec": 0}
        # velocity normal -> require 3 calm minutes to resume
        if pair in self.halted_since:
            calm_start = self.calm_since.get(pair)
            if calm_start is None:
                self.calm_since[pair] = now
                calm_start = now
            calm_for = now - calm_start
            if calm_for >= CALMMinutes_REQUIRED * 60:
                del self.halted_since[pair]
                del self.calm_since[pair]
                logger.info("tick breaker: %s resumed after calm period", pair)
                return {"halted": False, "reason": "resumed", "resume_in_sec": 0}
            return {"halted": True,
                    "reason": f"cooling down ({calm_for:.0f}s/{CALMMinutes_REQUIRED * 60}s calm)",
                    "resume_in_sec": int(CALMMinutes_REQUIRED * 60 - calm_for)}
        return {"halted": False, "reason": "", "resume_in_sec": 0}

    def is_halted(self, pair: str) -> bool:
        """Quick check for the entry path."""
        return pair.upper() in self.halted_since

    def status(self) -> dict:
        """Dashboard payload."""
        return {"halted_pairs": sorted(self.halted_since),
                "baselines": {k: round(v, 1) for k, v in self.baseline_velocity.items()}}
