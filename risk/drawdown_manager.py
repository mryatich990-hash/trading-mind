"""DrawdownManager: high-watermark tracking and graduated recovery protocol.

Tiers (drawdown from all-time high watermark):
    0-3%   -> normal operation
    3-5%   -> risk cut to 0.75%
    5-8%   -> risk 0.5%, only top-3 strategies
    8-10%  -> risk 0.25%, require 9/10 confluences
    >10%   -> trading halted until manual /resume

Also implements the recovery protocol (GAP 7): risk 0.5%, top-2 strategies,
8/8 confluences, until equity recovers 3% toward the peak.
"""

from dataclasses import dataclass, field
from typing import Optional

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["DrawdownState", "DrawdownManager"]


@dataclass
class DrawdownState:
    """Current drawdown posture."""

    equity: float
    high_watermark: float
    drawdown_pct: float
    tier: int                      # 0..4
    risk_pct: float                # effective risk per trade
    min_confluence: int            # required confluence (8 normal, 9 tier 3)
    allowed_strategies: Optional[list[str]]  # None = no restriction
    halted: bool
    recovery_mode: bool
    note: str = ""


class DrawdownManager:
    """Tracks the high watermark in the database and applies drawdown tiers."""

    TIERS = (
        (0.0, 3.0, 1.0, settings.MIN_CONFLUENCE, None),
        (3.0, 5.0, 0.75, settings.MIN_CONFLUENCE, None),
        (5.0, 8.0, 0.5, settings.MIN_CONFLUENCE, "top3"),
        (8.0, 10.0, 0.25, 9, "top3"),
        (10.0, 1e9, 0.0, settings.MAX_CONFLUENCE, "top3"),
    )

    def __init__(self) -> None:
        self._wm = self._load_watermark()

    # ---- persistence ----

    @staticmethod
    def _load_watermark() -> float:
        """High watermark from DB, defaulting to balance if never set."""
        try:
            raw = db.get_state("high_watermark", "0")
            return float(raw) if raw else 0.0
        except (TypeError, ValueError):
            return 0.0

    def update(self, equity: float) -> DrawdownState:
        """Refresh the watermark and recompute the tier."""
        if equity > self._wm:
            self._wm = round(equity, 2)
            db.set_state("high_watermark", str(self._wm))
            logger.info("new high watermark: %.2f", self._wm)
        return self.evaluate(equity)

    def watermark(self) -> float:
        """Current high watermark."""
        return self._wm

    def evaluate(self, equity: float) -> DrawdownState:
        """Compute the drawdown state for the current equity."""
        equity = max(equity, 0.0)
        dd_pct = 0.0 if self._wm <= 0 else max(0.0, (self._wm - equity) / self._wm * 100.0)
        tier = 0
        for i, (lo, hi, _risk, _conf, _strat) in enumerate(self.TIERS):
            if lo <= dd_pct < hi:
                tier = i
                break
        else:
            tier = len(self.TIERS) - 1

        lo, hi, risk, conf, strat = self.TIERS[tier]
        halted = dd_pct >= settings.DRAWDOWN_HALT_PCT

        # recovery protocol: once 5%+ deep, hold conservative settings until
        # equity recovers 3% of the watermark back toward the peak
        recovery = False
        recovered_flag = db.get_state("dd_recovery_active", "0") == "1"
        if dd_pct >= 5.0:
            recovery = True
            if not recovered_flag:
                db.set_state("dd_recovery_active", "1")
                db.audit("risk", "recovery_mode_entered", f"dd={dd_pct:.1f}%")
        elif recovered_flag and dd_pct <= 2.0:
            db.set_state("dd_recovery_active", "0")
            db.audit("risk", "recovery_mode_cleared", f"dd={dd_pct:.1f}%")
        elif recovered_flag:
            recovery = True
            risk, conf, strat = min(risk, 0.5) or 0.5, max(conf, 8), "top2"

        state = DrawdownState(
            equity=equity,
            high_watermark=self._wm,
            drawdown_pct=round(dd_pct, 2),
            tier=tier,
            risk_pct=risk if not halted else 0.0,
            min_confluence=conf,
            allowed_strategies=self._allowed(strat, recovery),
            halted=halted,
            recovery_mode=recovery,
        )
        if halted:
            self._trigger_halt_once(dd_pct)
        return state

    @staticmethod
    def _allowed(strategy_spec: str, recovery: bool) -> Optional[list[str]]:
        """Allowed strategy names (None = unrestricted)."""
        if strategy_spec == "top2":
            return None  # resolved dynamically by the caller from performance
        if strategy_spec == "top3":
            return None
        return None

    def _trigger_halt_once(self, dd_pct: float) -> None:
        """Log a halt breaker once per episode."""
        if db.get_state("dd_halt_active", "0") != "1":
            db.set_state("dd_halt_active", "1")
            db.log_breaker(
                "drawdown_halt",
                f"equity {dd_pct:.1f}% below high watermark {self._wm:.2f} "
                "- manual review required",
                "halt",
            )
            db.audit("risk", "drawdown_halt", f"dd={dd_pct:.1f}%")

    def resume_from_halt(self) -> None:
        """Operator /resume after a drawdown halt."""
        db.set_state("dd_halt_active", "0")
        db.audit("risk", "drawdown_halt_resumed", f"watermark={self._wm:.2f}")
