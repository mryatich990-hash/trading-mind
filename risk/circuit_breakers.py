"""Circuit breakers: hard halts and soft restrictions with auto-resume.

Hard breakers halt trading immediately and persist as unresolved rows in
circuit_breakers. Soft breakers restrict (size cuts, extra confluence) while
the bot continues. Auto-resumable breakers clear themselves when the
underlying condition disappears; critical ones require /resume.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["BreakerState", "CircuitBreakers", "ANTI_MARTINGALE"]

ANTI_MARTINGALE = {"win_step": 0.10, "win_max_streak": 3, "loss_step": 0.15,
                   "reset_after": 5}

AUTO_RESUME = {"data_stale", "mt5_disconnected", "groq_down", "atr_explosion",
               "vix_halt", "weekend", "weekend_pre"}
CRITICAL = {"daily_loss", "drawdown_halt", "margin_halt", "groq_rejections"}


@dataclass
class BreakerState:
    """Current breaker posture."""

    halted: bool
    halt_reasons: list[str] = field(default_factory=list)
    size_multiplier: float = 1.0
    extra_confluence: int = 0
    soft_notes: list[str] = field(default_factory=list)


class CircuitBreakers:
    """Evaluates all breaker conditions and maintains DB-persisted state."""

    def __init__(self, notifier: Optional[callable] = None) -> None:
        self.notifier = notifier
        self._lock = threading.Lock()
        self._last_trigger: dict[str, float] = {}

    def _notify(self, text: str) -> None:
        """Best-effort Telegram alert."""
        if self.notifier:
            try:
                self.notifier(text)
            except Exception as exc:  # pragma: no cover
                logger.error("breaker notify failed: %s", exc)

    # ---- triggering ----

    def trigger(self, breaker: str, reason: str, severity: str = "halt",
                cooldown_sec: int = 300) -> bool:
        """Activate a breaker (deduplicated by cooldown). True when new."""
        with self._lock:
            last = self._last_trigger.get(breaker, 0.0)
            if time.monotonic() - last < cooldown_sec:
                return False
            self._last_trigger[breaker] = time.monotonic()
        db.log_breaker(breaker, reason, severity)
        self._notify(f"🚨 BREAKER [{severity}] {breaker}: {reason}")
        return True

    def resolve(self, breaker: str) -> None:
        """Clear a specific breaker when its condition is gone."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext(
                "UPDATE circuit_breakers SET resolved = TRUE, resolved_at = :t "
                "WHERE breaker = :b AND resolved = FALSE"
            ), {"t": db._utcnow(), "b": breaker})
        self._notify(f"✅ breaker cleared: {breaker}")

    def resume_all(self) -> None:
        """Manual /resume: clear every unresolved breaker."""
        db.resolve_breakers()
        db.audit("risk", "manual_resume", "all breakers cleared via /resume")
        self._notify("▶️ Trading resumed manually. All breakers cleared.")

    # ---- periodic evaluation ----

    def evaluate(self, *, feed_age_sec: float = 0.0, mt5_ok: bool = True,
                 db_ok: bool = True, groq_ok: bool = True,
                 groq_rejections: int = 0, atr_ratio: float = 1.0,
                 vix: float = 0.0, margin_level: float = 1000.0,
                 now_utc_hour: Optional[int] = None,
                 now_utc_weekday: Optional[int] = None,
                 conviction_avg_5: float = 80.0) -> BreakerState:
        """Check every condition; return the aggregated state."""
        state = BreakerState(halted=False)

        def halted(name: str) -> bool:
            return name in db.unresolved_breakers()

        # weekend breaker first (Friday 20:00 UTC -> Sunday 21:00 UTC):
        # stale candles are EXPECTED during the close, so data_stale must not
        # fire while the market is shut.
        hour = now_utc_hour if now_utc_hour is not None else time.gmtime().tm_hour
        weekday = now_utc_weekday if now_utc_weekday is not None else time.gmtime().tm_wday
        weekend = (weekday == 4 and hour >= 20) or weekday in (5, 6) \
            or (weekday == 6 and hour < 21)

        # hard conditions -> trigger
        if feed_age_sec > settings.STALENESS_LIMIT_SEC and not weekend:
            self.trigger("data_stale", f"feed age {feed_age_sec:.0f}s")
        if not mt5_ok:
            self.trigger("mt5_disconnected", "MT5 unreachable")
        if not db_ok:
            self.trigger("db_down", "database unreachable")
        if not groq_ok:
            self.trigger("groq_down", "Groq API failing after retries")
        if groq_rejections >= settings.GROQ_MAX_REJECTIONS:
            self.trigger("groq_rejections", f"{groq_rejections} consecutive rejections")
        if atr_ratio >= settings.ATR_EXPLOSION_MULT:
            self.trigger("atr_explosion", f"ATR {atr_ratio:.1f}x average")
        if vix >= settings.VIX_HALT:
            self.trigger("vix_halt", f"VIX {vix:.0f}")
        if margin_level < settings.MARGIN_HALT_PCT:
            self.trigger("margin_halt", f"margin {margin_level:.0f}%")

        if weekend:
            self.trigger("weekend", "weekend protection window", "halt", cooldown_sec=3600)

        # aggregate halts
        for name in ("data_stale", "mt5_disconnected", "db_down", "groq_down",
                     "groq_rejections", "atr_explosion", "vix_halt", "margin_halt",
                     "daily_loss", "drawdown_halt", "weekend"):
            if halted(name):
                state.halted = True
                state.halt_reasons.append(name)

        # auto-resume when the condition has cleared
        for name in AUTO_RESUME:
            if halted(name):
                cleared = {
                    "data_stale": (feed_age_sec <= settings.STALENESS_LIMIT_SEC
                                   or weekend),
                    "mt5_disconnected": mt5_ok,
                    "db_down": db_ok,
                    "groq_down": groq_ok,
                    "atr_explosion": atr_ratio < settings.ATR_EXPLOSION_MULT,
                    "vix_halt": vix < settings.VIX_HALT,
                    "weekend": not weekend,
                }.get(name, False)
                if cleared:
                    self.resolve(name)

        # soft restrictions
        if settings.VIX_REDUCE_60 <= vix < settings.VIX_HALT:
            state.size_multiplier *= 0.4
            state.soft_notes.append("VIX 30-40: size x0.4")
        elif settings.VIX_REDUCE_30 <= vix < settings.VIX_REDUCE_60:
            state.size_multiplier *= 0.7
            state.soft_notes.append("VIX 20-30: size x0.7")
        if 1.5 <= atr_ratio < settings.ATR_EXPLOSION_MULT:
            state.size_multiplier *= 0.7
            state.soft_notes.append("ATR elevated: size x0.7")
        if margin_level < settings.MARGIN_ALERT_PCT:
            state.soft_notes.append(f"margin {margin_level:.0f}% below 500 alert")
        if conviction_avg_5 < 65:
            state.extra_confluence = 1
            state.soft_notes.append("low Groq conviction: +1 confluence required")
        if halted("weekend_pre"):
            state.soft_notes.append("Friday evening: no new trades")
        return state
