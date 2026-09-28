"""Circuit breakers: hard halts and soft restrictions with auto-resume.

Hard breakers halt trading immediately and persist as unresolved rows in
circuit_breakers. Soft breakers restrict (size cuts, extra confluence) while
the bot continues. Auto-resumable breakers clear themselves when the
underlying condition disappears; critical ones require /resume.

Alert escalation (anti-spam): a trigger alerts ONCE immediately, then only at
1h/3h/6h/12h/24h of continuous activity — at most 6 alerts in the first 24
hours, afterwards at most one per day. resolve() sends a single confirmation.
The schedule survives restarts via the system_state table, so a daemon bounce
cannot re-spam identical alerts.
"""

from __future__ import annotations

import json
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

# Alert schedule (hours of continuous activity before the next alert).
# IMMEDIATE (first trigger) + 1h + 3h + 6h + 12h + 24h = 6 alerts max in 24h.
# After 24h: one alert per day maximum.
ESCALATION_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]

_STATE_KEY = "breaker_alert_state"


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
        # name -> {"first_trigger": epoch, "last_alert": epoch,
        #          "alert_count": int, "resolved": bool, "reason": str}
        self.active_breakers: dict[str, dict] = self._load_state()

    # ---- alert persistence (survives restarts, no re-spam) ----

    def _load_state(self) -> dict[str, dict]:
        raw = ""
        try:
            raw = db.get_state(_STATE_KEY, "")
        except Exception as exc:  # db may not be initialised in unit tests
            logger.debug("breaker state load skipped: %s", exc)
        if not raw:
            return {}
        try:
            return {k: v for k, v in json.loads(raw).items()
                    if isinstance(v, dict)}
        except Exception:
            return {}

    def _save_state(self) -> None:
        try:
            db.set_state(_STATE_KEY, json.dumps(self.active_breakers, default=str))
        except Exception as exc:
            logger.debug("breaker state save skipped: %s", exc)

    # ---- notifications ----

    def _notify(self, text: str) -> None:
        """Best-effort Telegram alert."""
        if self.notifier:
            try:
                self.notifier(text)
            except Exception as exc:  # pragma: no cover
                logger.error("breaker notify failed: %s", exc)

    # ---- triggering with escalation ----

    def trigger(self, breaker: str, reason: str, severity: str = "halt",
                cooldown_sec: int = 300) -> bool:
        """Activate/refresh a breaker; alert per the escalation schedule.

        Returns True only when the breaker newly fired or the schedule emitted
        an escalation alert (never on silent re-evaluations).
        """
        now = time.time()
        new_fire = False
        with self._lock:
            entry = self.active_breakers.get(breaker)

            if entry is None or entry.get("resolved"):
                # First time firing (or re-fired after resolution).
                self.active_breakers[breaker] = {
                    "first_trigger": now,
                    "last_alert": now,
                    "alert_count": 1,
                    "resolved": False,
                    "reason": reason[:200],
                }
                self._save_state()
                new_fire = True
                alerted = True
                message = (f"🚨 CIRCUIT BREAKER [{severity}] {breaker}: {reason}\n"
                           f"Bot halted. Next alert only if unresolved for 1h.")
            else:
                # Already active: escalation schedule, otherwise stay silent.
                entry["reason"] = reason[:200]
                hours_active = (now - entry["first_trigger"]) / 3600.0
                count = entry["alert_count"]
                alerted = False
                message = ""
                if count <= len(ESCALATION_HOURS):
                    threshold = ESCALATION_HOURS[count - 1]
                    if hours_active >= threshold:
                        entry["alert_count"] = count + 1
                        entry["last_alert"] = now
                        self._save_state()
                        alerted = True
                        if count < len(ESCALATION_HOURS):
                            message = (f"⚠️ BREAKER STILL ACTIVE [{severity}] "
                                       f"{breaker}: {reason}\n"
                                       f"Active for {threshold:.0f}h. "
                                       f"Manual check recommended.")
                        else:
                            # 24h reached: 6th alert inside 24h.
                            message = (f"⚠️ BREAKER STILL ACTIVE [{severity}] "
                                       f"{breaker}: {reason}\n"
                                       f"Active for over 24h. Will alert at most "
                                       f"once per day until resolved.")
                else:
                    # Beyond the schedule: at most ONE alert per day.
                    if now - entry["last_alert"] >= 86400.0:
                        entry["alert_count"] = count + 1
                        entry["last_alert"] = now
                        self._save_state()
                        alerted = True
                        message = (f"⚠️ BREAKER STILL ACTIVE [{severity}] "
                                   f"{breaker}: {reason}\n"
                                   f"Active for {hours_active / 24:.0f} days.")

        if new_fire:
            db.log_breaker(breaker, reason, severity)  # halt state lives here
        if alerted:
            self._notify(message)
        return alerted

    def resolve(self, breaker: str) -> None:
        """Clear a specific breaker when its condition is gone."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext(
                "UPDATE circuit_breakers SET resolved = TRUE, resolved_at = :t "
                "WHERE breaker = :b AND resolved = FALSE"
            ), {"t": db._utcnow(), "b": breaker})

        with self._lock:
            entry = self.active_breakers.get(breaker)
            was_alerting = entry is not None and not entry.get("resolved")
            duration_min = 0.0
            if entry is not None:
                duration_min = max(0.0, (time.time() - entry["first_trigger"]) / 60.0)
                entry["resolved"] = True
                self._save_state()

        # Exactly ONE confirmation, only when the breaker had actually alerted.
        if was_alerting:
            self._notify(f"✅ BREAKER RESOLVED: {breaker} recovered after "
                         f"{duration_min:.0f} minutes. Bot resuming trading.")

    def resume_all(self) -> None:
        """Manual /resume: clear every unresolved breaker."""
        db.resolve_breakers()
        for name in list(self.active_breakers):
            self.resolve(name)
        db.audit("risk", "manual_resume", "all breakers cleared via /resume")
        self._notify("▶️ Trading resumed manually. All breakers cleared.")

    # ---- queries (used by AutoRecovery) ----

    def is_active(self, breaker: str) -> bool:
        """True when the breaker has an unresolved, alerting entry."""
        with self._lock:
            entry = self.active_breakers.get(breaker)
            return entry is not None and not entry.get("resolved", False)

    def has_active_breakers(self) -> bool:
        """True when any halt-severity breaker is unresolved in the DB."""
        if any(not v.get("resolved", False) for v in self.active_breakers.values()):
            return True
        try:
            return bool(db.unresolved_breakers())
        except Exception:
            return False

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
