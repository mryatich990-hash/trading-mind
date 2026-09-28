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

AUTO_RESUME = {"mt5_disconnected", "groq_down", "atr_explosion",
               "vix_halt", "weekend", "weekend_pre", "feed_dead"}
CRITICAL = {"daily_loss", "drawdown_halt", "margin_halt", "groq_rejections"}

# Alert schedule (hours of continuous activity before the next alert).
# IMMEDIATE (first trigger) + 1h + 3h + 6h + 12h + 24h = 6 alerts max in 24h.
# After 24h: one alert per day maximum.
ESCALATION_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]

_STATE_KEY = "breaker_alert_state"

# ---- engine state machine (RUNNING / OBSERVATION / HALTED) ----
# data_stale alone NEVER halts: it drops the engine to observation (no new
# trades, positions still managed). Full halt requires the feed dead
# FEED_DEAD_LIMIT_SEC (30 min) continuously. Recovery requires the feed
# stable for RECOVERY_STABLE_SEC (5 min) per transition - no flickering.
RECOVERY_STABLE_SEC = 300
TRANSITION_ALERTS = {
    ("running", "observation"): "👁 FEED UNSTABLE — observation mode: no new "
                                 "trades, existing positions still managed. "
                                 "Auto-resumes after 5 min of stable feed.",
    ("observation", "halted"): "🛑 FEED DEAD 30+ MIN — trading HALTED fully.",
    ("running", "halted"): "🛑 FEED DEAD 30+ MIN — trading HALTED fully.",
    ("halted", "observation"): "⚠️ Feed recovering — HALT lifted to observation. "
                                "5 more minutes of stable feed to resume.",
    ("observation", "running"): "✅ Feed stable 5+ min — observation cleared, "
                                 "trading RESUMED.",
    ("halted", "running"): "✅ Feed recovered — trading RESUMED.",
}


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
                cooldown_sec: int = 300, notify: bool = True) -> bool:
        """Activate/refresh a breaker; alert per the escalation schedule.

        notify=False records the breaker silently (the engine state machine
        announces transitions itself, so data_stale/feed_dead don't double-
        alert). Returns True only on a new fire or scheduled escalation.
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
                # Observation-severity breakers NEVER escalate: the engine
                # state machine announces state changes exactly once.
                if severity == "observe":
                    entry["reason"] = reason[:200]
                    self._save_state()
                    return False
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
        if alerted and notify:
            self._notify(message)
        return alerted

    def resolve(self, breaker: str, notify: bool = True) -> None:
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
        if was_alerting and notify:
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
        """True when ANY breaker (halt or observation) is unresolved in the DB."""
        if any(not v.get("resolved", False) for v in self.active_breakers.values()):
            return True
        try:
            return bool(db.unresolved_breakers("all"))
        except Exception:
            return False

    # ---- engine state machine (RUNNING / OBSERVATION / HALTED) ----

    def _persist_state(self, s: str) -> None:
        try:
            db.set_state("engine_state", s)
            db.set_state("engine_state_since", db._utcnow().isoformat())
            # keep the legacy observation key aligned so every reader
            # (run-loop gate, /resume, dashboards) sees the same posture
            db.set_state("observation_mode", "1" if s == "observation" else "0")
        except Exception:
            pass

    def state_since(self) -> float:
        """Seconds spent in the current engine state (0 if unknown)."""
        try:
            raw = db.get_state("engine_state_since", "")
            if raw:
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())
        except Exception:
            pass
        return 0.0

    def get_engine_state(self) -> str:
        """Derive the current state purely from unresolved breakers.

        feed_dead (halt severity)       -> HALTED
        data_stale (observe severity)   -> OBSERVATION
        anything else                   -> RUNNING
        Deriving (not storing) means the state survives restarts and can
        never desynchronise from the breaker rows.
        """
        try:
            if "feed_dead" in db.unresolved_breakers("halt"):
                return "halted"
            if "data_stale" in db.unresolved_breakers("observe"):
                return "observation"
        except Exception:
            pass
        return "running"

    def _transition(self, new_state: str, from_state: Optional[str] = None) -> None:
        """Move the engine to a new state; exactly ONE alert per transition.

        from_state: pass the state captured BEFORE the breaker rows changed
        (trigger() writes the DB immediately, which would otherwise make
        get_engine_state() already report the new state and swallow the
        transition alert).
        """
        old = from_state or self.get_engine_state()
        if old == new_state:
            return
        self._persist_state(new_state)
        db.audit("risk", f"engine_{old}_to_{new_state}", "feed state machine")
        message = TRANSITION_ALERTS.get((old, new_state))
        if message:
            self._notify(message)
        logger.info("engine state: %s -> %s", old, new_state)

    def _retire_legacy_data_stale_halt(self) -> None:
        """Resolve pre-state-machine HALT-severity data_stale rows once.

        Older versions logged data_stale as a halt; the state machine owns
        it as an observation breaker now, so stale halt rows must not keep
        blocking the risk gate.
        """
        if getattr(self, "_legacy_migrated", False):
            return
        try:
            from sqlalchemy import text as _sql
            from core.db import engine as _eng, _WRITE_LOCK as _wl
            with _eng.begin() as conn, _wl:
                conn.execute(_sql(
                    "UPDATE circuit_breakers SET resolved = TRUE, resolved_at = :t "
                    "WHERE breaker = 'data_stale' AND severity = 'halt' "
                    "AND resolved = FALSE"), {"t": db._utcnow()})
            self._legacy_migrated = True
        except Exception:
            pass

    def _observe(self, reason: str) -> None:
        """Enter/refresh observation mode (soft data_stale breaker)."""
        prev = self.get_engine_state()          # capture BEFORE the DB write
        self._retire_legacy_data_stale_halt()
        self.trigger("data_stale", reason, severity="observe", notify=False)
        self._transition("observation", from_state=prev)

    def _full_halt(self, reason: str) -> None:
        """Full stop: feed dead 30+ min (or escalated by other halt rules)."""
        prev = self.get_engine_state()          # capture BEFORE the DB write
        self.trigger("feed_dead", reason, severity="halt", notify=False)
        # the soft breaker is superseded by the halt
        self.resolve("data_stale", notify=False)
        self._transition("halted", from_state=prev)

    def _recover_observation(self) -> None:
        """Feed stable again: clear observation, back to RUNNING."""
        prev = self.get_engine_state()
        self.resolve("data_stale", notify=False)
        self._transition("running", from_state=prev)

    def _recover_halt(self, feed_ok: bool, stable_for: float) -> None:
        """Step the engine down from HALTED once the feed has stabilised."""
        if not feed_ok:
            return
        if stable_for >= RECOVERY_STABLE_SEC:
            prev = self.get_engine_state()
            self.resolve("feed_dead", notify=False)
            self._transition("observation", from_state=prev)  # one more stable window to RUNNING
        # else: stay halted until the feed proves itself for 5 minutes

    # ---- feed rules (shared by evaluate() and the 30s AutoRecovery loop) ----

    def evaluate_feed_rules(self, *, feed_age_sec: float = 0.0,
                            mt5_tick_live: bool = False,
                            feed_stability: float = 100.0,
                            seconds_since_success: float = 0.0,
                            feed_stable_for_sec: float = 0.0,
                            weekend: Optional[bool] = None) -> None:
        """Run the RUNNING/OBSERVATION/HALTED state machine for the feed.

        Idempotent: safe to call every 30 seconds; transitions (and their
        single alerts) only happen on actual state changes.
        """
        if weekend is None:
            gm = time.gmtime()
            weekend = ((gm.tm_wday == 4 and gm.tm_hour >= 20)
                       or gm.tm_wday in (5, 6)
                       or (gm.tm_wday == 6 and gm.tm_hour < 21))
        if weekend:
            return  # stale candles are EXPECTED during the close

        self._retire_legacy_data_stale_halt()

        # Rule 6: a live MT5 tick stream means the feed CANNOT be stale.
        if mt5_tick_live:
            feed_age_sec = 0.0
            feed_stability = max(feed_stability, 100.0)

        # Rule 3: a single gap is noise; a PATTERN of gaps (success rate
        # under 50% over the last 20 checks) is what fires the breaker.
        # Pure candle-age lag with successful fetches never triggers it.
        unstable_pattern = feed_stability < 50.0

        # Rule 4 clock: UNINTERRUPTED health (resets on every failed fetch),
        # used for the 5-minute-stable recovery ladder.
        stable_for = max(0.0, feed_stable_for_sec)

        # Rule 2: full halt only after 30 min with zero successful fetches.
        dead = seconds_since_success >= settings.FEED_DEAD_LIMIT_SEC

        if dead:
            self._full_halt(f"no successful fetch for "
                            f"{seconds_since_success / 60:.0f} min")
        elif unstable_pattern:
            self._observe(f"feed stability {feed_stability:.0f}% "
                          f"(age {feed_age_sec:.0f}s)")
        else:
            # feed healthy right now: resume only after 5 minutes of
            # UNBROKEN stability (healthy_since resets on every failed
            # fetch, so halt -> observation -> running always spans two
            # separate stable windows — no flicker)
            cur = self.get_engine_state()
            if cur == "halted":
                self._recover_halt(feed_ok=True, stable_for=stable_for)
            elif cur == "observation" and stable_for >= RECOVERY_STABLE_SEC:
                self._recover_observation()

    # ---- periodic evaluation ----

    def evaluate(self, *, feed_age_sec: float = 0.0, mt5_ok: bool = True,
                 mt5_tick_live: bool = False, feed_stability: float = 100.0,
                 seconds_since_success: float = 0.0,
                 feed_stable_for_sec: float = 0.0,
                 db_ok: bool = True, groq_ok: bool = True,
                 groq_rejections: int = 0, atr_ratio: float = 1.0,
                 vix: float = 0.0, margin_level: float = 1000.0,
                 now_utc_hour: Optional[int] = None,
                 now_utc_weekday: Optional[int] = None,
                 conviction_avg_5: float = 80.0) -> BreakerState:
        """Check every condition; run the state machine; return the posture."""
        state = BreakerState(halted=False)

        def halted(name: str) -> bool:
            return name in db.unresolved_breakers()

        # weekend breaker first (Friday 20:00 UTC -> Sunday 21:00 UTC):
        # stale candles are EXPECTED during the close, so feed rules must not
        # fire while the market is shut.
        hour = now_utc_hour if now_utc_hour is not None else time.gmtime().tm_hour
        weekday = now_utc_weekday if now_utc_weekday is not None else time.gmtime().tm_wday
        weekend = (weekday == 4 and hour >= 20) or weekday in (5, 6) \
            or (weekday == 6 and hour < 21)

        # ---- feed rules (state machine owns data_stale / feed_dead) ----
        self.evaluate_feed_rules(
            feed_age_sec=feed_age_sec, mt5_tick_live=mt5_tick_live,
            feed_stability=feed_stability,
            seconds_since_success=seconds_since_success,
            feed_stable_for_sec=feed_stable_for_sec, weekend=weekend)

        # ---- non-feed breakers (unchanged behaviour) ----
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

        # auto-resume when the condition has cleared (non-feed breakers only;
        # data_stale/feed_dead are owned by the state machine above)
        for name in AUTO_RESUME:
            if name in ("data_stale", "feed_dead"):
                continue
            if halted(name):
                cleared = {
                    "mt5_disconnected": mt5_ok,
                    "db_down": db_ok,
                    "groq_down": groq_ok,
                    "atr_explosion": atr_ratio < settings.ATR_EXPLOSION_MULT,
                    "vix_halt": vix < settings.VIX_HALT,
                    "weekend": not weekend,
                }.get(name, False)
                if cleared:
                    self.resolve(name)

        # aggregate halts (data_stale deliberately EXCLUDED: it observes,
        # it does not halt — that is the fix for the halt/resume loop)
        for name in ("feed_dead", "mt5_disconnected", "db_down", "groq_down",
                     "groq_rejections", "atr_explosion", "vix_halt", "margin_halt",
                     "daily_loss", "drawdown_halt", "weekend"):
            if halted(name):
                state.halted = True
                state.halt_reasons.append(name)

        # observation posture: no new trades, keep managing positions
        if self.get_engine_state() == "observation" and not state.halted:
            state.halt_reasons.append("data_stale(observe)")

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
