"""AutoRecovery: 60-second self-healing loop for the trading engine.

Responsibilities (CRITICAL FIX #3):
  1. Feed watch    - when the data feed works again while ``data_stale`` is
     active, resolve the breaker, log the recovery and announce the resume.
  2. MT5 watchdog  - attempt terminal reconnection when MT5 drops
     (throttled to one attempt per 5 minutes).
  3. Auto-resume   - outside the weekend window, when the engine is halted
     but NO circuit breaker is active, lift the halt automatically.

Safety rules:
  - An operator pause (``trading_paused=1`` via /pause or the dashboard) is
    NEVER overridden by auto-recovery.
  - Observation mode (failed startup checks) is only lifted manually (/resume)
    unless the engine was halted purely by breakers that have since cleared.
  - Weekend protection is left to the weekend breaker (auto-lifts Sun 21:00).
  - Every automatic action is written to the audit log.
"""

from __future__ import annotations

import time
from typing import Optional

from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["AutoRecovery"]


class AutoRecovery:
    """Periodic engine self-healing (data feed, MT5, halt state)."""

    RECONNECT_THROTTLE_SEC = 300  # MT5 re-init at most every 5 minutes

    def __init__(self, data_engine, breakers, notifier: Optional[callable] = None,
                 interval_sec: int = 60) -> None:
        self.data = data_engine
        self.breakers = breakers
        self.notifier = notifier
        self.recovery_interval = interval_sec
        self._last_reconnect_attempt = 0.0
        self._last_feed_ok: Optional[bool] = None

    # ---- public loop ----

    async def monitor(self) -> None:
        """Background task: run check_and_recover every 60 seconds."""
        import asyncio

        # give startup (checks + first cycles) a moment before healing kicks in
        await asyncio.sleep(self.recovery_interval)
        while True:
            try:
                await asyncio.to_thread(self.check_and_recover)
            except Exception as exc:
                logger.error("recovery check failed: %s", exc)
            await asyncio.sleep(self.recovery_interval)

    # ---- recovery checks ----

    def check_and_recover(self) -> None:
        """One recovery pass: feed watch, MT5 reconnect, auto-resume."""
        self._check_feed_recovery()
        self._check_mt5_connection()
        self._check_auto_resume()

    def _check_feed_recovery(self) -> None:
        """Resolve data_stale when the feed is producing fresh candles again."""
        try:
            status = self.data.get_feed_status()
        except Exception as exc:
            logger.error("feed status unavailable: %s", exc)
            return

        feed_ok = status.get("active_feed") not in (None, "none") \
            and status.get("consecutive_failures", 0) == 0

        # edge-triggered logging: one audit row per feed outage/recovery
        if feed_ok != self._last_feed_ok:
            if feed_ok:
                db.audit("data", "feed_recovered",
                         f"active feed {status.get('active_feed')} "
                         f"(failures cleared)")
                logger.info("recovery: data feed healthy via %s",
                            status.get("active_feed"))
            else:
                db.audit("data", "feed_outage",
                         f"consecutive failures "
                         f"{status.get('consecutive_failures')}")
                logger.warning("recovery: data feed failing")
            self._last_feed_ok = feed_ok

        if not feed_ok:
            return
        try:
            stale_active = self.breakers.is_active("data_stale") \
                or "data_stale" in db.unresolved_breakers()
            if stale_active:
                self.breakers.resolve("data_stale")
                db.audit("data", "auto_recovery",
                         "data_stale resolved: feed recovered")
                logger.info("recovery: data_stale cleared, feed is back")
        except Exception as exc:
            logger.error("data_stale recovery failed: %s", exc)

        # Boot-heal: the process started with every feed dead (observation
        # mode); a feed now works, so re-enable trading.
        try:
            if db.get_state("feed_boot_failure", "0") == "1":
                db.set_state("feed_boot_failure", "0")
                db.set_state("observation_mode", "0")
                db.set_state("running", "1")
                db.audit("system", "auto_recovery",
                         "feed recovered after failed boot: trading re-enabled")
                self._notify("✅ Data feed recovered after startup failure — "
                             "bot auto-resumed.")
                logger.info("recovery: feed-boot failure healed, trading re-enabled")
        except Exception as exc:
            logger.error("feed-boot recovery failed: %s", exc)

    def _check_mt5_connection(self) -> None:
        """Reconnect the MT5 terminal when it drops (throttled)."""
        try:
            mt5_feed = next((f for f in self.data.feeds if f.name == "mt5"), None)
            if mt5_feed is None or not mt5_feed.configured:
                return  # package not installed on this host: nothing to do
            if mt5_feed.connected():
                return
            if time.monotonic() - self._last_reconnect_attempt \
                    < self.RECONNECT_THROTTLE_SEC:
                return
            self._last_reconnect_attempt = time.monotonic()
            if mt5_feed.reconnect():
                db.audit("data", "mt5_reconnected", "terminal re-initialized")
                self._notify("🔁 MT5 reconnected: terminal back online.")
                logger.info("recovery: MT5 reconnected")
            else:
                logger.warning("recovery: MT5 reconnect attempt failed")
        except Exception as exc:
            logger.error("MT5 reconnect check failed: %s", exc)

    def _check_auto_resume(self) -> None:
        """Lift a halt when no breaker is active and the market is open."""
        from data.market_data_engine import fx_market_closed

        if fx_market_closed():
            return  # weekend: the weekend breaker manages the halt window

        # Operator pause wins: never auto-resume over a human decision.
        if db.get_state("trading_paused", "0") == "1":
            return

        try:
            if self.breakers.has_active_breakers():
                return  # still halted for a reason
        except Exception as exc:
            logger.error("breaker check failed during auto-resume: %s", exc)
            return

        if db.get_state("running", "1") != "0":
            return  # engine already running

        db.set_state("running", "1")
        db.audit("risk", "auto_resume",
                 "halt lifted: no active breakers, market open")
        self._notify("▶️ Bot auto-resumed: no active circuit breakers, "
                     "market open.")
        logger.info("recovery: bot auto-resumed (no active breakers)")

    def _notify(self, text: str) -> None:
        """Best-effort Telegram alert."""
        if self.notifier:
            try:
                self.notifier(text)
            except Exception as exc:  # pragma: no cover
                logger.error("recovery notify failed: %s", exc)
