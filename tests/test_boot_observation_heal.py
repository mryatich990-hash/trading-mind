"""Regression test: boot observation flag must heal when the breaker machine
is running and the feed is ok.

Real-world sequence (2026-09-29): laptop slept; on wake the process rebooted,
the startup feed check failed (network not yet up) -> observation_mode=1 and
feed_boot_failure=1; startup reconcile then cleared feed_boot_failure, the
breaker machine booted straight into "running" (no transition), and the old
boot-heal required feed_boot_failure==1 -> permanent signal suppression.
"""

from unittest.mock import MagicMock


class TestBootObservationHeal:
    def _recovery(self):
        from core.autorecovery import AutoRecovery

        sent = []
        brk = MagicMock()
        brk.get_engine_state.return_value = "running"
        brk.has_active_breakers.return_value = False
        data = MagicMock()
        data.get_feed_status.return_value = {
            "active_feed": "yfinance", "consecutive_failures": 0,
            "mt5_connected": False}
        data.feeds = []
        data.mt5_tick_live = MagicMock(return_value=False)
        rec = AutoRecovery(data_engine=data, breakers=brk, notifier=sent.append)
        return rec, sent

    def test_stuck_observation_flag_heals(self, temp_db):
        """observation_mode=1 + machine running + feed ok -> flag cleared."""
        from core import db

        db.set_state("observation_mode", "1")
        db.set_state("feed_boot_failure", "0")  # already reconciled at boot
        rec, sent = self._recovery()
        rec.check_and_recover()
        assert db.get_state("observation_mode", "1") == "0"
        assert any("auto-resumed" in m for m in sent)

    def test_healthy_system_not_renotified(self, temp_db):
        """No divergence: no repeated notifications (idempotent heal)."""
        from core import db

        db.set_state("observation_mode", "0")
        rec, sent = self._recovery()
        rec.check_and_recover()
        rec.check_and_recover()
        assert sent == []

    def test_operator_pause_not_overridden(self, temp_db):
        """A human pause must never be auto-lifted by the heal."""
        from core import db

        db.set_state("observation_mode", "1")
        db.set_state("trading_paused", "1")
        rec, sent = self._recovery()
        rec.check_and_recover()
        assert db.get_state("observation_mode", "1") == "1"
        assert sent == []
