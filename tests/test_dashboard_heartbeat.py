"""Dashboard must not show OFFLINE while the engine heartbeat is fresh.

On 2026-10-01 the `running` flag went stale after the overnight feed
observation boot and the dashboard showed OFFLINE for hours while the
engine was alive and trading (forced trade #3 filled at 10:20 UTC).
api_status now treats a <5-minute-old engine_heartbeat as proof of life,
same contract as /health.
"""

from datetime import datetime, timedelta, timezone

from core import db


class TestDashboardHeartbeat:
    def _client(self):
        from dashboard.app import app

        app.config["TESTING"] = True
        return app.test_client()

    def test_stale_running_flag_with_fresh_heartbeat_shows_running(self, temp_db):
        # the incident state: flag stale/absent, but the supervisor is beating
        db.set_state("running", "0")
        fresh = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        db.set_state("engine_heartbeat", fresh)

        data = self._client().get("/api/status").get_json()
        assert data["status"] == "RUNNING"

    def test_no_heartbeat_and_no_flag_shows_offline(self, temp_db):
        db.set_state("running", "0")
        db.set_state("engine_heartbeat", "")

        data = self._client().get("/api/status").get_json()
        assert data["status"] == "OFFLINE"

    def test_stale_heartbeat_shows_offline(self, temp_db):
        db.set_state("running", "0")
        old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        db.set_state("engine_heartbeat", old)

        data = self._client().get("/api/status").get_json()
        assert data["status"] == "OFFLINE"

    def test_flag_lying_off_with_halt_still_halts(self, temp_db):
        """Breakers outrank liveness exactly as before."""
        db.set_state("running", "1")
        db.log_breaker("daily_loss", "test", "halt")

        data = self._client().get("/api/status").get_json()
        assert data["status"] == "HALTED"
        db.resolve_breakers()
