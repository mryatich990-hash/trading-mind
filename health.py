"""Render web service: tiny health/status endpoint for UptimeRobot.

Runs on Render's FREE web tier (worker tier is paid) while the actual bot
engine runs as a background process inside this same service via
deploy/run_engine.py. Two separate Render services would double the cost
and idle-spin the health app, so this "health" service doubles as the
engine host.

Endpoints:
    GET /health  -> engine liveness + DB probe (for UptimeRobot, 5-min pings)
    GET /status  -> engine snapshot (trades today, PnL, breakers)
"""

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

_engine_proc: subprocess.Popen | None = None


def _ensure_engine() -> None:
    """Spawn deploy/run_engine.py (supervisor) once, in a daemon thread."""
    global _engine_proc
    if _engine_proc is not None or os.environ.get("RUN_ENGINE", "1") != "1":
        return
    _engine_proc = subprocess.Popen(
        [sys.executable, str(ROOT / "deploy" / "run_engine.py")], cwd=str(ROOT)
    )


def _engine_alive() -> bool:
    """Child heartbeat: engine writes engine_heartbeat to system_state."""
    try:
        from core.db import get_state

        ts = get_state("engine_heartbeat", "")
        if not ts:
            return False
        beat = datetime.fromisoformat(ts)
        return (datetime.now(timezone.utc) - beat).total_seconds() < 300
    except Exception:
        return False


def _supervisor_alive() -> str:
    """File-based supervisor heartbeat (/tmp/engine.beat, written every 30s)."""
    try:
        from pathlib import Path

        p = Path("/tmp/engine.beat")
        if not p.exists():
            return "no-file"
        age = time.time() - float(p.read_text().strip())
        return f"fresh({int(age)}s)" if age < 90 else f"stale({int(age)}s)"
    except Exception as exc:
        return f"err:{exc}"[:60]


@app.route("/health")
def health():
    """Liveness probe: 200 while the process runs; detailed status in body.

    Always-200 keeps the Render health check and UptimeRobot pings working
    even when the database is unreachable — the JSON body carries the exact
    DB error so operators can diagnose remotely (e.g. pooler auth issues).
    """
    db_status, db_detail = "unknown", ""
    try:
        from core.db import get_state

        get_state("engine_heartbeat", "")  # cheap DB roundtrip
        db_status = "connected"
    except Exception as exc:
        db_status = "down"
        db_detail = f"{type(exc).__name__}: {exc}"[:300]

    return jsonify({
        "status": "running",
        "database": db_status,
        "database_error": db_detail,
        "engine": "alive" if _engine_alive() else "starting",
        "supervisor": _supervisor_alive(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/status")
def status():
    """Snapshot for humans / uptime dashboards."""
    payload: dict = {"timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        from core import db

        payload.update({
            "open_trades": db.count_open_trades(),
            "trades_today": len(db.trades_today()),
            "daily_pnl": db.daily_pnl(),
            "halted_by": db.unresolved_breakers(),
            "paused": db.get_state("trading_paused", "0") == "1",
            "engine": "alive" if _engine_alive() else "starting",
        })
    except Exception as exc:
        payload["error"] = str(exc)
    return jsonify(payload), 200


if __name__ == "__main__":
    _ensure_engine()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
