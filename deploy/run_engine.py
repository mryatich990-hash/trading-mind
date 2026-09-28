"""Production engine runner: heartbeat + loop watchdog.

Wrapped by health.py on Render so the FREE web tier can host the engine
(worker tier is paid). Writes engine_heartbeat to system_state every 30s;
health.py /health endpoint treats a >5-minute-old beat as "starting".

Hang detection (2026-09-28 incident): a HUNG child never exits, so the old
"restart on exit" loop kept the service up forever while the engine did
nothing. The child now touches /tmp/child.beat every cycle and the
supervisor force-restarts it when that beat goes stale.

Keeps the same restart-on-crash semantics as the local systemd unit.
"""

import subprocess
import sys
import time
from pathlib import Path

HEARTBEAT_SEC = 30
CHILD_BEAT_FILE = Path("/tmp/child.beat")     # written by main.py each cycle
CHILD_STALE_SEC = 420                          # no beat for 7 min -> hung
BEAT_FILE = Path("/tmp/engine.beat")


def _write_heartbeat() -> None:
    """Heartbeat: file (always works) + DB state key (best-effort)."""
    try:
        BEAT_FILE.write_text(str(time.time()))
    except Exception:
        pass
    try:
        sys.path.insert(0, str(ROOT))
        from datetime import datetime, timezone

        from core.db import set_state

        set_state("engine_heartbeat", datetime.now(timezone.utc).isoformat())
    except Exception:
        pass


def _child_beat_stale() -> bool:
    """True when the child hasn't touched its beat file recently."""
    try:
        return time.time() - float(CHILD_BEAT_FILE.read_text().strip()) > CHILD_STALE_SEC
    except Exception:
        return True


def main() -> None:
    # Child stdout+stderr -> /tmp/child.log so /health can surface crashes
    # (Render's log API is unavailable on this workspace).
    logf = open("/tmp/child.log", "a", buffering=1)
    ROOT = Path(__file__).resolve().parent.parent
    while True:  # supervisor loop: restart engine if it exits OR hangs
        logf.write(f"\n[run_engine] starting engine child: main.py at {time.strftime('%H:%M:%S')}\n")
        print("[run_engine] starting engine child: main.py", flush=True)
        CHILD_BEAT_FILE.write_text(str(time.time()))  # grace period starts now
        child = subprocess.Popen(
            [sys.executable, "-u", "main.py"], cwd=str(ROOT),
            stdout=logf, stderr=subprocess.STDOUT,
        )
        last_beat = 0.0
        while True:
            code = child.poll()
            if code is not None:
                logf.write(f"[run_engine] engine exited with code {code}\n")
                logf.flush()
                print(f"[run_engine] engine exited with {code}; restarting in 10s", flush=True)
                time.sleep(10)
                break
            if time.time() - last_beat >= HEARTBEAT_SEC:
                _write_heartbeat()
                last_beat = time.time()
            # --- hang detection: stale child beat -> kill and restart ---
            # (beat file is written at spawn, so a slow-but-alive startup has
            # the full CHILD_STALE_SEC grace before any forced restart)
            if _child_beat_stale():
                logf.write("[run_engine] engine child HUNG (no beat in "
                           f"{CHILD_STALE_SEC}s); force-restarting\n")
                logf.flush()
                print("[run_engine] engine child hung; force-restarting", flush=True)
                child.kill()
                time.sleep(5)
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
