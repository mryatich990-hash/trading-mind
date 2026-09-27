"""Production engine runner: heartbeat + loop watchdog.

Wrapped by health.py on Render so the FREE web tier can host the engine
(worker tier is paid). Writes engine_heartbeat to system_state every 30s;
health.py /health endpoint treats a >5-minute-old beat as "starting".

Keeps the same restart-on-crash semantics as the local systemd unit.
"""

import subprocess
import sys
import time
from pathlib import Path

HEARTBEAT_SEC = 30

ROOT = Path(__file__).resolve().parent.parent
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


def main() -> None:
    while True:  # supervisor loop: restart engine if it ever exits
        print("[run_engine] starting engine child: main.py", flush=True)
        child = subprocess.Popen([sys.executable, "main.py"], cwd=str(ROOT))
        last_beat = 0.0
        while True:
            code = child.poll()
            if code is not None:
                print(f"[run_engine] engine exited with {code}; restarting in 10s",
                      flush=True)
                time.sleep(10)
                break
            if time.time() - last_beat >= HEARTBEAT_SEC:
                _write_heartbeat()
                last_beat = time.time()
            time.sleep(1)


if __name__ == "__main__":
    main()
