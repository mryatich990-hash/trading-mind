#!/bin/sh
# Render start script: engine supervisor in background, web in foreground.
set -e

# Resolve a Python interpreter (some images only ship python3).
PYBIN="$(command -v python3 || command -v python)"
echo "[start_render] using interpreter: $PYBIN"

# Engine supervisor (restarts main.py forever, writes heartbeat).
RUN_ENGINE=1 "$PYBIN" -u deploy/run_engine.py &

# Web process (health.py) in foreground; RUN_ENGINE=0 prevents a 2nd engine.
export RUN_ENGINE=0
exec gunicorn --bind "0.0.0.0:${PORT:-8080}" --workers 1 --threads 8 \
  --timeout 120 --preload health:app
