"""Hot reload (UPGRADE 7): live config changes without restart.

Watches an override file (config/runtime_overrides.json) every
HOT_RELOAD_SEC; recognized keys are applied to settings at runtime, every
change is audit-logged and pushed to Telegram. Allowed keys: risk %, max
trades, pairs, strategy weights, confluence minimum.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

OVERRIDES_PATH = os.path.join("config", "runtime_overrides.json")

ALLOWED_KEYS = {
    "RISK_PER_TRADE_PCT": (float, 0.05, 2.0),
    "MAX_TRADES_PER_DAY": (int, 1, 50),
    "MIN_CONFLUENCE": (int, 1, 10),
    "MAX_OPEN_TRADES": (int, 1, 10),
    "TRADING_PAIRS": (list, None, None),
}


def apply_overrides(data: dict, audit_fn=None) -> list[str]:
    """Apply allowed runtime overrides to settings; returns applied keys."""
    applied = []
    for key, value in data.items():
        spec = ALLOWED_KEYS.get(key)
        if spec is None:
            logger.warning("hot reload: key %s not allowed", key)
            continue
        cast, low, high = spec
        try:
            if cast is list:
                value = [str(v).strip().upper() for v in value if str(v).strip()]
                if not value:
                    continue
                settings.TRADING_PAIRS = value
            else:
                value = cast(value)
                if low is not None and not (low <= value <= high):
                    logger.warning("hot reload: %s=%s out of range", key, value)
                    continue
                setattr(settings, key, value)
            applied.append(key)
            if audit_fn:
                audit_fn("config", "hot_reload", f"{key} = {value}")
        except (TypeError, ValueError) as exc:
            logger.warning("hot reload: bad value for %s: %s (%s)", key, value, exc)
    return applied


class HotReloader:
    """Background watcher for runtime_overrides.json."""

    def __init__(self, notifier=None) -> None:
        self.notifier = notifier
        self._mtime = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.applied_history: list[dict] = []
        if os.path.exists(OVERRIDES_PATH):
            try:
                self._mtime = os.path.getmtime(OVERRIDES_PATH)
            except OSError:
                pass

    def check_once(self) -> list[str]:
        """Apply the override file when it changed; returns applied keys."""
        try:
            mtime = os.path.getmtime(OVERRIDES_PATH)
        except OSError:
            return []
        if mtime == self._mtime:
            return []
        self._mtime = mtime
        try:
            with open(OVERRIDES_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
        except (ValueError, OSError) as exc:
            logger.warning("hot reload: unreadable overrides: %s", exc)
            return []
        applied = apply_overrides(data, audit_fn=self._audit)
        if applied:
            entry = {"ts": time.time(), "keys": applied}
            self.applied_history.append(entry)
            if self.notifier:
                try:
                    self.notifier.send(f"🔧 Config reloaded live: {', '.join(applied)}")
                except Exception:
                    pass
            logger.info("hot reload applied: %s", applied)
        return applied

    def _audit(self, actor: str, action: str, detail: str) -> None:
        try:
            from core import db

            db.audit(actor, action, detail, source="hot_reload")
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.check_once()
            self._stop.wait(settings.HOT_RELOAD_SEC)

    def start(self) -> None:
        """Start the watcher thread (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="hot-reload", daemon=True)
        self._thread.start()
        logger.info("hot reload watching %s every %ss", OVERRIDES_PATH,
                    settings.HOT_RELOAD_SEC)

    def stop(self) -> None:
        """Stop watching."""
        self._stop.set()

    def status(self) -> dict:
        """Dashboard payload."""
        return {"path": OVERRIDES_PATH, "allowed_keys": sorted(ALLOWED_KEYS),
                "applied": self.applied_history[-10:]}
