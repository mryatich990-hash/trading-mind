"""Performance profiler (UPGRADE 7): time every major call against budgets.

Wraps research steps, Groq calls, indicator calculations and DB queries with
a context manager/decorator; rolling stats are kept in memory and hourly
summaries of the slowest functions go to the profiling_log table. Breaches
of PROFILE_BUDGETS raise a warning (and alert via Telegram when wired).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Optional

from config import settings

logger = logging.getLogger(__name__)

BUDGETS = settings.PROFILE_BUDGETS  # {"data_fetch": 5.0, ...}


class Profiler:
    """Function timing with budget alerts and hourly persistence."""

    def __init__(self, notifier=None) -> None:
        self.notifier = notifier
        self.stats: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self.budget_breaches: dict[str, int] = defaultdict(int)

    # ---- instrumentation ----

    def record(self, name: str, seconds: float) -> None:
        """Record one timing sample; alert on budget breach."""
        with self._lock:
            self.stats[name].append(seconds)
        budget = self._budget_for(name)
        if budget and seconds > budget:
            self.budget_breaches[name] += 1
            logger.warning("profile: %s took %.2fs (budget %.1fs)", name, seconds, budget)
            if self.notifier and self.budget_breaches[name] % 10 == 1:
                try:
                    self.notifier.send(f"⏱️ Performance: {name} {seconds:.1f}s "
                                       f"exceeds {budget}s budget")
                except Exception:
                    pass

    @staticmethod
    def _budget_for(name: str) -> Optional[float]:
        """Match a function name to its budget bucket."""
        lowered = name.lower()
        if "groq" in lowered or "verify" in lowered:
            return BUDGETS.get("groq")
        if "fetch" in lowered or "candle" in lowered or "feed" in lowered:
            return BUDGETS.get("data_fetch")
        if "indicator" in lowered or "atr" in lowered or "rsi" in lowered:
            return BUDGETS.get("indicators")
        if "research" in lowered or "evaluate" in lowered:
            return BUDGETS.get("total_research")
        return None

    def timed(self, name: str) -> Callable:
        """Decorator: @profiler.timed('research_evaluate')."""
        profiler = self

        def deco(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                started = time.monotonic()
                try:
                    return fn(*args, **kwargs)
                finally:
                    profiler.record(name, time.monotonic() - started)
            return wrapper
        return deco

    def context(self, name: str):
        """Context manager for inline timing."""
        profiler = self
        profiler_ref = self

        class _Timer:
            def __enter__(self):
                self.started = time.monotonic()
                return self

            def __exit__(self, *exc):
                profiler_ref.record(name, time.monotonic() - self.started)
                return False

        return _Timer()

    # ---- reporting ----

    def summary(self, top: int = 10) -> list[dict]:
        """Slowest functions by average over the rolling window."""
        rows = []
        with self._lock:
            for name, samples in self.stats.items():
                if samples:
                    rows.append({"function": name, "calls": len(samples),
                                 "avg_ms": round(sum(samples) / len(samples) * 1000, 1),
                                 "max_ms": round(max(samples) * 1000, 1)})
        rows.sort(key=lambda r: r["avg_ms"], reverse=True)
        return rows[:top]

    def flush_hourly(self) -> int:
        """Persist the hourly slowest-functions summary; returns rows written."""
        if time.time() - self._last_flush < 3600:
            return 0
        self._last_flush = time.time()
        summary = self.summary(10)
        if not summary:
            return 0
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine, _WRITE_LOCK, pg_compatible

            with engine.begin() as conn, _WRITE_LOCK:
                conn.execute(sqltext(pg_compatible(
                    "CREATE TABLE IF NOT EXISTS profiling_log ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TIMESTAMP "
                    "DEFAULT CURRENT_TIMESTAMP, summary_json TEXT)")))
                conn.execute(sqltext(
                    "INSERT INTO profiling_log (summary_json) VALUES (:j)"),
                    {"j": json_dumps(summary)})
            return len(summary)
        except Exception as exc:
            logger.debug("profile flush failed: %s", exc)
            return 0

    def status(self) -> dict:
        """Dashboard payload."""
        return {"budgets": BUDGETS, "slowest": self.summary(),
                "breaches": dict(self.budget_breaches)}


def json_dumps(obj: Any) -> str:
    """Compact JSON with fallbacks for exotic types."""
    import json

    return json.dumps(obj, default=str)


# process-wide singleton
_profiler: Optional[Profiler] = None


def get_profiler(notifier=None) -> Profiler:
    """Shared profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = Profiler(notifier=notifier)
    return _profiler
