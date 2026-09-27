"""Redis caching layer (UPGRADE 7): faster research with graceful fallback.

Caches indicator values, COT, sentiment, strategy weights and the correlation
matrix with per-type TTLs. Falls back to a process-local dict when Redis is
unavailable (REDIS_URL empty or connection refused) — the API is identical.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

try:  # guarded dependency
    import redis  # type: ignore
    REDIS_AVAILABLE = True
except Exception:  # pragma: no cover
    redis = None
    REDIS_AVAILABLE = False

TTL = {"indicators": 900, "cot": 7 * 86400, "sentiment": 1800,
       "strategy_weights": 3600, "correlation": 900, "default": 600}


class CacheLayer:
    """Redis-first, memory-fallback key/value store with TTLs."""

    def __init__(self) -> None:
        self.backend = "memory"
        self._client = None
        self._memory: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if settings.REDIS_URL and REDIS_AVAILABLE:
            try:
                self._client = redis.Redis.from_url(settings.REDIS_URL,
                                                    socket_connect_timeout=2,
                                                    decode_responses=True)
                self._client.ping()
                self.backend = "redis"
            except Exception as exc:
                logger.warning("redis unavailable (%s) - memory cache active", exc)
                self._client = None

    # ---- core ----

    def _ttl(self, namespace: str) -> int:
        return TTL.get(namespace, TTL["default"])

    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Fetch a cached value (JSON round-trip) or None on miss/expiry."""
        full = f"{namespace}:{key}"
        if self.backend == "redis" and self._client is not None:
            try:
                raw = self._client.get(full)
                if raw is None:
                    self.misses += 1
                    return None
                self.hits += 1
                return json.loads(raw)
            except Exception:
                pass
        with self._lock:
            entry = self._memory.get(full)
            if entry is None:
                self.misses += 1
                return None
            ts, raw = entry
            if time.time() - ts > self._ttl(namespace):
                del self._memory[full]
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(raw)

    def set(self, key: str, value: Any, namespace: str = "default",
            ttl_sec: Optional[int] = None) -> None:
        """Store a JSON-serializable value."""
        full = f"{namespace}:{key}"
        raw = json.dumps(value, default=str)
        ttl = ttl_sec or self._ttl(namespace)
        if self.backend == "redis" and self._client is not None:
            try:
                self._client.setex(full, ttl, raw)
                return
            except Exception:
                pass
        with self._lock:
            if len(self._memory) > 2000:
                oldest = min(self._memory, key=lambda k: self._memory[k][0])
                del self._memory[oldest]
            self._memory[full] = (time.time(), raw)

    def get_or_set(self, key: str, producer, namespace: str = "default",
                   ttl_sec: Optional[int] = None) -> Any:
        """Cached read-through: call producer() only on miss."""
        value = self.get(key, namespace)
        if value is None:
            value = producer()
            if value is not None:
                self.set(key, value, namespace, ttl_sec)
        return value

    # ---- metrics ----

    def hit_rate(self) -> float:
        """Cache hit rate percentage."""
        total = self.hits + self.misses
        return round(self.hits / total * 100.0, 1) if total else 0.0

    def status(self) -> dict:
        """Dashboard payload."""
        return {"backend": self.backend, "hit_rate": self.hit_rate(),
                "hits": self.hits, "misses": self.misses,
                "memory_keys": len(self._memory)}


# process-wide singleton
_cache: Optional[CacheLayer] = None


def get_cache() -> CacheLayer:
    """Shared cache instance."""
    global _cache
    if _cache is None:
        _cache = CacheLayer()
    return _cache
