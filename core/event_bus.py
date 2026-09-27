"""Asyncio event bus: every module communicates through typed queues.

Publishers call ``await bus.publish(EventType.X, payload)``; subscribers either
register an async handler or consume from their own queue. The bus guarantees
handler exceptions never kill the engine — they're logged and alerted.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from core.logging_utils import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """All event types flowing through the system."""

    CANDLE_CLOSED = "candle_closed"
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_ACCEPTED = "signal_accepted"
    SIGNAL_REJECTED = "signal_rejected"
    RESEARCH_COMPLETED = "research_completed"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    TRADE_UPDATED = "trade_updated"
    REGIME_CHANGED = "regime_changed"
    BREAKER_TRIGGERED = "breaker_triggered"
    BREAKER_CLEARED = "breaker_cleared"
    NEWS_ALERT = "news_alert"
    HEALTH_ALERT = "health_alert"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_REVIEW = "weekly_review"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    """An event envelope."""

    type: EventType
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.monotonic)


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Central async pub/sub with per-subscriber bounded queues."""

    def __init__(self, queue_size: int = 500) -> None:
        self._subscribers: dict[EventType, list[asyncio.Queue]] = defaultdict(list)
        self._handlers: list[Handler] = []
        self._queue_size = queue_size
        self.published = 0
        self.dropped = 0

    def subscribe(self, *types: EventType) -> asyncio.Queue:
        """Register a bounded queue receiving the given event types."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        for t in types:
            self._subscribers[t].append(q)
        return q

    def on(self, handler: Handler) -> None:
        """Register a global async handler invoked for every event."""
        self._handlers.append(handler)

    async def publish(self, etype: EventType, payload: Optional[dict] = None) -> None:
        """Fan an event out to all subscriber queues and global handlers."""
        event = Event(type=etype, payload=payload or {})
        self.published += 1
        for q in self._subscribers[etype]:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped += 1
                logger.warning("event queue full, dropping %s event", etype.value)
        for handler in self._handlers:
            try:
                await handler(event)
            except Exception as exc:
                logger.exception("event handler failed on %s: %s", etype.value, exc)

    async def run_consumer(self, q: asyncio.Queue, worker: Callable[[Event], Awaitable[None]]) -> None:
        """Consume a queue until cancelled; worker exceptions are contained."""
        while True:
            event = await q.get()
            try:
                await worker(event)
            except Exception as exc:
                logger.exception("consumer failed on %s: %s", event.type.value, exc)
            finally:
                q.task_done()


bus = EventBus()
