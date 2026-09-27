"""Graceful shutdown handler: SIGTERM/SIGINT -> close trades, save state, alert."""

import asyncio
import signal
from typing import Optional

from core import db
from core.event_bus import EventBus, EventType
from core.logging_utils import get_logger

logger = get_logger(__name__)


class GracefulShutdown:
    """Coordinates an orderly stop of the trading system."""

    def __init__(self, bus: EventBus, notifier: Optional[callable] = None) -> None:
        self.bus = bus
        self.notifier = notifier or (lambda text: logger.warning("ALERT: %s", text))
        self.shutting_down = False
        self._stop_event = asyncio.Event()

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        """Wire SIGTERM/SIGINT to request_shutdown."""

        def _handler(signum, frame):
            logger.info("signal %s received", signum)
            loop.call_soon_threadsafe(self.request_shutdown)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def request_shutdown(self) -> None:
        """Idempotently trigger the shutdown sequence."""
        if not self.shutting_down:
            self.shutting_down = True
            self._stop_event.set()

    async def wait(self) -> None:
        """Block until shutdown requested."""
        await self._stop_event.wait()

    async def run(self, close_all_trades: callable) -> None:
        """Execute the shutdown sequence and alert."""
        logger.info("graceful shutdown starting")
        try:
            await self.bus.publish(EventType.SHUTDOWN, {"reason": "signal"})
            closed = await close_all_trades()
            db.set_state("shutdown_at", db._utcnow().isoformat())
            db.set_state("running", "0")
            db.audit("system", "shutdown", f"closed {closed} trades")
            self.notifier(f"🛑 Bot shutting down safely. {closed} open trades closed. "
                          "State saved. See you next session.")
        except Exception as exc:
            logger.exception("shutdown sequence error: %s", exc)
            self.notifier(f"⚠️ Shutdown encountered an error: {exc}")
