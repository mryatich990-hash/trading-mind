"""Telegram bot: alerts plus the full operator command set.

Commands: /status /trades /balance /pairs /research /stats /cot /sentiment
/vix /pause /resume /close /demo /live /enable /disable /backtest /kelly
/montecarlo /heatmap /version — all chat-ID-authorized, long-polling worker.
Alert helpers format exactly per the master prompt templates.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

import requests

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["TelegramBot", "make_bot"]

API = "https://api.telegram.org/bot{token}/{method}"


class TelegramBot:
    """Sends alerts and processes operator commands."""

    def __init__(self, token: str = "", chat_id: str = "") -> None:
        self.token = token or settings.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or settings.TELEGRAM_CHAT_ID
        self.session = requests.Session()
        self._authorized = {self.chat_id} if self.chat_id else set()
        self._handlers: dict[str, callable] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ---- low level ----

    def _call(self, method: str, payload: Optional[dict] = None,
              retries: int = 3) -> Optional[dict]:
        """Telegram API call with exponential backoff; None on failure."""
        if not self.token:
            return None
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.post(API.format(token=self.token, method=method),
                                         json=payload or {}, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                logger.warning("telegram %s attempt %d failed: %s", method, attempt, exc)
                time.sleep(2 ** attempt)
        return None

    def healthy(self) -> bool:
        """getMe responds."""
        return bool(self._call("getMe", retries=1))

    # ---- alerts ----

    def send(self, text: str, chat_id: Optional[str] = None) -> bool:
        """Send a message; returns success."""
        target = chat_id or self.chat_id
        if not self.token or not target:
            logger.info("telegram (no config): %s", text[:200])
            return False
        data = self._call("sendMessage", {"chat_id": target, "text": text[:4000],
                                          "parse_mode": "HTML"})
        return bool(data and data.get("ok"))

    # ---- command registration ----

    def register(self, command: str, handler: callable) -> None:
        """Register handler(args: list[str]) -> str for a /command."""
        self._handlers[command] = handler

    # ---- polling worker ----

    def start_polling(self) -> None:
        """Begin the long-poll loop in a daemon thread."""
        if not self.token or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True,
                                        name="telegram-poll")
        self._thread.start()
        logger.info("telegram polling started")

    def stop_polling(self) -> None:
        """Stop the polling loop."""
        self._running = False

    def _poll_loop(self) -> None:
        """Long-poll getUpdates and dispatch commands."""
        offset = 0
        while self._running:
            data = self._call("getUpdates", {"offset": offset, "timeout": 25}, retries=1)
            if not data:
                time.sleep(5)
                continue
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or {}
                self._handle_message(msg)

    def _handle_message(self, msg: dict) -> None:
        """Authorize and dispatch one message."""
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = str(msg.get("text", "")).strip()
        if not text or not chat_id:
            return
        if chat_id not in self._authorized:
            logger.warning("unauthorized telegram access from chat %s", chat_id)
            db.audit("security", "unauthorized_telegram", f"chat_id={chat_id}")
            self.send("Unauthorized. This incident has been logged.", chat_id)
            return
        if not text.startswith("/"):
            return
        parts = text.split()
        command = parts[0][1:].split("@")[0].lower()
        args = parts[1:]
        handler = self._handlers.get(command)
        if handler is None:
            self.send(f"Unknown command: /{command}")
            return
        try:
            reply = handler(args)
            if reply:
                self.send(reply)
        except Exception as exc:
            logger.exception("command /%s failed: %s", command, exc)
            self.send(f"⚠️ command failed: {exc}")


def make_bot(notifier_handlers: Optional[dict] = None) -> TelegramBot:
    """Build the bot wired to the operator command handlers from telegram_cmds."""
    bot = TelegramBot()
    try:
        from notifications.telegram_cmds import register_all
        register_all(bot)
    except Exception as exc:
        logger.warning("telegram command registration deferred: %s", exc)
    for cmd, fn in (notifier_handlers or {}).items():
        bot.register(cmd, fn)
    return bot
