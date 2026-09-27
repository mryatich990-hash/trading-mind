"""Email signal bridge: TradingView FREE-plan alert emails -> signal pipeline.

TradingView webhooks need a paid plan, but alert EMAILS are free. Point the
alert's notification at your inbox with this JSON as the message::

    {"secret": "<TV_WEBHOOK_SECRET>", "pair": "OANDA:EURUSD", "action": "buy",
     "price": 1.08450, "sl": 1.08200, "tp": 1.09000}

This bridge polls the inbox over IMAP (Gmail: app password + IMAP enabled),
picks up mail from TradingView, extracts that JSON from the body and feeds it
into the SAME pipeline as the webhook: parse -> dedupe -> _process_signal
(research verify -> risk -> execution). Processed mails are marked \\Seen so
nothing fires twice across restarts; a short in-memory dedupe guards against
duplicate deliveries.

Run standalone for testing:  python -m notifications.email_bridge
"""

from __future__ import annotations

import email
import imaplib
import json
import logging
import re
import time
from email.header import decode_header
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

SENDER_PATTERN = re.compile(r"@(tradingview\.com|notifications\.tradingview\.com)$", re.I)


def _allowed_domains() -> list[str]:
    """Configurable alert-sender allowlist (comma-separated domains)."""
    raw = getattr(settings, "BRIDGE_ALLOWED_SENDERS", "tradingview.com")
    return [d.strip().lower() for d in raw.split(",") if d.strip()]


def extract_alert_json(text: str) -> Optional[dict]:
    """Pull the alert JSON object out of an email body (first {...} blob).

    Returns None when the body has no JSON with an 'action' key.
    """
    start = text.find("{")
    while start != -1:
        end = text.rfind("}")
        if end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return None
        if isinstance(data, dict) and "action" in data:
            return data
        return None
    return None


def is_tradingview_sender(from_addr: str) -> bool:
    """True when the From address is on the alert-sender allowlist.

    Defaults to TradingView domains; extend BRIDGE_ALLOWED_SENDERS to accept
    other free alert sources (investing.com, finviz.com, stockcharts.com...).
    """
    addr = (from_addr or "").strip().lower()
    if SENDER_PATTERN.search(addr):
        return True
    return any(addr.endswith("@" + d) or addr.endswith("." + d)
               for d in _allowed_domains())


class EmailSignalBridge:
    """IMAP poller converting TradingView alert mails into pipeline signals."""

    def __init__(self, process_signal=None, host: str = "", user: str = "",
                 password: str = "", folder: str = "INBOX",
                 poll_sec: int = 20, secret: str = "") -> None:
        self.process_signal = process_signal
        self.host = host or settings.BRIDGE_IMAP_HOST
        self.user = user or settings.BRIDGE_EMAIL_USER
        self.password = password or settings.BRIDGE_EMAIL_PASSWORD
        self.folder = folder or settings.BRIDGE_IMAP_FOLDER
        self.poll_sec = poll_sec or settings.BRIDGE_POLL_SEC
        self.secret = secret if secret else settings.TV_WEBHOOK_SECRET
        self._stop = False
        self._thread: Optional[Any] = None
        self.tv_emails_seen = 0

    # ---- imap plumbing ----

    def _connect(self) -> imaplib.IMAP4_SSL:
        """Connect + login + select folder."""
        conn = imaplib.IMAP4_SSL(self.host, 993)
        conn.login(self.user, self.password)
        conn.select(self.folder, readonly=False)
        return conn

    def fetch_unread_alerts(self, conn: imaplib.IMAP4_SSL,
                            limit: int = 10) -> list[dict]:
        """Fetch unseen TradingView mails -> list of parsed alert dicts.

        Marks each mail \\Seen regardless of parse outcome so failures don't
        loop forever.
        """
        alerts: list[dict] = []
        status, data = conn.search(None, "UNSEEN")
        if status != "OK" or not data or not data[0]:
            return alerts
        ids = data[0].split()[-limit:]
        for mail_id in ids:
            status, parts = conn.fetch(mail_id, "(RFC822)")
            conn.store(mail_id, "+FLAGS", "\\Seen")
            if status != "OK" or not parts or not parts[0]:
                continue
            try:
                msg = email.message_from_bytes(parts[0][1])
                sender = email.utils.parseaddr(str(msg.get("From", "")))[1]
                if not is_tradingview_sender(sender):
                    continue
                body = self._body_text(msg)
                data = extract_alert_json(body)
                if data is None:
                    logger.info("bridge: TV mail without alert json skipped (%s)", sender)
                    continue
                data.setdefault("secret", "")
                if self.secret and data.get("secret") != self.secret:
                    logger.warning("bridge: TV alert with bad secret skipped")
                    continue
                self.tv_emails_seen += 1
                alerts.append(data)
            except Exception:
                logger.exception("bridge: failed to process mail %s", mail_id)
        return alerts

    @staticmethod
    def _body_text(msg: email.message.Message) -> str:
        """Plain-text body (first text/plain part, else stripped html)."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True) or b""
                    return payload.decode("utf-8", errors="replace")
            return ""
        payload = msg.get_payload(decode=True) or b""
        return payload.decode("utf-8", errors="replace")

    # ---- loop ----

    def run_forever(self) -> None:
        """Poll loop; never raises (errors logged, short backoff)."""
        backoff = self.poll_sec
        while not self._stop:
            try:
                conn = self._connect()
                try:
                    alerts = self.fetch_unread_alerts(conn)
                finally:
                    try:
                        conn.logout()
                    except Exception:
                        pass
                for data in alerts:
                    self._route(data)
                backoff = self.poll_sec
            except Exception as exc:
                logger.warning("bridge poll failed: %s", exc)
                backoff = min(backoff * 2, 300)
            for _ in range(max(1, int(backoff * 10))):
                if self._stop:
                    return
                time.sleep(0.1)

    def _route(self, data: dict) -> None:
        """Parse one alert and push it through the webhook pipeline logic."""
        from execution.tv_webhook import get_server

        server = get_server()
        # reuse the webhook's dedupe + pipeline via its internals
        server._process(json.dumps(data).encode(), ip="email-bridge")

    def start(self) -> None:
        """Start polling in a daemon thread (idempotent)."""
        if self._thread is not None or not self.user or not self.password:
            if not self.user or not self.password:
                logger.info("bridge: no email credentials -> disabled")
            return
        try:
            from execution.tv_webhook import get_server

            if get_server().secret != self.secret:
                logger.warning("bridge: TV_WEBHOOK_SECRET differs between bridge and "
                               "webhook server - email alerts will be rejected")
        except Exception:
            pass
        import threading

        self._thread = threading.Thread(target=self.run_forever,
                                        name="tv-email-bridge", daemon=True)
        self._thread.start()
        logger.info("bridge: polling %s for TV alert mails", self.user)

    def stop(self) -> None:
        """Signal the poll loop to exit."""
        self._stop = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bridge = EmailSignalBridge(process_signal=None)
    bridge.start()
    print("email bridge (notify-only) running - ctrl-c to stop")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        bridge.stop()
