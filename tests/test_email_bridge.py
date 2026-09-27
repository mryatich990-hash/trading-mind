"""Tests for the TradingView email bridge (extract, sender filter, routing)."""

import email.message
import json

import pytest

from notifications.email_bridge import (
    EmailSignalBridge,
    extract_alert_json,
    is_tradingview_sender,
)

SECRET = "bridge-secret"
ALERT = {"secret": SECRET, "pair": "OANDA:EURUSD", "action": "buy",
         "price": 1.0845, "sl": 1.082, "tp": 1.09}


def _mail(from_addr: str, body: str) -> email.message.Message:
    msg = email.message.Message()
    msg["From"] = from_addr
    msg.set_payload(body)
    return msg


class TestExtractAlertJson:
    def test_plain_json(self):
        assert extract_alert_json(json.dumps(ALERT))["action"] == "buy"

    def test_json_inside_prose(self):
        body = f"Your alert fired!\n\n{json.dumps(ALERT)}\n\n-- TradingView"
        data = extract_alert_json(body)
        assert data is not None and data["pair"] == "OANDA:EURUSD"

    def test_no_json(self):
        assert extract_alert_json("Price crossed the level.") is None

    def test_json_without_action(self):
        assert extract_alert_json('{"hello": "world"}') is None


class TestSenderFilter:
    def test_tradingview_domains(self):
        assert is_tradingview_sender("noreply@tradingview.com")
        assert is_tradingview_sender("alerts@notifications.tradingview.com")

    def test_other_senders(self):
        assert not is_tradingview_sender("spam@gmail.com")
        assert not is_tradingview_sender("fake@nottradingview.com")


class TestBodyText:
    def test_plain_payload(self):
        bridge = EmailSignalBridge(secret=SECRET)
        msg = _mail("noreply@tradingview.com", json.dumps(ALERT))
        assert bridge._body_text(msg) == json.dumps(ALERT)


class TestFetchUnreadAlerts:
    def _fake_conn(self, mails):
        """Minimal IMAP conn stub: search -> ids, fetch -> RFC822 bytes."""
        class FakeConn:
            def __init__(self):
                self.seen = []

            def search(self, *a):
                return "OK", [b" ".join(str(i + 1).encode() for i in range(len(mails)))]

            def fetch(self, mail_id, spec):
                idx = int(mail_id) - 1
                return "OK", [(b"1 (RFC822 {100}", mails[idx].as_bytes()), b")"]

            def store(self, mail_id, flags, value):
                self.seen.append(mail_id)
                return "OK", [""]

        return FakeConn()

    def test_alert_mail_parsed_and_marked_seen(self, temp_db):
        msg = _mail("noreply@tradingview.com", json.dumps(ALERT))
        bridge = EmailSignalBridge(secret=SECRET)
        conn = self._fake_conn([msg])
        alerts = bridge.fetch_unread_alerts(conn)
        assert len(alerts) == 1 and alerts[0]["action"] == "buy"
        assert conn.seen == [b"1"]

    def test_non_tv_sender_ignored(self, temp_db):
        msg = _mail("someone@gmail.com", json.dumps(ALERT))
        bridge = EmailSignalBridge(secret=SECRET)
        alerts = bridge.fetch_unread_alerts(self._fake_conn([msg]))
        assert alerts == []

    def test_bad_secret_rejected(self, temp_db):
        bad = dict(ALERT, secret="wrong")
        msg = _mail("noreply@tradingview.com", json.dumps(bad))
        bridge = EmailSignalBridge(secret=SECRET)
        alerts = bridge.fetch_unread_alerts(self._fake_conn([msg]))
        assert alerts == []


class TestRouting:
    def test_route_goes_through_webhook_pipeline(self, temp_db):
        from execution.tv_webhook import get_server

        recorded: list = []

        def fake_pipeline(signal):
            recorded.append(signal)

        server = get_server(process_signal=fake_pipeline)
        server.secret = SECRET  # production: both read the same settings value
        from execution.tv_webhook import TVWebhookState
        server.state = TVWebhookState()  # fresh dedupe state per test
        bridge = EmailSignalBridge(secret=SECRET)
        bridge._route(dict(ALERT))
        assert len(recorded) == 1 and recorded[0].pair == "EURUSD"

    def test_route_dedupes(self, temp_db):
        from execution.tv_webhook import get_server

        recorded: list = []
        server = get_server(process_signal=recorded.append)
        server.secret = SECRET
        from execution.tv_webhook import TVWebhookState
        server.state = TVWebhookState()
        bridge = EmailSignalBridge(secret=SECRET)
        bridge._route(dict(ALERT))
        bridge._route(dict(ALERT))
        assert len(recorded) == 1
