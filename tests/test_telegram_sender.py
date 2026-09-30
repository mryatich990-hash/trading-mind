"""Regression tests for Telegram sender fixes.

1. sendMessage must be PLAIN TEXT (parse_mode=HTML caused permanent 400s on
   dynamic alert text like prices and '<' comparisons).
2. Long-poll getUpdates needs a read timeout strictly greater than the poll
   window (25s poll vs 15s read timeout timed out every ~22s on Render).
"""

from types import SimpleNamespace

from notifications.telegram_bot import TelegramBot


class _FakeResp:
    def __init__(self, ok=True, status=200, body=None):
        self._body = body if body is not None else {"ok": ok}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _bot(monkeypatch, calls):
    bot = TelegramBot(token="TESTTOKEN", chat_id="42")

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "payload": json, "timeout": timeout})
        return _FakeResp(ok=True, body={"ok": True, "result": []})

    bot.session = SimpleNamespace(post=fake_post)
    return bot


def test_send_is_plain_text_no_parse_mode(monkeypatch):
    calls = []
    bot = _bot(monkeypatch, calls)
    assert bot.send("▶ USDJPY BUY @ 157.26, risk if < 157.20") is True
    payload = calls[0]["payload"]
    assert "parse_mode" not in payload  # the 400 source is gone
    assert payload["text"].startswith("▶ USDJPY")


def test_getupdates_read_timeout_exceeds_poll_window(monkeypatch):
    calls = []
    bot = _bot(monkeypatch, calls)
    bot._call("getUpdates", {"offset": 0, "timeout": 25},
              retries=1, read_timeout=40)
    connect_to, read_to = calls[0]["timeout"]
    assert read_to >= 35  # must outlive the 25s long-poll window


def test_permanent_400_not_retried(monkeypatch):
    calls = []
    bot = TelegramBot(token="TESTTOKEN", chat_id="42")
    attempts = []

    def fake_post(url, json=None, timeout=None):
        attempts.append(1)
        resp = SimpleNamespace(status_code=400, json=lambda: {
            "ok": False, "description": "Bad Request: chat not found"})
        raise SimpleNamespace(response=resp, __class__=Exception)

    # simulate requests.HTTPError path with a real exception object
    import requests

    def fake_post2(url, json=None, timeout=None):
        attempts.append(1)
        resp = requests.Response()
        resp.status_code = 400
        resp._content = b'{"ok":false,"description":"Bad Request: chat not found"}'
        raise requests.HTTPError(response=resp)

    bot.session = SimpleNamespace(post=fake_post2)
    result = bot._call("sendMessage", {"chat_id": "42", "text": "x"})
    assert result is None
    assert len(attempts) == 1  # permanent 4xx: no retry burn


def test_transient_error_retried(monkeypatch):
    attempts = []
    bot = TelegramBot(token="TESTTOKEN", chat_id="42")

    def fake_post(url, json=None, timeout=None):
        attempts.append(1)
        if len(attempts) < 2:
            raise ConnectionError("boom")
        return _FakeResp(ok=True, body={"ok": True})

    bot.session = SimpleNamespace(post=fake_post)
    monkeypatch.setattr("notifications.telegram_bot.time.sleep", lambda s: None)
    result = bot._call("sendMessage", {"chat_id": "42", "text": "x"})
    assert result == {"ok": True}
    assert len(attempts) == 2
