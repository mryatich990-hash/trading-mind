"""Tests for the TradingView webhook receiver (parse, normalize, dedupe, route)."""

import json

import pytest

from execution.tv_webhook import (
    TVWebhookServer,
    TVWebhookState,
    alert_to_signal,
    normalize_ticker,
    parse_alert,
)

SECRET = "test-secret"
BODY = {
    "secret": SECRET, "pair": "OANDA:EURUSD", "action": "buy",
    "price": 1.08450, "sl": 1.08200, "tp": 1.09000,
}


def _body(**over) -> bytes:
    d = {**BODY, **over}
    return json.dumps(d).encode()


class TestNormalizeTicker:
    def test_exchange_prefix_stripped(self):
        assert normalize_ticker("OANDA:EURUSD") == "EURUSD"
        assert normalize_ticker("FX:EURUSD") == "EURUSD"
        assert normalize_ticker("CAPITALCOM_X:XAUUSD") == "XAUUSD"

    def test_aliases(self):
        assert normalize_ticker("GOLD") == "XAUUSD"
        assert normalize_ticker("US100") == "NAS100"
        assert normalize_ticker("DJ30") == "US30"

    def test_plain_symbol_passthrough(self):
        assert normalize_ticker("gbpjpy") == "GBPJPY"


class TestParseAlert:
    def test_valid_alert(self):
        data, err = parse_alert(_body(), secret=SECRET)
        assert err == ""
        assert data["pair"] == "OANDA:EURUSD"

    def test_bad_secret_rejected(self):
        data, err = parse_alert(_body(secret="wrong"), secret=SECRET)
        assert data is None and err == "bad secret"

    def test_missing_secret_config_accepts(self):
        # when no secret configured, parse succeeds (server warns upstream)
        data, err = parse_alert(_body(), secret="")
        assert err == "" and data is not None

    def test_bad_action(self):
        data, err = parse_alert(_body(action="yolo"), secret=SECRET)
        assert data is None and "bad action" in err

    def test_missing_fields(self):
        data, err = parse_alert(_body(sl=None), secret=SECRET)
        assert data is None and err == "missing field: sl"
        body = json.dumps({"secret": SECRET, "pair": "EURUSD", "action": "buy",
                           "price": 1.0, "sl": 0.9}).encode()
        data, err = parse_alert(body, secret=SECRET)
        assert data is None and err == "missing field: tp"

    def test_invalid_json(self):
        data, err = parse_alert(b"not json", secret=SECRET)
        assert data is None and err == "invalid json"

    def test_body_too_large(self):
        data, err = parse_alert(b"x" * 5000, secret=SECRET)
        assert data is None and err == "body too large"

    def test_exit_needs_no_prices(self):
        body = json.dumps({"secret": SECRET, "pair": "EURUSD",
                           "action": "exit"}).encode()
        data, err = parse_alert(body, secret=SECRET)
        assert err == "" and data["action"] == "exit"


class TestAlertToSignal:
    def test_signal_fields(self):
        signal = alert_to_signal(parse_alert(_body(), secret=SECRET)[0])
        assert signal is not None
        assert signal.pair == "EURUSD"
        assert signal.direction == "buy"
        assert signal.strategy.startswith("tv_")
        assert signal.entry == 1.08450 and signal.sl == 1.08200
        assert "tradingview_alert" in signal.confluences

    def test_exit_returns_none(self):
        assert alert_to_signal({"action": "exit"}) is None


class TestDedupe:
    def test_same_alert_within_window(self):
        state = TVWebhookState()
        assert not state.seen_before("EURUSD", "buy", 1.0845)
        assert state.seen_before("EURUSD", "buy", 1.08450)

    def test_different_direction_or_price(self):
        state = TVWebhookState()
        assert not state.seen_before("EURUSD", "buy", 1.0845)
        assert not state.seen_before("EURUSD", "sell", 1.0845)
        assert not state.seen_before("EURUSD", "buy", 1.0846)


class TestServerRouting:
    def _server(self, recorded):
        def fake_pipeline(signal):
            recorded.append(signal)

        return TVWebhookServer(process_signal=fake_pipeline, secret=SECRET)

    def test_accepted_alert_reaches_pipeline(self, temp_db):
        recorded: list = []
        server = self._server(recorded)
        server._process(_body(), ip="52.89.214.188")
        assert len(recorded) == 1
        assert recorded[0].pair == "EURUSD"

    def test_bad_secret_never_reaches_pipeline(self, temp_db):
        recorded: list = []
        server = self._server(recorded)
        server._process(_body(secret="nope"), ip="52.89.214.188")
        assert recorded == []

    def test_duplicate_not_processed_twice(self, temp_db):
        recorded: list = []
        server = self._server(recorded)
        server._process(_body(), ip="52.89.214.188")
        server._process(_body(), ip="52.89.214.188")
        assert len(recorded) == 1

    def test_notify_only_mode_logs_without_pipeline(self, temp_db):
        recorded: list = []
        server = TVWebhookServer(process_signal=None, secret=SECRET)
        server._process(_body(), ip="52.89.214.188")  # must not raise
        assert recorded == []

    def test_exit_routes_to_close_path(self, temp_db):
        recorded: list = []
        server = self._server(recorded)
        # no open trades in temp db -> must not raise and must not signal
        server._process(json.dumps({"secret": SECRET, "pair": "EURUSD",
                                    "action": "exit"}).encode(), ip="52.89.214.188")
        assert recorded == []
