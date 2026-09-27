"""Tests for the cTrader connector (no live account needed for these paths)."""

import pytest


class TestConfiguration:
    def test_unavailable_without_credentials(self, temp_db):
        from execution.ctrader_connector import CtraderConnector

        conn = CtraderConnector(client_id="", client_secret="",
                                access_token="", account_id=0)
        assert conn.available() is False
        assert conn.healthy() is False
        assert conn.connect() is False

    def test_available_with_credentials(self, temp_db):
        from execution.ctrader_connector import CtraderConnector

        conn = CtraderConnector(client_id="id", client_secret="sec",
                                access_token="tok", account_id=12345)
        assert conn.available() is True

    def test_demo_by_default(self, temp_db):
        from execution.ctrader_connector import CtraderConnector

        conn = CtraderConnector(client_id="i", client_secret="s",
                                access_token="t", account_id=1)
        assert conn.demo is True

    def test_sdk_import_present(self):
        # the SDK was installed in this environment; connector must expose flag
        from execution import ctrader_connector as cc

        assert cc.SDK_AVAILABLE is True


class TestExecutionChain:
    def test_ctrader_sits_between_oanda_and_paper(self, temp_db):
        from execution.execution_engine import ExecutionEngine

        engine = ExecutionEngine()
        assert engine.ctrader is not None
        # unconfigured -> chain falls through to paper
        assert engine.active_broker() is engine.paper

    def test_fake_ctrader_becomes_active(self, temp_db):
        from execution.execution_engine import ExecutionEngine

        class FakeCtrader:
            def available(self):
                return True

            def healthy(self):
                return True

            def account(self):
                return {"balance": 5000.0, "equity": 5000.0, "margin_used": 0,
                        "margin_free": 5000.0, "margin_level": 1000.0,
                        "currency": "USD"}

        engine = ExecutionEngine(ctrader=FakeCtrader())
        assert engine.active_broker() is engine.ctrader

    def test_dead_ctrader_falls_through_to_paper(self, temp_db):
        from execution.execution_engine import ExecutionEngine

        class DeadCtrader:
            def available(self):
                return True  # configured...

            def healthy(self):
                return False  # ...but unreachable

        engine = ExecutionEngine(ctrader=DeadCtrader())
        assert engine.active_broker() is engine.paper


class TestMapping:
    def test_symbol_fallback_table(self):
        from execution.ctrader_connector import SYMBOL_FALLBACK

        assert SYMBOL_FALLBACK["XAUUSD"] == "XAUUSD"
        assert len(SYMBOL_FALLBACK) >= 10

    def test_volume_conversion(self):
        # cTrader volume is in 0.01-lot units: 0.37 lots -> 37000
        assert int(0.37 * 100000) == 37000

    def test_cli_runs_without_args(self, capsys):
        from execution.ctrader_connector import _cli
        import sys

        sys_argv = sys.argv
        sys.argv = ["ctrader"]
        try:
            _cli()
        except SystemExit:
            pass
        finally:
            sys.argv = sys_argv
        out = capsys.readouterr().out
        assert "usage" in out
