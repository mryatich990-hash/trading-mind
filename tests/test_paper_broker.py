"""Tests for the internal paper broker (fills, PnL, persistence, interface)."""

import pytest


@pytest.fixture()
def broker(temp_db):
    """Paper broker against a temp DB with a stubbed data engine."""
    from execution.paper_broker import PaperBroker

    class StubData:
        price = 1.0850

        def get_candles(self, pair, tf, count):
            from types import SimpleNamespace

            import pandas as pd

            df = pd.DataFrame({
                "close": [self.price, self.price, self.price]})
            return SimpleNamespace(df=df, last_close=self.price, spread_pips=1.0)

    return PaperBroker(data_engine=StubData())


class TestInterface:
    def test_available_healthy(self, broker):
        assert broker.available() and broker.healthy() and broker.connect()

    def test_account_shape(self, broker):
        acct = broker.account()
        for key in ("balance", "equity", "margin_used", "margin_level", "currency"):
            assert key in acct
        assert acct["balance"] == 10000.0

    def test_tick_spread(self, broker):
        tick = broker.tick("EURUSD")
        assert tick["ask"] > tick["bid"] > 0
        assert tick["spread_pips"] >= 1.0


class TestOrders:
    def test_market_order_fills_with_slippage(self, broker):
        ticket = broker.market_order("EURUSD", "buy", 0.5, 1.0800, 1.0900, "test")
        assert ticket >= 1001
        positions = broker.positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos["direction"] == "buy" and pos["lots"] == 0.5
        # buy fills at ask: entry above mid 1.0850
        assert pos["entry"] > 1.0850
        # slippage is adverse: never better than the raw ask
        assert pos["entry"] >= broker.tick("EURUSD")["ask"] - 0.00001

    def test_sell_fills_below_mid(self, broker):
        broker.market_order("EURUSD", "sell", 0.2, 1.0900, 1.0800)
        assert broker.positions()[0]["entry"] < 1.0850

    def test_close_realizes_pnl(self, broker):
        broker.market_order("EURUSD", "buy", 1.0, 1.0800, 1.0900)
        before = broker.account()["balance"]
        assert broker.close_position(broker.positions()[0]["ticket"]) is True
        after = broker.account()["balance"]
        # round trip pays spread both ways + slippage: small loss, ~$13-35
        assert before - 40 <= after <= before
        assert broker.positions() == []

    def test_partial_close(self, broker):
        broker.market_order("EURUSD", "buy", 1.0, 1.0800, 1.0900)
        ticket = broker.positions()[0]["ticket"]
        broker.close_position(ticket, lots=0.35)
        remaining = [p for p in broker.positions() if p["ticket"] == ticket]
        assert len(remaining) == 1 and abs(remaining[0]["lots"] - 0.65) < 0.001

    def test_modify_sl_tp(self, broker):
        broker.market_order("EURUSD", "buy", 0.1, 1.0800, 1.0900)
        ticket = broker.positions()[0]["ticket"]
        assert broker.modify_sl_tp(ticket, 1.0840, 1.0950) is True
        assert broker.positions()[0]["sl"] == 1.0840

    def test_close_unknown_ticket(self, broker):
        assert broker.close_position(999999) is False


class TestPersistence:
    def test_state_survives_reinstantiation(self, temp_db):
        from execution.paper_broker import PaperBroker

        class StubData:
            def get_candles(self, pair, tf, count):
                from types import SimpleNamespace

                import pandas as pd

                return SimpleNamespace(df=pd.DataFrame({"close": [1.085] * 3}),
                                       last_close=1.085, spread_pips=1.0)

        b1 = PaperBroker(data_engine=StubData())
        b1.market_order("EURUSD", "buy", 0.5, 1.0800, 1.0900)
        bal_after_open = b1.account()["balance"]

        b2 = PaperBroker(data_engine=StubData())
        assert len(b2.positions()) == 1
        assert b2.account()["balance"] == bal_after_open


class TestReset:
    def test_reset_clears_everything(self, broker):
        broker.market_order("EURUSD", "buy", 0.5, 1.0800, 1.0900)
        broker.reset()
        assert broker.positions() == []
        assert broker.account()["balance"] == 10000.0
