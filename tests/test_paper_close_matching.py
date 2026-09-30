"""Regression test: closing a trade must settle the PAPER position.

The trades table has no ticket column, so TradeManager used the row id as
the paper ticket -> "unknown ticket" -> the paper position stayed open and
P&L was never realized into the balance. close_matching() reconciles on
pair+direction instead.
"""

from execution.paper_broker import PaperBroker


def test_close_matching_settles_pnl(temp_db, monkeypatch):
    broker = PaperBroker(data_engine=None, start_balance=10000.0)
    monkeypatch.setattr("execution.paper_broker.random.uniform", lambda a, b: 0.0)
    # open fills at ask, close exits at bid -> stub both sides explicitly
    broker.tick = lambda pair: {"bid": 157.2600, "ask": 157.2600,
                                "spread_pips": 0.0}
    ticket = broker.market_order("USDJPY", "buy", 2.42, sl=157.21, tp=157.30,
                                 comment="liquidity_sweep")
    assert ticket in broker._positions

    broker.tick = lambda pair: {"bid": 157.3100, "ask": 157.3100,
                                "spread_pips": 0.0}  # +5 pips in profit
    closed = broker.close_matching("USDJPY", "buy")
    assert closed == ticket
    assert ticket not in broker._positions
    assert broker._balance > 10000.0  # P&L realized into the book
    broker._save()
    from core import db as core_db
    assert float(core_db.get_state("paper_balance")) > 10000.0
    assert core_db.get_state("paper_positions", "{}") == "{}"


def test_close_matching_no_match_returns_none(temp_db):
    broker = PaperBroker(data_engine=None, start_balance=10000.0)
    broker._mid = lambda pair: 1.1000
    broker.market_order("EURUSD", "buy", 0.01, sl=1.098, tp=1.106)
    assert broker.close_matching("USDJPY", "buy") is None
    assert broker.close_matching("EURUSD", "sell") is None
    assert len(broker._positions) == 1
