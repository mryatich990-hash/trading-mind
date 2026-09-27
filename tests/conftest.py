"""Shared pytest fixtures: temp SQLite DB with the full schema applied."""

import os
import sys

import pytest

# ensure repo root on sys.path when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Point core.db at a throwaway SQLite file with the full schema."""
    import core.db as core_db

    test_engine = create_engine(f"sqlite:///{tmp_path}/test.db", future=True)
    monkeypatch.setattr(core_db, "engine", test_engine)
    monkeypatch.setattr(core_db, "SessionLocal", sessionmaker(bind=test_engine,
                                                              expire_on_commit=False,
                                                              future=True))
    # reset schema-init flag so init_db runs against the temp engine
    monkeypatch.setattr(core_db, "_INITIALIZED", False)
    # route init_db's engine usage through module attribute lookup
    monkeypatch.setattr(core_db, "init_db", _make_init(test_engine))
    core_db.init_db()
    yield test_engine


def _make_init(test_engine):
    """Build an init_db that applies the schema to the given engine."""
    from sqlalchemy import text as sqltext

    def _init() -> None:
        schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "config", "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as fh:
            statements = [s.strip() for s in fh.read().split(";") if s.strip()]
        with test_engine.begin() as conn:
            for stmt in statements:
                try:
                    conn.execute(sqltext(stmt))
                except Exception:
                    continue  # engine-specific syntax skipped on SQLite

    return _init


@pytest.fixture()
def seeded_db(temp_db):
    """Temp DB pre-loaded with a closed-trade history for Kelly/stats tests."""
    from core import db

    results = [(1, 120.0), (1, 85.0), (0, -100.0), (1, 150.0), (0, -95.0),
               (1, 60.0), (0, -100.0), (1, 110.0), (1, 70.0), (0, -100.0),
               (1, 90.0), (0, -80.0)]
    for i, (won, pnl) in enumerate(results):
        db.record_trade(pair="EURUSD", direction="buy" if won else "sell",
                        lots=0.1, entry=1.1000 + i / 10000, sl=1.0950, tp=1.1100,
                        strategy="london_breakout", session="London", confluence=8,
                        conviction=80, signal_hash=f"seed{i}", mode="demo")
        rows = db.closed_trades(limit=1, mode=None)
        db.close_trade(db.open_trades("demo")[0]["id"] if db.open_trades("demo")
                       else rows[0]["id"], 1.1050, 50.0, pnl)
    return temp_db
