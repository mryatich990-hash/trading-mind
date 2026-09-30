"""Unified database layer: schema bootstrap, state, trades and audit helpers.

Uses SQLAlchemy for ORM access and executes config/schema.sql on init so
PostgreSQL and SQLite both get the full schema idempotently.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import settings as _settings  # noqa: F401
from config.settings import DATABASE_URL, SECRET_KEY  # noqa: F401
from core.logging_utils import get_logger

logger = get_logger(__name__)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)

_WRITE_LOCK = threading.RLock()
_INITIALIZED = False
_LAST_TRADE_COUNTS = {"today": 0, "all_time": 0}


def _utcnow() -> datetime:
    """UTC now."""
    return datetime.now(timezone.utc)


def init_db() -> None:
    """Apply config/schema.sql (idempotent) and mark initialized."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    schema_path = os.path.join(os.path.dirname(__file__), "..", "config", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as fh:
        statements = [s.strip() for s in fh.read().split(";") if s.strip()]
    # AUTOCOMMIT: each statement stands alone. In a single transaction one
    # failure would abort the rest on Postgres (InFailedSqlTransaction cascade).
    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        for stmt in statements:
            try:
                conn.execute(text(pg_compatible(stmt)))
            except Exception as exc:  # pragma: no cover - schema drift safety
                logger.warning("schema stmt skipped: %s (%s)", exc, stmt[:60])
    # Idempotent column migration for databases created before strategy
    # versioning existed (ALTER fails silently when the column is present).
    with engine.begin() as conn:
        try:
            conn.execute(text(pg_compatible(
                "ALTER TABLE trades ADD COLUMN strategy_version VARCHAR(16) DEFAULT ''"
            )))
            logger.info("migration applied: trades.strategy_version")
        except Exception:
            pass
    _INITIALIZED = True
    logger.info("database schema applied (%d statements)", len(statements))


def _session():
    """Session factory with lazy schema init."""
    global _INITIALIZED
    if not _INITIALIZED:
        init_db()
    return SessionLocal()


def _fix(sql: str) -> str:
    """Normalize named params for SQLite (:name works on both engines)."""
    return sql


def pg_compatible(ddl: str) -> str:
    """Rewrite SQLite-only DDL fragments so the same statement runs on
    PostgreSQL (used by aux modules that CREATE TABLE inline)."""
    if engine.dialect.name == "postgresql":
        ddl = ddl.replace(
            "INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY"
        )
    return ddl


# ---- system state ----


def set_state(key: str, value: str) -> None:
    """Upsert a state value (thread-safe)."""
    now = _utcnow().isoformat()
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix(
                "INSERT INTO system_state (key, value, updated_at) VALUES (:k, :v, :t) "
                "ON CONFLICT(key) DO UPDATE SET value = :v, updated_at = :t"
                if engine.dialect.name == "postgresql"
                else "INSERT INTO system_state (key, value, updated_at) VALUES (:k, :v, :t) "
                     "ON CONFLICT(key) DO UPDATE SET value = :v, updated_at = :t"
            )),
            {"k": key, "v": value, "t": now},
        )
        s.commit()


def get_state(key: str, default: str = "") -> str:
    """Read a state value."""
    with _session() as s:
        row = s.execute(
            text(_fix("SELECT value FROM system_state WHERE key = :k")), {"k": key}
        ).first()
        return row[0] if row else default


# ---- audit / logging ----


def audit(category: str, action: str, detail: str = "", source: str = "system") -> None:
    """Append to the immutable audit log."""
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix(
                "INSERT INTO audit_log (category, action, detail, source, created_at) "
                "VALUES (:c, :a, :d, :s, :t)"
            )),
            {"c": category, "a": action, "d": detail[:4000], "s": source, "t": _utcnow()},
        )
        s.commit()


def log_startup_check(name: str, passed: bool, detail: str = "") -> None:
    """Persist a startup checklist result."""
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix(
                "INSERT INTO startup_checks (check_name, passed, detail, started_at) "
                "VALUES (:n, :p, :d, :t)"
            )),
            {"n": name, "p": passed, "d": detail[:255], "t": _utcnow()},
        )
        s.commit()


def log_breaker(breaker: str, reason: str, severity: str = "halt") -> None:
    """Record a circuit-breaker trigger."""
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix(
                "INSERT INTO circuit_breakers (breaker, reason, severity, triggered_at) "
                "VALUES (:b, :r, :s, :t)"
            )),
            {"b": breaker, "r": reason[:2000], "s": severity, "t": _utcnow()},
        )
        s.commit()
    logger.warning("CIRCUIT BREAKER [%s/%s]: %s", breaker, severity, reason)


def resolve_breakers() -> None:
    """Mark all unresolved breakers resolved (auto-resume path)."""
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix(
                "UPDATE circuit_breakers SET resolved = TRUE, resolved_at = :t "
                "WHERE resolved = FALSE"
            )),
            {"t": _utcnow()},
        )
        s.commit()


def _temp_breaker_filter(breakers: list[str]) -> list[str]:
    """TEMP window: suppress every breaker NOT on the capital-protection
    allowlist. Auto-expires via settings.temp_breaker_allowlist() (empty list
    once the window ends -> nothing suppressed, steady state restores)."""
    allow = _settings.temp_breaker_allowlist()
    if not allow:
        return breakers
    kept = [b for b in breakers if b in allow]
    dropped = [b for b in breakers if b not in allow]
    if dropped:
        logger.warning("TEMP WINDOW: breakers suppressed: %s (allowlist: %s)",
                       dropped, allow)
    return kept


def unresolved_breakers(severity: str = "halt") -> list[str]:
    """Names of active (unresolved) breakers, default: halt-severity only.

    severity="observe" returns observation-severity breakers (e.g. a soft
    data_stale: no new trades, but existing positions keep being managed).
    During the operator TEMP window, breakers outside the capital-protection
    allowlist are suppressed (daily_loss / drawdown_halt / margin_halt stay).
    """
    if severity not in ("halt", "observe", "all"):
        severity = "halt"
    with _session() as s:
        if severity == "all":
            sql = ("SELECT DISTINCT breaker FROM circuit_breakers "
                   "WHERE resolved = FALSE")
            params = {}
        else:
            sql = ("SELECT DISTINCT breaker FROM circuit_breakers "
                   "WHERE resolved = FALSE AND severity = :sev")
            params = {"sev": severity}
        rows = s.execute(text(_fix(sql)), params).all()
        return _temp_breaker_filter([r[0] for r in rows])


def log_feed_health(component: str, healthy: bool, detail: str = "") -> None:
    """Record a component health probe."""
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix(
                "INSERT INTO feed_health (component, healthy, detail, checked_at) "
                "VALUES (:c, :h, :d, :t)"
            )),
            {"c": component, "h": healthy, "d": detail[:255], "t": _utcnow()},
        )
        s.commit()


# ---- trades ----


def record_trade(
    pair: str,
    direction: str,
    lots: float,
    entry: float,
    sl: float,
    tp: float,
    strategy: str = "",
    session: str = "",
    confluence: int = 0,
    conviction: int = 0,
    reasoning: str = "",
    signal_hash: str = "",
    mode: str = "live",
    tp2: float = 0.0,
    tp3: float = 0.0,
    features: Optional[dict] = None,
) -> int:
    """Insert a trade row; returns id."""
    with _session() as s, _WRITE_LOCK:
        result = s.execute(
            text(_fix(
                "INSERT INTO trades (pair, direction, lots, entry_price, sl, tp, tp2, tp3, "
                "opened_at, status, strategy, session, confluence_score, groq_conviction, "
                "groq_reasoning, signal_hash, mode, features_json, created_at) "
                "VALUES (:pair, :dir, :lots, :entry, :sl, :tp, :tp2, :tp3, :t, 'open', "
                ":strategy, :session, :conf, :conv, :reason, :hash, :mode, :features, :t)"
            )),
            {
                "pair": pair, "dir": direction, "lots": lots, "entry": entry, "sl": sl,
                "tp": tp, "tp2": tp2, "tp3": tp3, "t": _utcnow(), "strategy": strategy,
                "session": session, "conf": confluence, "conv": conviction,
                "reason": reasoning[:4000], "hash": signal_hash, "mode": mode,
                "features": json.dumps(features or {}),
            },
        )
        s.commit()
        row = s.execute(
            text(_fix("SELECT id FROM trades WHERE signal_hash = :h ORDER BY id DESC LIMIT 1")),
            {"h": signal_hash},
        ).first()
        return int(row[0])


def close_trade(trade_id: int, exit_price: float, pips: float, pnl_usd: float,
                rr: float = 0.0, swap_cost: float = 0.0) -> None:
    """Close a trade with outcome fields."""
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix(
                "UPDATE trades SET status='closed', exit_price=:x, closed_at=:t, "
                "pips=:p, pnl_usd=:pnl, rr_achieved=:rr, swap_cost_usd=:swap WHERE id=:id"
            )),
            {"x": exit_price, "t": _utcnow(), "p": pips, "pnl": round(pnl_usd, 2),
             "rr": rr, "swap": swap_cost, "id": trade_id},
        )
        s.commit()


def open_trades(mode: Optional[str] = None) -> list[dict]:
    """All open trades as dicts."""
    sql = "SELECT id, pair, direction, lots, entry_price, sl, tp, strategy, session, opened_at, mode FROM trades WHERE status='open'"
    params: dict[str, Any] = {}
    if mode:
        sql += " AND mode = :m"
        params["m"] = mode
    with _session() as s:
        rows = s.execute(text(_fix(sql)), params).mappings().all()
        return [dict(r) for r in rows]


def closed_trades(limit: int = 100, mode: Optional[str] = None) -> list[dict]:
    """Recent closed trades, newest first."""
    sql = ("SELECT * FROM trades WHERE status='closed'"
           + (" AND mode = :m" if mode else "")
           + " ORDER BY closed_at DESC LIMIT :n")
    params: dict[str, Any] = {"n": limit}
    if mode:
        params["m"] = mode
    with _session() as s:
        rows = s.execute(text(_fix(sql)), params).mappings().all()
        return [dict(r) for r in rows]


def duplicate_signal_recently(signal_hash: str, minutes: int = 5) -> bool:
    """True when the same signal hash was executed within N minutes."""
    cutoff = _utcnow().timestamp() - minutes * 60
    with _session() as s:
        row = s.execute(
            text(_fix(
                "SELECT id FROM trades WHERE signal_hash = :h AND created_at > :c LIMIT 1"
            )),
            {"h": signal_hash, "c": datetime.fromtimestamp(cutoff, tz=timezone.utc)},
        ).first()
        return row is not None


def count_open_trades(mode: Optional[str] = None) -> int:
    """Number of open trades."""
    sql = "SELECT COUNT(*) FROM trades WHERE status='open'"
    params: dict[str, Any] = {}
    if mode:
        sql += " AND mode = :m"
        params["m"] = mode
    with _session() as s:
        return int(s.execute(text(_fix(sql)), params).scalar_one())


def trades_today(mode: Optional[str] = None) -> list[dict]:
    """Today's trades (UTC day)."""
    day = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    sql = ("SELECT * FROM trades WHERE created_at >= :d"
           + (" AND mode = :m" if mode else ""))
    params: dict[str, Any] = {"d": day}
    if mode:
        params["m"] = mode
    with _session() as s:
        rows = s.execute(text(_fix(sql)), params).mappings().all()
        return [dict(r) for r in rows]


def daily_pnl(mode: Optional[str] = None) -> float:
    """Sum of today's closed PnL."""
    rows = trades_today(mode)
    return round(sum(float(r["pnl_usd"] or 0) for r in rows if r["status"] == "closed"), 2)


def trade_counts(mode: Optional[str] = None) -> dict:
    """Trade counters for notifications: {'today': N, 'all_time': M}.

    Fail-safe: on any DB error returns the last known values (zeros on the
    first failure) so a counting hiccup never breaks an alert.
    """
    global _LAST_TRADE_COUNTS
    try:
        day = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        sql = "SELECT COUNT(*) AS n, CASE WHEN created_at >= :d THEN 1 ELSE 0 END AS is_today FROM trades"
        params: dict[str, Any] = {"d": day}
        if mode:
            sql += " WHERE mode = :m"
            params["m"] = mode
        with _session() as s:
            rows = s.execute(text(_fix(sql)), params).mappings().all()
        counts = {"today": 0, "all_time": 0}
        for row in rows:
            n = int(row["n"] or 0)
            counts["all_time"] += n
            if int(row["is_today"] or 0):
                counts["today"] += n
        _LAST_TRADE_COUNTS = counts
        return counts
    except Exception:
        logger.exception("trade_counts query failed (using last known values)")
        return dict(_LAST_TRADE_COUNTS)


def record_research(pair: str, result: str, reason: str, confluence: int = 0,
                    conviction: int = 0, research: Optional[dict] = None) -> None:
    """Log one research cycle."""
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix(
                "INSERT INTO research_cycles (pair, result, reason, confluence_score, "
                "conviction, research_json, created_at) VALUES (:p, :r, :reason, :c, :v, :j, :t)"
            )),
            {"p": pair, "r": result, "reason": reason[:255], "c": confluence,
             "v": conviction, "j": json.dumps(research or {})[:8000], "t": _utcnow()},
        )
        s.commit()


def has_any_trades() -> bool:
    """True once at least one trade row exists (any mode/status).

    Expiry condition for the first-trade pilot. Fail-safe: a DB error
    returns True so the pilot never widens gates when the database is
    unreachable.
    """
    try:
        with _session() as s:
            row = s.execute(text(_fix("SELECT 1 FROM trades LIMIT 1"))).first()
            return row is not None
    except Exception:
        logger.exception("has_any_trades query failed (failing closed)")
        return True


def recent_research(limit: int = 5) -> list[dict]:
    """Last N research cycles."""
    with _session() as s:
        rows = s.execute(
            text(_fix(
                "SELECT pair, result, reason, confluence_score, conviction, created_at "
                "FROM research_cycles ORDER BY id DESC LIMIT :n"
            )),
            {"n": limit},
        ).mappings().all()
        return [dict(r) for r in rows]


def log_selection(pair: str, scores: dict, selected: str, reason: str) -> None:
    """Persist per-candle strategy selection."""
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix(
                "INSERT INTO strategy_selections (pair, scores, selected, reason, created_at) "
                "VALUES (:p, :sc, :se, :r, :t)"
            )),
            {"p": pair, "sc": json.dumps(scores), "se": selected, "r": reason[:255],
             "t": _utcnow()},
        )
        s.commit()


def strategy_weight(strategy: str) -> float:
    """Current selector weight for a strategy (default 1.0)."""
    with _session() as s:
        row = s.execute(
            text(_fix("SELECT weight FROM strategy_weights WHERE strategy = :s")),
            {"s": strategy},
        ).first()
        return float(row[0]) if row else 1.0


def strategy_enabled(strategy: str) -> bool:
    """True when the strategy is enabled."""
    with _session() as s:
        row = s.execute(
            text(_fix("SELECT enabled FROM strategy_weights WHERE strategy = :s")),
            {"s": strategy},
        ).first()
        return bool(row[0]) if row else True


def set_strategy_weight(strategy: str, weight: float, enabled: bool = True,
                        reason: str = "") -> None:
    """Upsert a strategy weight."""
    with _session() as s, _WRITE_LOCK:
        s.execute(
            text(_fix("DELETE FROM strategy_weights WHERE strategy = :s")), {"s": strategy}
        )
        s.execute(
            text(_fix(
                "INSERT INTO strategy_weights (strategy, weight, enabled, updated_at, reason) "
                "VALUES (:s, :w, :e, :t, :r)"
            )),
            {"s": strategy, "w": weight, "e": enabled, "t": _utcnow(), "r": reason[:255]},
        )
        s.commit()
        audit("strategy", "weight_change", f"{strategy} -> {weight:.2f}: {reason}")
