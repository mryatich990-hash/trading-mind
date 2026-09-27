"""Strategy versioning (UPGRADE 7): never lose track of what changed.

Each strategy has a semantic version; every trade records the version that
took it. Bumping a version stores a changelog entry, and performance is
compared across versions automatically.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

TABLE = """CREATE TABLE IF NOT EXISTS strategy_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strategy VARCHAR(48), version VARCHAR(16), note TEXT)"""


class StrategyVersioning:
    """Version registry + per-version performance comparison."""

    def __init__(self) -> None:
        self._ensure_table()

    @staticmethod
    def _ensure_table() -> None:
        """Create the registry table when missing."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine, _WRITE_LOCK, pg_compatible

            with engine.begin() as conn, _WRITE_LOCK:
                conn.execute(sqltext(pg_compatible(TABLE)))
        except Exception as exc:
            logger.warning("strategy_versions table unavailable: %s", exc)

    def bump(self, strategy: str, version: str, note: str = "") -> None:
        """Record a new version for a strategy."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine, _WRITE_LOCK

            with engine.begin() as conn, _WRITE_LOCK:
                conn.execute(sqltext(
                    "INSERT INTO strategy_versions (strategy, version, note) "
                    "VALUES (:s, :v, :n)"), {"s": strategy, "v": version, "n": note[:500]})
            logger.info("strategy version: %s -> %s (%s)", strategy, version, note)
        except Exception as exc:
            logger.warning("version bump failed: %s", exc)

    def current(self, strategy: str) -> str:
        """Latest version string for a strategy (1.0.0 default)."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine

            with engine.begin() as conn:
                row = conn.execute(sqltext(
                    "SELECT version FROM strategy_versions WHERE strategy = :s "
                    "ORDER BY id DESC LIMIT 1"), {"s": strategy}).first()
                return row[0] if row else "1.0.0"
        except Exception:
            return "1.0.0"

    def all_current(self) -> dict:
        """Latest version per strategy (dashboard)."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine

            with engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT strategy, version FROM strategy_versions sv "
                    "WHERE id = (SELECT MAX(id) FROM strategy_versions "
                    "WHERE strategy = sv.strategy)")).all()
            return {r[0]: r[1] for r in rows}
        except Exception:
            return {}

    def performance_by_version(self, strategy: str) -> list[dict]:
        """Trades, win rate and PnL per version of one strategy."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine

            with engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT strategy_version, COUNT(*), "
                    "SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END), SUM(pnl_usd) "
                    "FROM trades WHERE strategy = :s AND status = 'closed' "
                    "GROUP BY strategy_version ORDER BY strategy_version"),
                    {"s": strategy}).all()
            out = []
            for version, n, wins, pnl in rows:
                out.append({"version": version or "unknown", "trades": n,
                            "win_rate": round((wins or 0) / n * 100, 1) if n else 0.0,
                            "pnl": round(float(pnl or 0.0), 2)})
            return out
        except Exception:
            return []

    def changelog(self, strategy: str = "", limit: int = 20) -> list[dict]:
        """Recent version entries."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine

            with engine.begin() as conn:
                if strategy:
                    rows = conn.execute(sqltext(
                        "SELECT created_at, strategy, version, note FROM "
                        "strategy_versions WHERE strategy = :s ORDER BY id DESC LIMIT :l"),
                        {"s": strategy, "l": limit}).mappings().all()
                else:
                    rows = conn.execute(sqltext(
                        "SELECT created_at, strategy, version, note FROM "
                        "strategy_versions ORDER BY id DESC LIMIT :l"),
                        {"l": limit}).mappings().all()
                return [dict(r) for r in rows]
        except Exception:
            return []
