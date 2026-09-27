"""Session optimizer: hour-by-day performance heatmap with automatic restriction.

Tracks win rate per (strategy, hour, weekday). After 100 trades the optimizer
restricts trading to hours above 52% win rate and blacklists hours below 40%.
The heatmap is refreshed every 25 trades and shown on the dashboard.
"""

from __future__ import annotations

import threading
from typing import Optional

from sqlalchemy import text as sqltext

from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["SessionOptimizer"]

MIN_TRADES = 100
GOOD_WIN_RATE = 52.0
BAD_WIN_RATE = 40.0
REFRESH_EVERY = 25


class SessionOptimizer:
    """Builds and enforces the session heatmap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ---- data collection ----

    def refresh(self, force: bool = False) -> bool:
        """Recompute the heatmap from recent closed trades; True when refreshed."""
        with self._lock:
            since = int(db.get_state("heatmap_trades_since", "0") or 0) + 1
            if not force and since < REFRESH_EVERY:
                db.set_state("heatmap_trades_since", str(since))
                return False
            db.set_state("heatmap_trades_since", "0")
        rows = db.closed_trades(limit=1000)
        stats: dict[tuple[str, int, int], list[int]] = {}
        for r in rows:
            opened = r.get("opened_at") or r.get("created_at")
            if not opened:
                continue
            try:
                from datetime import datetime
                ts = opened if isinstance(opened, datetime) else \
                    datetime.fromisoformat(str(opened).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            key = (str(r.get("strategy", "")), ts.hour, ts.weekday())
            stats.setdefault(key, [0, 0])
            stats[key][0] += 1
            stats[key][1] += 1 if float(r.get("pnl_usd") or 0) > 0 else 0
        with db.engine.begin() as conn, db._WRITE_LOCK:
            conn.execute(sqltext("DELETE FROM session_heatmap"))
            for (strategy, hour, dow), (trades, wins) in stats.items():
                wr = wins / trades * 100.0 if trades else 0.0
                blacklisted = trades >= 10 and wr < BAD_WIN_RATE
                conn.execute(sqltext(
                    "INSERT INTO session_heatmap (updated_at, strategy, hour, day_of_week, "
                    "trades, wins, win_rate, blacklisted) VALUES (:t, :s, :h, :d, :n, :w, :r, :b)"
                ), {"t": db._utcnow(), "s": strategy, "h": hour, "d": dow,
                    "n": trades, "w": wins, "r": round(wr, 1), "b": blacklisted})
        logger.info("session heatmap refreshed: %d cells", len(stats))
        return True

    # ---- queries ----

    def hour_allowed(self, strategy: str, hour: int, weekday: int) -> bool:
        """True when the strategy may trade in this hour (conservative default)."""
        total_trades = 0
        try:
            with db.engine.begin() as conn:
                row = conn.execute(sqltext(
                    "SELECT SUM(trades) FROM session_heatmap"
                )).first()
                total_trades = int(row[0] or 0)
                cells = conn.execute(sqltext(
                    "SELECT trades, win_rate, blacklisted FROM session_heatmap "
                    "WHERE strategy = :s AND hour = :h AND day_of_week = :d"
                ), {"s": strategy, "h": hour, "d": weekday}).first()
        except Exception as exc:
            logger.warning("heatmap query failed: %s", exc)
            return True
        if total_trades < MIN_TRADES:
            return True  # not enough data: allow everything
        if cells is None:
            return True
        trades, win_rate, blacklisted = int(cells[0]), float(cells[1]), bool(cells[2])
        if blacklisted:
            return False
        if trades >= 10 and win_rate < BAD_WIN_RATE:
            return False
        if total_trades >= MIN_TRADES and trades >= 5 and win_rate < GOOD_WIN_RATE:
            # after 100 global trades only above-52% hours stay enabled
            return True if win_rate >= GOOD_WIN_RATE else trades < 5
        return True

    def heatmap(self) -> list[dict]:
        """All heatmap cells for the dashboard."""
        try:
            with db.engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT strategy, hour, day_of_week, trades, wins, win_rate, "
                    "blacklisted FROM session_heatmap ORDER BY strategy, hour"
                )).mappings().all()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("heatmap read failed: %s", exc)
            return []
