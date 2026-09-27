"""Advanced analytics (UPGRADE 6): rolling performance ratios.

Sharpe, Sortino, Calmar, profit factor, expectancy and recovery factor over
30/90/365-day windows, computed from closed trades in the database.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from config import settings

WINDOWS = (30, 90, 365)
RISK_FREE_ANNUAL = 0.04  # for Sharpe excess return


def compute_ratios(pnls: list[float], balances: Optional[list[float]] = None) -> dict:
    """All ratios for one PnL series (chronological)."""
    if not pnls:
        return {"sharpe": None, "sortino": None, "calmar": None, "profit_factor": None,
                "expectancy": None, "recovery": None, "trades": 0}
    arr = np.asarray(pnls, dtype=float)
    n = len(arr)
    mean, std = float(arr.mean()), float(arr.std(ddof=1)) if n > 1 else 0.0
    downside = arr[arr < 0]
    dstd = float(np.sqrt((downside ** 2).mean())) if len(downside) else 0.0

    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max()) if n else 0.0
    total = float(arr.sum())

    daily_equiv = math.sqrt(252.0)
    sharpe = ((mean - RISK_FREE_ANNUAL / 252.0) / std * daily_equiv) if std > 0 else None
    sortino = ((mean - RISK_FREE_ANNUAL / 252.0) / dstd * daily_equiv) if dstd > 0 else None
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None
    win_rate = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    recovery = total / max_dd if max_dd > 0 else None
    # annualized return / max dd (approximate from average trade)
    ann_ret = mean * 252.0
    calmar = ann_ret / max_dd if max_dd > 0 else None
    return {"sharpe": _r(sharpe), "sortino": _r(sortino), "calmar": _r(calmar),
            "profit_factor": _r(pf), "expectancy": _r(expectancy),
            "recovery": _r(recovery), "max_dd": _r(max_dd), "trades": n}


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 3)


class AdvancedAnalytics:
    """Rolling analytics from the trades table."""

    def ratios_by_window(self, limit: int = 1000) -> dict:
        """Ratios per rolling window (30/90/365 days)."""
        out = {}
        for days in WINDOWS:
            pnls = self._closed_pnls(days, limit)
            out[f"{days}d"] = compute_ratios(pnls)
        return out

    @staticmethod
    def _closed_pnls(days: int, limit: int) -> list[float]:
        """Chronological closed-trade PnLs within the window."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine

            with engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT pnl_usd FROM trades WHERE status = 'closed' "
                    "AND closed_at >= CURRENT_TIMESTAMP - (:days || ' days')::interval "
                    "ORDER BY closed_at ASC LIMIT :l"),
                    {"days": days, "l": limit}).all()
            return [float(r[0] or 0.0) for r in rows]
        except Exception:
            # SQLite fallback (no interval cast)
            try:
                import datetime as dt

                from sqlalchemy import text as sqltext

                from core.db import engine

                cutoff = (dt.datetime.now(dt.timezone.utc)
                          - dt.timedelta(days=days)).isoformat()
                with engine.begin() as conn:
                    rows = conn.execute(sqltext(
                        "SELECT pnl_usd FROM trades WHERE status = 'closed' "
                        "AND closed_at >= :c ORDER BY closed_at ASC LIMIT :l"),
                        {"c": cutoff, "l": limit}).all()
                return [float(r[0] or 0.0) for r in rows]
            except Exception as exc:
                from core.logging_utils import get_logger

                get_logger(__name__).warning("analytics query failed: %s", exc)
                return []

    def status(self) -> dict:
        """Dashboard payload."""
        return {"windows": self.ratios_by_window()}
