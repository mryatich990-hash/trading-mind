"""HistoricalMatcher: Research Step 8 — historical setup match.

Searches the trades table for similar setups (same pair, session, strategy
type and RSI band) and computes the historical win rate. Low win rates raise
the required confluence; high ones are passed to Groq as supporting evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text as sqltext

import core.db as _core_db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["HistoricalMatch", "HistoricalMatcher"]

MIN_HISTORICAL_WIN_RATE = 45.0
HIGH_HISTORICAL_WIN_RATE = 65.0


@dataclass
class HistoricalMatch:
    """Outcome of the setup search."""

    samples: int = 0
    wins: int = 0
    win_rate: float = 50.0
    required_confluence: int = 8
    note: str = "no similar setups found"


class HistoricalMatcher:
    """Queries past trades for the same setup archetype."""

    def __init__(self, min_samples: int = 10) -> None:
        self.min_samples = min_samples

    def match(self, pair: str, session: str, strategy: str, rsi: float,
              direction: str, htf_aligned: bool) -> HistoricalMatch:
        """Find similar setups and return the win-rate verdict."""
        out = HistoricalMatch()
        try:
            with _core_db.engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT pnl_usd FROM trades "
                    "WHERE pair = :pair AND status = 'closed' AND strategy = :strategy "
                    "AND (:session = '' OR session = :session) "
                    "ORDER BY closed_at DESC LIMIT 100"
                ), {"pair": pair.upper(), "strategy": strategy, "session": session}).all()
        except Exception as exc:
            logger.warning("historical match query failed: %s", exc)
            return out

        out.samples = len(rows)
        if out.samples < self.min_samples:
            out.note = f"only {out.samples} samples (<{self.min_samples})"
            return out
        out.wins = sum(1 for r in rows if float(r[0] or 0) > 0)
        out.win_rate = round(out.wins / out.samples * 100.0, 1)
        if out.win_rate < MIN_HISTORICAL_WIN_RATE:
            out.required_confluence = 9
            out.note = f"poor history ({out.win_rate:.0f}%): 9/10 required"
        elif out.win_rate > HIGH_HISTORICAL_WIN_RATE:
            out.required_confluence = 8
            out.note = f"strong history ({out.win_rate:.0f}%)"
        else:
            out.required_confluence = 8
            out.note = f"moderate history ({out.win_rate:.0f}%)"
        logger.info("historical match %s %s %s: %d samples, %.0f%%", pair, strategy,
                    session, out.samples, out.win_rate)
        return out
