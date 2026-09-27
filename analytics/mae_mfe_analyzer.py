"""MAE/MFE analyzer (UPGRADE 6): optimal stop-loss and take-profit distances.

For every closed trade the Maximum Adverse Excursion (how far price went
AGAINST the position) and Maximum Favorable Excursion (how far IN FAVOR) are
recorded in pips. Distributions give:
- optimal SL: the distance 85% of winners never exceeded (tighten without
  cutting winners)
- optimal TP: where most winners reached before reversing
Updated every 50 closed trades; recommendations surface on the dashboard and
feed the SmartSL engine.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PERCENTILE_WINNER_MAE = 85  # 85% of winners never went further against us


class MAEMFEAnalyzer:
    """Excursion distribution builder + SL/TP recommendations."""

    def __init__(self, update_every: int = 50) -> None:
        self.update_every = update_every
        self._last_count = -1
        self._cache: Optional[dict] = None

    # ---- data ----

    def fetch_excursions(self, limit: int = 500) -> list[dict]:
        """Rows with mae_pips/mfe_pips when present, derived otherwise."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine

            cols = [r[1] for r in engine.connect().exec_driver_sql(
                "PRAGMA table_info(trades)").fetchall()] if "sqlite" in str(engine.url) else None
            has_cols = cols and "mae_pips" in cols and "mfe_pips" in cols
            with engine.begin() as conn:
                if has_cols:
                    rows = conn.execute(sqltext(
                        "SELECT pnl_usd, mae_pips, mfe_pips FROM trades "
                        "WHERE status='closed' AND mae_pips IS NOT NULL "
                        "ORDER BY closed_at DESC LIMIT :l"), {"l": limit}).all()
                    return [{"won": float(p or 0) > 0, "mae": float(m or 0),
                             "mfe": float(f or 0)} for p, m, f in rows]
                # derive from pips when excursion columns absent (approximation)
                rows = conn.execute(sqltext(
                    "SELECT pnl_usd, pips, rr_achieved FROM trades "
                    "WHERE status='closed' ORDER BY closed_at DESC LIMIT :l"),
                    {"l": limit}).all()
            out = []
            for pnl, pips, rr in rows:
                pips = float(pips or 0.0)
                won = float(pnl or 0.0) > 0
                # winner: adverse ≈ small fraction of pips; loser: full adverse
                mae = max(0.0, -pips) if not won else max(0.0, pips * 0.25)
                mfe = max(0.0, pips) if won else max(0.0, pips * 0.4)
                out.append({"won": won, "mae": mae, "mfe": mfe})
            return out
        except Exception as exc:
            logger.warning("excursion fetch failed: %s", exc)
            return []

    # ---- analysis ----

    def analyze(self, force: bool = False) -> dict:
        """Distribution stats + recommendations (cached between updates)."""
        rows = self.fetch_excursions()
        if not force and self._cache is not None and len(rows) == self._last_count:
            return self._cache
        if not rows:
            self._cache = {"samples": 0, "optimal_sl_pips": None,
                           "optimal_tp_pips": None}
            return self._cache
        winners = [r for r in rows if r["won"]]
        losers = [r for r in rows if not r["won"]]

        mae_winners = np.asarray([r["mae"] for r in winners]) if winners else np.zeros(1)
        mfe_winners = np.asarray([r["mfe"] for r in winners]) if winners else np.zeros(1)

        optimal_sl = float(np.percentile(mae_winners, PERCENTILE_WINNER_MAE)) \
            if winners else None
        # optimal TP: median peak of winners (most winners reached here)
        optimal_tp = float(np.median(mfe_winners)) if winners else None

        self._last_count = len(rows)
        self._cache = {
            "samples": len(rows),
            "winners": len(winners), "losers": len(losers),
            "mae_mean_winners": round(float(mae_winners.mean()), 1),
            "mae_mean_losers": round(float(np.mean([r["mae"] for r in losers])), 1) if losers else 0.0,
            "mfe_median_winners": round(float(np.median(mfe_winners)), 1) if winners else None,
            "mae_histogram": self._hist([r["mae"] for r in rows]),
            "mfe_histogram": self._hist([r["mfe"] for r in rows]),
            "optimal_sl_pips": round(optimal_sl, 1) if optimal_sl else None,
            "optimal_tp_pips": round(optimal_tp, 1) if optimal_tp else None,
        }
        return self._cache

    @staticmethod
    def _hist(values: list[float], bins: int = 10) -> list[dict]:
        """Histogram buckets for the dashboard chart."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or float(arr.max()) == 0:
            return []
        counts, edges = np.histogram(arr, bins=bins)
        return [{"from": round(float(edges[i]), 1), "to": round(float(edges[i + 1]), 1),
                 "count": int(c)} for i, c in enumerate(counts)]

    def should_update(self, closed_trades: int) -> bool:
        """True every `update_every` closed trades."""
        return closed_trades // self.update_every != self._last_count // self.update_every \
            or self._cache is None

    def recommendation(self) -> dict:
        """Dashboard-facing recommendation text + numbers."""
        data = self.analyze()
        if not data.get("samples"):
            return {"ready": False, "text": "Not enough closed trades yet."}
        sl, tp = data.get("optimal_sl_pips"), data.get("optimal_tp_pips")
        return {"ready": True, **data,
                "text": f"Optimal SL ≈ {sl} pips (85% of winners stayed closer); "
                        f"optimal TP ≈ {tp} pips (median winner peak)."}
