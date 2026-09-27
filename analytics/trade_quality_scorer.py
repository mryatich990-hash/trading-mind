"""Trade quality scorer (UPGRADE 6): every trade scored 1-10.

Components: entry timing (immediate follow-through vs drawdown first),
confluence quality, exit capture (fraction of the achieved move captured vs
the peak), and realized RR vs planned. Low scores are flagged for journal
review; the rolling average is a performance metric on the dashboard.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TradeQualityScorer:
    """Scores closed trades 1-10 from their recorded outcomes."""

    def score(self, *, won: bool, rr_achieved: float = 0.0, rr_planned: float = 2.0,
              confluence: int = 3, immediate_move: bool = True,
              captured_fraction: float = 0.8, adverse_first_pips: float = 0.0) -> dict:
        """Composite quality score with component breakdown."""
        rr_planned = max(rr_planned, 0.5)
        # 1. entry timing (0-3): immediate move good, deep adverse first bad
        if immediate_move and adverse_first_pips <= 2:
            entry = 3.0
        elif adverse_first_pips <= 5:
            entry = 2.0
        elif adverse_first_pips <= 10:
            entry = 1.0
        else:
            entry = 0.0
        # 2. confluence (0-2.5): 6+ excellent, 3 minimum
        conf = min(max(confluence - 2, 0), 5) / 2.0
        # 3. exit capture (0-2.5)
        exit_c = min(max(captured_fraction, 0.0), 1.0) * 2.5
        # 4. RR achieved vs planned (0-2)
        rr_score = min(rr_achieved / rr_planned, 1.0) * 2.0
        if not won:
            rr_score *= 0.3
        total = entry + conf + exit_c + rr_score
        total = round(min(max(total, 1.0), 10.0), 1)
        return {"score": total,
                "components": {"entry": round(entry, 1), "confluence": round(conf, 1),
                               "exit": round(exit_c, 1), "rr": round(rr_score, 1)},
                "flagged": total < 4.0}

    def score_trade_row(self, row: dict) -> dict:
        """Score from a trades-table row (missing fields degrade gracefully)."""
        pnl = float(row.get("pnl_usd") or 0.0)
        pips = float(row.get("pips") or 0.0)
        rr_achieved = float(row.get("rr_achieved") or 0.0)
        return self.score(
            won=pnl > 0, rr_achieved=rr_achieved,
            confluence=int(row.get("confluence_score") or 3),
            immediate_move=pips >= 0 or pnl > 0,
            captured_fraction=min(max(rr_achieved / 2.0, 0.0), 1.0))

    def average(self, limit: int = 100) -> dict:
        """Average quality over recent closed trades (derived approximation)."""
        from analytics.mae_mfe_analyzer import MAEMFEAnalyzer

        rows = MAEMFEAnalyzer().fetch_excursions(limit)
        if not rows:
            return {"average": None, "samples": 0, "flagged": 0}
        scores = []
        for r in rows:
            s = self.score(won=r["won"], rr_achieved=r["mfe"] / 25.0 if r["mfe"] else 0.0,
                           immediate_move=r["mae"] < 3.0,
                           captured_fraction=min(r["mfe"] / max(r["mfe"] + r["mae"], 1e-9), 1.0),
                           adverse_first_pips=r["mae"])
            scores.append(s["score"])
        flagged = sum(1 for s in scores if s < 4.0)
        return {"average": round(sum(scores) / len(scores), 2), "samples": len(scores),
                "flagged": flagged}

    def status(self) -> dict:
        """Dashboard payload."""
        return self.average()
