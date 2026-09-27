"""ShadowTrader: parallel demo tracking, live-vs-shadow divergence and slippage.

Every live fill is mirrored into shadow_trades at the intended entry price.
The shadow side is closed when the live trade closes (at the same prices the
signal implied). Divergence and per-pair slippage feed the weekly report.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import text as sqltext

from core import db
from core.logging_utils import get_logger
from data.market_data_engine import pip_size

logger = get_logger(__name__)

__all__ = ["ShadowTrader"]

DIVERGENCE_ALERT_PCT = 15.0
SLIPPAGE_ALERT_PIPS = 2.0


class ShadowTrader:
    """Compares live execution quality against the parallel shadow run."""

    def __init__(self, notifier: Optional[callable] = None) -> None:
        self.notifier = notifier

    def _notify(self, text: str) -> None:
        if self.notifier:
            try:
                self.notifier(text)
            except Exception as exc:
                logger.error("shadow notify failed: %s", exc)

    # ---- shadow lifecycle ----

    def close_shadow(self, live_trade_id: int, shadow_exit: float) -> None:
        """Close the shadow trade when its live counterpart closes."""
        with db.engine.begin() as conn, db._WRITE_LOCK:
            row = conn.execute(sqltext(
                "SELECT pair, direction, intended_entry, shadow_fill FROM shadow_trades "
                "WHERE live_trade_id = :i AND shadow_status = 'open' ORDER BY id DESC LIMIT 1"
            ), {"i": live_trade_id}).first()
            if row is None:
                return
            pair, direction, intended, fill = row
            pip = pip_size(str(pair))
            pnl_pips = ((float(shadow_exit) - float(fill)) if direction == "buy"
                        else (float(fill) - float(shadow_exit))) / pip
            conn.execute(sqltext(
                "UPDATE shadow_trades SET shadow_status = 'closed', shadow_exit = :x, "
                "shadow_pnl = :p WHERE live_trade_id = :i AND shadow_status = 'open'"
            ), {"x": shadow_exit, "p": round(pnl_pips, 1), "i": live_trade_id})
        logger.info("shadow trade closed for live #%d", live_trade_id)

    # ---- analysis ----

    def daily_divergence(self) -> float:
        """Live PnL minus shadow-implied PnL for today, in pips (negative = live worse)."""
        try:
            with db.engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT s.live_trade_id, s.intended_entry, s.live_fill, s.shadow_pnl, "
                    "s.pair, s.direction FROM shadow_trades s "
                    "WHERE s.shadow_status = 'closed' "
                    "AND s.created_at >= CURRENT_DATE"
                )).all()
        except Exception as exc:
            logger.warning("divergence query failed: %s", exc)
            return 0.0
        total = 0.0
        for live_id, intended, live_fill, shadow_pnl, pair, direction in rows:
            pip = pip_size(str(pair))
            live_pips = ((float(live_fill) - float(intended)) if direction == "buy"
                         else (float(intended) - float(live_fill))) / pip
            total += live_pips - float(shadow_pnl or 0)
        return round(total, 1)

    def check_divergence(self) -> Optional[str]:
        """Alert when shadow outperforms live by more than the threshold."""
        divergence = self.daily_divergence()
        if divergence < -DIVERGENCE_ALERT_PCT:
            msg = ("⚠️ Execution quality issue detected. Live slippage exceeding "
                   f"threshold ({divergence:.1f} pips today). Consider switching broker "
                   "or reducing lot size.")
            self._notify(msg)
            db.log_feed_health("execution", False, f"divergence {divergence}")
            return msg
        if divergence > DIVERGENCE_ALERT_PCT:
            msg = (f"✅ Positive slippage detected ({divergence:+.1f} pips today). "
                   "Execution quality excellent.")
            self._notify(msg)
            return msg
        return None

    def slippage_by_pair(self, days: int = 7) -> list[dict]:
        """Average slippage in pips per pair over the window."""
        try:
            with db.engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT pair, direction, intended_entry, live_fill "
                    "FROM shadow_trades WHERE created_at >= CURRENT_DATE - :d"
                ), {"d": days}).all()
        except Exception as exc:
            logger.warning("slippage query failed: %s", exc)
            return []
        agg: dict[str, list[float]] = {}
        for pair, direction, intended, fill in rows:
            pip = pip_size(str(pair))
            slip = abs(float(fill) - float(intended)) / pip
            agg.setdefault(str(pair), []).append(slip)
        return [{"pair": p, "avg_slippage_pips": round(sum(v) / len(v), 2),
                 "samples": len(v)} for p, v in sorted(agg.items())]

    def weekly_report(self) -> str:
        """Weekly slippage report text for Telegram."""
        rows = self.slippage_by_pair(7)
        if not rows:
            return "📊 Slippage report: no trades this week."
        lines = ["📊 Weekly Slippage Report:"]
        for r in rows:
            flag = " ⚠️" if r["avg_slippage_pips"] > SLIPPAGE_ALERT_PIPS else ""
            lines.append(f"• {r['pair']}: {r['avg_slippage_pips']:.2f} pips avg "
                         f"({r['samples']} trades){flag}")
            if r["avg_slippage_pips"] > SLIPPAGE_ALERT_PIPS:
                self._notify(f"⚠️ {r['pair']} average slippage "
                             f"{r['avg_slippage_pips']:.2f} pips exceeds "
                             f"{SLIPPAGE_ALERT_PIPS} pips")
        return "\n".join(lines)
