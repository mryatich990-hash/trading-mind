"""RiskManager: the final pre-trade gate (Research Step 10) and sizing authority.

Chain: Kelly fraction -> drawdown tier -> VIX/news/central-bank multipliers ->
daily rules (max trades, max daily loss, loss-streak pauses) -> margin and
correlation checks -> position size in lots. Every rejection is logged with a
reason so the dashboard/Telegram can show exactly why no trade happened.
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from config import settings
from core import db
from core.logging_utils import get_logger
from risk.correlation_filter import CorrelationFilter, correlation_of
from risk.drawdown_manager import DrawdownManager
from risk.kelly_criterion import KellyCriterion, KellyResult

logger = get_logger(__name__)

__all__ = ["SizingDecision", "RiskManager", "lots_for_risk", "signal_hash"]

# pip value in USD per 1.0 standard lot (approximations; USDJPY is dynamic)
PIP_VALUE_PER_LOT = {"EURUSD": 10.0, "GBPUSD": 10.0, "USDJPY": 6.8, "XAUUSD": 10.0,
                     "NAS100": 1.0, "US30": 1.0}


def lots_for_risk(balance: float, risk_pct: float, sl_pips: float, pair: str,
                  usdjpy: float = 0.0) -> Decimal:
    """Position lots so that SL hit loses exactly risk_pct of balance (Decimal math)."""
    if sl_pips <= 0 or balance <= 0:
        return Decimal("0.01")
    pip_value = PIP_VALUE_PER_LOT.get(pair.upper(), 10.0)
    if pair.upper() == "USDJPY" and usdjpy > 0:
        pip_value = 1000.0 / usdjpy
    risk_amount = Decimal(str(balance)) * Decimal(str(risk_pct)) / Decimal("100")
    raw = risk_amount / (Decimal(str(sl_pips)) * Decimal(str(pip_value)))
    lots = raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return max(lots, Decimal("0.01"))


def signal_hash(pair: str, direction: str, strategy: str, entry: float) -> str:
    """Stable hash for duplicate-order protection."""
    blob = f"{pair}|{direction}|{strategy}|{round(entry, 5)}".encode()
    return hashlib.sha256(blob).hexdigest()[:32]


@dataclass
class SizingDecision:
    """Outcome of the final risk gate."""

    approved: bool
    reason: str = ""
    lots: Decimal = Decimal("0.00")
    risk_pct: float = 0.0
    kelly: Optional[KellyResult] = None
    checks: dict = field(default_factory=dict)


class RiskManager:
    """Applies every hard rule before an order may be sent."""

    def __init__(self, kelly: Optional[KellyCriterion] = None,
                 drawdown: Optional[DrawdownManager] = None,
                 correlation: Optional[CorrelationFilter] = None) -> None:
        self.kelly = kelly or KellyCriterion()
        self.drawdown = drawdown or DrawdownManager()
        self.correlation = correlation or CorrelationFilter()
        self._lock = threading.Lock()
        self._last_streak_check = 0

    # ---- helpers ----

    @staticmethod
    def consecutive_losses(mode: Optional[str] = None) -> int:
        """Current losing streak from closed trades (newest first)."""
        rows = db.closed_trades(limit=20, mode=mode)
        streak = 0
        for r in rows:
            if float(r.get("pnl_usd") or 0) < 0:
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def margin_level(equity: float, used_margin: float) -> float:
        """Margin level percent; 1000 when nothing is committed."""
        if used_margin <= 0:
            return 1000.0
        return round(equity / used_margin * 100.0, 1)

    # ---- streak pause state ----

    def _streak_pause_active(self) -> bool:
        """2h pause after 3 consecutive losses; day stop after 5."""
        streak = self.consecutive_losses()
        if streak >= 5:
            return True
        if streak >= 3:
            until = db.get_state("loss_streak_pause_until", "")
            if not until:
                import datetime as _dt
                db.set_state("loss_streak_pause_until",
                             (db._utcnow() + _dt.timedelta(hours=2)).isoformat())
            return True
        return False

    # ---- main gate ----

    def evaluate(self, pair: str, direction: str, strategy: str, entry: float,
                 sl: float, tp: float, balance: float, equity: float,
                 used_margin: float = 0.0, usdjpy: float = 0.0,
                 open_positions: Optional[list[dict]] = None,
                 vix: float = 0.0, spread_pips: float = 0.0,
                 confluence: int = 8, mode: Optional[str] = None,
                 size_multipliers: Optional[dict] = None) -> SizingDecision:
        """Run the full gate; returns approval with exact lots or a rejection reason."""
        with self._lock:
            checks: dict = {}
            open_positions = open_positions or []
            size_multipliers = size_multipliers or {}

            # 1. hard breakers (query unresolved halt breakers)
            active = db.unresolved_breakers()
            checks["breakers"] = active
            if active:
                return SizingDecision(False, f"halted by breakers: {', '.join(active)}",
                                      checks=checks)

            # 2. daily loss limit
            day_pnl = db.daily_pnl(mode)
            day_loss_pct = abs(min(day_pnl, 0.0)) / balance * 100.0 if balance > 0 else 0.0
            checks["daily_loss_pct"] = round(day_loss_pct, 2)
            if day_loss_pct >= settings.MAX_DAILY_LOSS_PCT:
                db.log_breaker("daily_loss", f"-{day_loss_pct:.1f}% today", "halt")
                return SizingDecision(False, f"daily loss limit {day_loss_pct:.1f}%",
                                      checks=checks)

            # 3. trade count rules
            n_today = len(db.trades_today(mode))
            checks["trades_today"] = n_today
            if n_today >= settings.MAX_TRADES_PER_DAY:
                return SizingDecision(False, "max daily trades reached", checks=checks)
            if db.count_open_trades(mode) >= settings.MAX_OPEN_TRADES:
                return SizingDecision(False, "max open trades reached", checks=checks)
            if self._streak_pause_active():
                return SizingDecision(False, "loss-streak pause active", checks=checks)

            # 4. drawdown tier
            dd = self.drawdown.update(equity)
            checks["drawdown_pct"] = dd.drawdown_pct
            checks["tier"] = dd.tier
            if dd.halted:
                return SizingDecision(False, f"drawdown {dd.drawdown_pct:.1f}% halt",
                                      checks=checks)

            # 5. confluence requirement from drawdown tier
            if confluence < dd.min_confluence:
                return SizingDecision(
                    False, f"confluence {confluence} < required {dd.min_confluence}",
                    checks=checks)

            # 6. total open risk
            open_risk = len(open_positions) * settings.RISK_PER_TRADE_PCT
            if open_risk + settings.RISK_PER_TRADE_PCT > settings.MAX_TOTAL_OPEN_RISK_PCT:
                return SizingDecision(False, "total open risk would exceed cap",
                                      checks=checks)

            # 7. correlation rules
            ok, why = self.correlation.can_open(pair, direction, open_positions)
            if not ok:
                return SizingDecision(False, why, checks=checks)
            new_lots_preview = lots_for_risk(balance, settings.RISK_PER_TRADE_PCT,
                                             abs(entry - sl) /
                                             _pip(pair) if sl else 20.0, pair, usdjpy)
            ok, why = CorrelationFilter.portfolio_lot_guard(
                pair, direction, float(new_lots_preview), open_positions)
            if not ok:
                return SizingDecision(False, why, checks=checks)

            # 8. margin level
            margin = self.margin_level(equity, used_margin)
            checks["margin_level"] = margin
            if margin < settings.MARGIN_HALT_PCT:
                db.log_breaker("margin", f"margin level {margin:.0f}%", "halt")
                return SizingDecision(False, f"margin level {margin:.0f}%", checks=checks)
            if margin < settings.MARGIN_REDUCE_PCT:
                return SizingDecision(False, f"margin level {margin:.0f}% < 300%",
                                      checks=checks)

            # 9. Kelly sizing with every multiplier stacked
            kres = self.kelly.compute(db.closed_trades(limit=60, mode=mode))
            risk_pct = kres.applied_pct
            risk_pct *= dd.risk_pct if dd.risk_pct > 0 else 0.0
            risk_pct *= size_multipliers.get("vix", 1.0)
            risk_pct *= size_multipliers.get("calendar", 1.0)
            risk_pct *= size_multipliers.get("central_bank", 1.0)
            risk_pct *= size_multipliers.get("anti_martingale", 1.0)
            risk_pct *= size_multipliers.get("sentiment_extreme", 1.0)
            risk_pct = min(max(risk_pct, 0.0), settings.MAX_RISK_PER_TRADE_PCT)
            if risk_pct < settings.MIN_RISK_PER_TRADE_PCT * 0.5:
                return SizingDecision(False, "effective risk below floor", checks=checks)

            sl_pips = abs(entry - sl) / _pip(pair) if entry != sl else 20.0
            lots = lots_for_risk(balance, risk_pct, sl_pips, pair, usdjpy)
            checks["risk_pct"] = round(risk_pct, 2)
            checks["sl_pips"] = round(sl_pips, 1)
            checks["lots"] = float(lots)
            logger.info("risk gate PASSED %s %s: %.2f%% risk, %.2f lots (dd tier %d)",
                        pair, direction, risk_pct, float(lots), dd.tier)
            return SizingDecision(True, "approved", lots, round(risk_pct, 2), kres, checks)

    def anti_martingale_multiplier(self) -> float:
        """+10% after a win (max 3 stacked), -15% after a loss, reset every 5 trades."""
        rows = db.closed_trades(limit=5)
        if not rows:
            return 1.0
        wins = sum(1 for r in rows if float(r["pnl_usd"] or 0) > 0)
        losses = len(rows) - wins
        recent = rows[0]
        last_won = float(recent["pnl_usd"] or 0) > 0
        if len(rows) >= 5 and wins + losses == 5:
            return 1.0  # reset after every 5 trades
        streak = 0
        for r in rows:
            if (float(r["pnl_usd"] or 0) > 0) == last_won:
                streak += 1
            else:
                break
        if last_won:
            return min(1.0 + 0.10 * streak, 1.30)
        return max(1.0 - 0.15 * streak, 0.55)


def _pip(pair: str) -> float:
    """Pip size helper (avoids circular import of data engine)."""
    pair = pair.upper()
    if pair in ("USDJPY", "GBPJPY", "EURJPY"):
        return 0.01
    if pair == "XAUUSD":
        return 0.1
    if pair in ("NAS100", "US30"):
        return 1.0
    return 0.0001
