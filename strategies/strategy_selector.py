"""StrategySelector: scores all 15 strategies per M15 close, picks the best.

Score = session 25 + trend 25 + volatility 20 + recent win rate 15 +
confluence richness 15, multiplied by the strategy's persisted weight.
Below threshold -> no trade, logged. Ties -> higher score wins.
"""

import threading
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import settings
from core import db
from core.logging_utils import get_logger
from strategies.base_strategy import BaseStrategy, MarketContext, StrategySignal

logger = get_logger(__name__)

__all__ = ["SelectionResult", "StrategySelector", "session_of"]

MAX_SESSION, MAX_TREND, MAX_VOL, MAX_PERF, MAX_CONF = 25.0, 25.0, 20.0, 15.0, 15.0


def session_of(ts: pd.Timestamp) -> str:
    """Session for a UTC timestamp."""
    h = ts.hour + ts.minute / 60.0
    if 12 <= h < 14:
        return "Overlap"
    if 7 <= h < 12:
        return "London"
    if 12 <= h < 17:
        return "NewYork"
    if 0 <= h < 7:
        return "Asian"
    return "Dead"


@dataclass
class SelectionResult:
    """One selection cycle outcome."""

    winner: Optional[StrategySignal]
    scores: dict[str, float]
    reason: str


class StrategySelector:
    """Scores all strategies using weights and recent win rates."""

    def __init__(self, strategies: Optional[list[BaseStrategy]] = None,
                 min_score: int = 0) -> None:
        self.strategies = strategies or self._default_strategies()
        self.min_score = min_score or settings.SELECTOR_MIN_SCORE
        self._lock = threading.Lock()

    @staticmethod
    def _default_strategies() -> list[BaseStrategy]:
        """All 15 strategies."""
        from strategies.library import all_strategies

        return all_strategies()

    # ---- scoring ----

    @staticmethod
    def _strategy_win_rate(strategy: BaseStrategy, window: int = 15) -> float:
        """Win rate over the strategy's last N closed signals (50 when unknown)."""
        from sqlalchemy import text as sqltext

        from core.db import engine
        with engine.begin() as conn:
            rows = conn.execute(sqltext(
                "SELECT pnl_usd FROM trades WHERE strategy = :s AND status = 'closed' "
                "ORDER BY closed_at DESC LIMIT :n"
            ), {"s": strategy.name, "n": window}).all()
        if not rows:
            return 50.0
        wins = sum(1 for r in rows if float(r[0]) > 0)
        return wins / len(rows) * 100.0

    def score(self, strategy: BaseStrategy, ctx: MarketContext,
              signal: Optional[StrategySignal]) -> float:
        """Compute the 0-100 score for one strategy."""
        session_pts = MAX_SESSION if strategy.session_match(ctx) else 0.0
        trend = ctx.h4_bias() if ctx.h4_bias() != "ranging" else ctx.h1_trend()
        if signal is not None:
            want = "buy" if trend == "bullish" else "sell" if trend == "bearish" else ""
            trend_pts = MAX_TREND if (want and signal.direction == want) or not want else 0.0
            conf_pts = min(len(signal.confluences), 5) / 5.0 * MAX_CONF
        else:
            trend_pts = MAX_TREND / 2 if trend in ("bullish", "bearish") else 0.0
            conf_pts = MAX_CONF / 2
        atr_pips = ctx.atr_m15() / self._pip(strategy, ctx)
        lo, hi = strategy.atr_range_pips
        if lo <= atr_pips <= hi:
            vol_pts = MAX_VOL
        elif lo * 0.7 <= atr_pips <= hi * 1.3:
            vol_pts = MAX_VOL / 2
        else:
            vol_pts = 0.0
        perf_pts = MAX_PERF * self._strategy_win_rate(strategy) / 100.0
        weight = db.strategy_weight(strategy.name)
        return round((session_pts + trend_pts + vol_pts + perf_pts + conf_pts) * weight, 1)

    @staticmethod
    def _pip(strategy: BaseStrategy, ctx: MarketContext) -> float:
        """Pip size for the context pair."""
        return pip_size(ctx.pair) if ctx.pair else 0.0001

    # ---- main ----

    def run(self, ctx: MarketContext) -> SelectionResult:
        """Evaluate every strategy and select the best above threshold."""
        with self._lock:
            scores: dict[str, float] = {}
            signals: dict[str, StrategySignal] = {}
            for strategy in self.strategies:
                if not db.strategy_enabled(strategy.name):
                    scores[strategy.name] = 0.0
                    continue
                if not strategy.session_match(ctx):
                    scores[strategy.name] = 0.0
                    continue
                try:
                    signal = strategy.evaluate(ctx)
                except Exception as exc:
                    logger.exception("strategy %s failed: %s", strategy.name, exc)
                    signal = None
                signals[strategy.name] = signal
                scores[strategy.name] = self.score(strategy, ctx, signal)

            qualified = [(n, s) for n, s in signals.items() if s and scores[n] >= self.min_score]
            if not qualified:
                best = max(scores, key=scores.get) if scores else ""
                reason = f"no setup above {self.min_score} (best {best}={scores.get(best, 0):.0f})"
                db.log_selection(ctx.pair, scores, "", reason)
                return SelectionResult(None, scores, reason)

            qualified.sort(key=lambda kv: scores[kv[0]], reverse=True)
            name, sig = qualified[0]
            sig.confluences.append(f"selector_score_{scores[name]:.0f}")
            reason = f"{name} selected ({scores[name]:.0f})"
            db.log_selection(ctx.pair, scores, name, reason)
            logger.info("selection %s: %s", ctx.pair, reason)
            return SelectionResult(sig, scores, reason)


def pip_size(pair: str) -> float:
    """Re-exported pip size."""
    from strategies.base_strategy import pip_size as _pip

    return _pip(pair)
