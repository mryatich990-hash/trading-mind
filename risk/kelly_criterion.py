"""Kelly Criterion sizing: half-Kelly with hard caps from settings."""

from dataclasses import dataclass

from config import settings
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["KellyResult", "KellyCriterion"]


@dataclass
class KellyResult:
    """Kelly computation outcome."""

    win_rate: float
    avg_rr: float
    full_kelly_pct: float
    half_kelly_pct: float
    applied_pct: float
    capped: bool


class KellyCriterion:
    """Computes risk % per trade from the rolling trade history."""

    def __init__(self, window: int = 50) -> None:
        self.window = window

    def compute(self, closed_trades: list[dict]) -> KellyResult:
        """Kelly from the last N closed trades (floors at neutral values)."""
        if not closed_trades:
            return KellyResult(50.0, 1.5, 0.0, 0.0, settings.RISK_PER_TRADE_PCT, False)
        trades = closed_trades[: self.window]
        wins = [t for t in trades if float(t["pnl_usd"] or 0) > 0]
        losses = [t for t in trades if float(t["pnl_usd"] or 0) <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.5
        avg_win = sum(float(t["pnl_usd"]) for t in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(float(t["pnl_usd"]) for t in losses) / len(losses)) if losses else 1.0
        avg_rr = (avg_win / avg_loss) if avg_loss > 0 else 1.5
        loss_rate = 1.0 - win_rate
        full_kelly = (win_rate * avg_rr - loss_rate) / avg_rr if avg_rr > 0 else 0.0
        full_kelly_pct = max(0.0, full_kelly * 100.0)
        half_kelly_pct = full_kelly_pct / 2.0
        applied = min(max(half_kelly_pct, settings.KELLY_FLOOR_PCT), settings.KELLY_CAP_PCT)
        result = KellyResult(
            win_rate=round(win_rate * 100.0, 1), avg_rr=round(avg_rr, 2),
            full_kelly_pct=round(full_kelly_pct, 2), half_kelly_pct=round(half_kelly_pct, 2),
            applied_pct=round(applied, 2), capped=half_kelly_pct > settings.KELLY_CAP_PCT,
        )
        logger.info("kelly: wr=%.0f%% rr=%.2f full=%.2f%% applied=%.2f%%",
                    result.win_rate, result.avg_rr, result.full_kelly_pct, result.applied_pct)
        return result
