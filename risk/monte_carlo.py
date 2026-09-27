"""Monte Carlo risk engine: 10,000 simulations of the next 100 trades."""

import random
from dataclasses import dataclass, field

from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["MonteCarloResult", "MonteCarloEngine"]


@dataclass
class MonteCarloResult:
    """Aggregated simulation results."""

    simulations: int
    prob_dd10: float  # probability of a >=10% drawdown within 100 trades
    prob_dd20: float
    prob_dd30: float
    prob_ruin: float  # probability of >=50% drawdown
    p5_outcome: float
    p50_outcome: float
    p95_outcome: float
    reduce_size_recommended: bool


class MonteCarloEngine:
    """Simulates trade sequences using actual win rate and payoff statistics."""

    def __init__(self, simulations: int = 10000, horizon: int = 100,
                 risk_pct: float = 1.0) -> None:
        self.simulations = simulations
        self.horizon = horizon
        self.risk_pct = risk_pct

    def run(self, closed_trades: list[dict], seed: int = 42) -> MonteCarloResult:
        """Simulate using actual per-trade returns (compounded on equity)."""
        if not closed_trades:
            return MonteCarloResult(0, 0, 0, 0, 0, 0, 0, 0, False)
        returns = [float(t["pnl_usd"] or 0) / 10000.0 for t in closed_trades[:200]]  # per 10k
        if not returns:
            return MonteCarloResult(0, 0, 0, 0, 0, 0, 0, 0, False)

        rng = random.Random(seed)
        dd10 = dd20 = dd30 = ruin = 0
        finals: list[float] = []
        for _ in range(self.simulations):
            equity = 1.0
            peak = 1.0
            max_dd = 0.0
            for _ in range(self.horizon):
                r = rng.choice(returns) * (self.risk_pct / 1.0)
                equity *= (1.0 + r)
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
            finals.append(equity)
            if max_dd >= 0.10:
                dd10 += 1
            if max_dd >= 0.20:
                dd20 += 1
            if max_dd >= 0.30:
                dd30 += 1
            if max_dd >= 0.50:
                ruin += 1

        finals.sort()
        n = self.simulations
        result = MonteCarloResult(
            simulations=n,
            prob_dd10=round(dd10 / n * 100.0, 1),
            prob_dd20=round(dd20 / n * 100.0, 1),
            prob_dd30=round(dd30 / n * 100.0, 1),
            prob_ruin=round(ruin / n * 100.0, 1),
            p5_outcome=round((finals[int(n * 0.05)] - 1) * 100.0, 1),
            p50_outcome=round((finals[int(n * 0.50)] - 1) * 100.0, 1),
            p95_outcome=round((finals[int(n * 0.95)] - 1) * 100.0, 1),
            reduce_size_recommended=False,
        )
        if result.prob_dd30 > 5.0:
            result.reduce_size_recommended = True
        logger.info("monte carlo: dd10=%.0f%% dd30=%.0f%% ruin=%.0f%%",
                    result.prob_dd10, result.prob_dd30, result.prob_ruin)
        return result
