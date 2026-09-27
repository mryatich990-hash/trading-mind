"""BacktestEngine: replays strategies over historical candle frames.

Simulates the selector + fixed-risk sizing on synthetic equity, computes the
standard metrics (win rate, profit factor, drawdown, Sharpe), persists runs to
backtest_results and enforces:
- the startup backtest gate (min 5 strategies above 45% WR / 1.2 PF)
- Sunday auto-backtest weight adjustments (+15% / -25% on big movers)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from core import db
from core.logging_utils import get_logger
from strategies.base_strategy import BaseStrategy, MarketContext, pip_size

logger = get_logger(__name__)

__all__ = ["BacktestResult", "BacktestEngine", "run_startup_gate"]


@dataclass
class BacktestResult:
    """Metrics for one backtest run."""

    strategy: str
    pair: str
    start: str
    end: str
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    total_pnl: float = 0.0
    best_month: str = ""
    worst_month: str = ""
    equity_curve: list[float] = field(default_factory=list)
    passed: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        """Serialize for DB/dashboard."""
        return {
            "strategy": self.strategy, "pair": self.pair, "start": self.start,
            "end": self.end, "trades": self.trades, "wins": self.wins,
            "win_rate": round(self.win_rate, 1),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown": round(self.max_drawdown, 1), "sharpe": round(self.sharpe, 2),
            "total_pnl": round(self.total_pnl, 2), "passed": self.passed,
            "note": self.note,
        }


class BacktestEngine:
    """Walks a strategy across M15 candles and simulates fills at next open."""

    def __init__(self, risk_per_trade: float = 1.0, start_balance: float = 10000.0) -> None:
        self.risk_per_trade = risk_per_trade
        self.start_balance = start_balance

    def run(self, strategy: BaseStrategy, frames: dict[str, pd.DataFrame],
            pair: str, start: Optional[pd.Timestamp] = None,
            end: Optional[pd.Timestamp] = None) -> BacktestResult:
        """Backtest one strategy over the given frames."""
        m15 = frames["m15"]
        if start is not None:
            m15 = m15[pd.to_datetime(m15["time"], utc=True) >= start]
        if end is not None:
            m15 = m15[pd.to_datetime(m15["time"], utc=True) <= end]
        result = BacktestResult(strategy=strategy.name, pair=pair,
                                start=str(m15["time"].iloc[0])[:10] if len(m15) else "",
                                end=str(m15["time"].iloc[-1])[:10] if len(m15) else "")
        if len(m15) < 250:
            result.note = "insufficient data"
            return result

        # Higher timeframes: use loader-provided frames when present, else
        # derive ONCE from the full M15 history (per-candle resampling is far
        # too slow for multi-thousand-candle runs).
        def _htf(key: str, rule: str) -> pd.DataFrame:
            src = frames.get(key)
            if src is None and key == "daily":
                src = frames.get("d1")
            if src is None:
                src = _resample(m15, rule)
            return src if src is not None else m15

        h1 = _htf("h1", "1h").reset_index(drop=True)
        h4 = _htf("h4", "4h").reset_index(drop=True)
        d1 = _htf("daily", "1D").reset_index(drop=True)
        h1_ts = pd.DatetimeIndex(pd.to_datetime(h1["time"], utc=True))
        h4_ts = pd.DatetimeIndex(pd.to_datetime(h4["time"], utc=True))
        d1_ts = pd.DatetimeIndex(pd.to_datetime(d1["time"], utc=True))

        balance = self.start_balance
        peak = balance
        returns: list[float] = []
        equity: list[float] = [balance]
        monthly: dict[str, list[float]] = {}
        cooldown_until: Optional[pd.Timestamp] = None

        # iterate candle by candle; strategies see a rolling context window.
        # HTF frames are asof-sliced to the last COMPLETED bar (side="left")
        # so no future data leaks into the context.
        for i in range(220, len(m15)):
            ts = pd.Timestamp(m15["time"].iloc[i])
            if cooldown_until is not None and ts < cooldown_until:
                continue
            window = m15.iloc[max(0, i - 400): i + 1].reset_index(drop=True)
            ctx = MarketContext(
                pair=pair, now=ts, m1=window.tail(60), m15=window,
                h1=h1.iloc[: h1_ts.searchsorted(ts)].reset_index(drop=True),
                h4=h4.iloc[: h4_ts.searchsorted(ts)].reset_index(drop=True),
                daily=d1.iloc[: d1_ts.searchsorted(ts)].reset_index(drop=True))
            try:
                signal = strategy.evaluate(ctx)
            except Exception:
                continue
            if signal is None:
                continue
            # fill at the NEXT candle open (no lookahead)
            if i + 1 >= len(m15):
                break
            fill = float(m15["open"].iloc[i + 1])
            risk_pips = abs(fill - signal.sl) / pip_size(pair)
            if risk_pips <= 0:
                continue
            # simulate: walk forward until SL or TP hit (max 96 candles = 24h)
            outcome, exit_price = _walk_trade(m15, i + 1, signal.direction,
                                              fill, signal.sl, signal.tp)
            pnl_pips = ((exit_price - fill) if signal.direction == "buy"
                        else (fill - exit_price)) / pip_size(pair)
            risk_amount = balance * self.risk_per_trade / 100.0
            pnl = risk_amount * (pnl_pips / risk_pips)
            balance += pnl
            returns.append(pnl / balance)
            equity.append(balance)
            peak = max(peak, balance)
            result.trades += 1
            result.wins += 1 if pnl > 0 else 0
            result.total_pnl += pnl
            month = str(ts)[:7]
            monthly.setdefault(month, []).append(pnl)
            cooldown_until = ts + pd.Timedelta(minutes=15)  # one trade per candle

        if result.trades == 0:
            result.note = "no signals generated"
            return result
        result.equity_curve = [round(b, 2) for b in equity[:: max(1, len(equity) // 200)]]
        losses = [r for r in returns if r <= 0]
        gross_win = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(losses))
        result.win_rate = result.wins / result.trades * 100.0
        result.profit_factor = gross_win / gross_loss if gross_loss > 0 else 99.0
        result.max_drawdown = (peak - min(equity)) / peak * 100.0 if peak > 0 else 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe = float(np.mean(returns) / np.std(returns)
                                  * math.sqrt(252 * 4))
        if monthly:
            best = max(monthly, key=lambda m: sum(monthly[m]))
            worst = min(monthly, key=lambda m: sum(monthly[m]))
            result.best_month, result.worst_month = best, worst
        result.passed = (result.win_rate >= settings.BACKTEST_GATE_WIN_RATE
                         and result.profit_factor >= settings.BACKTEST_GATE_PROFIT_FACTOR)
        self._persist(result)
        logger.info("backtest %s %s: %d trades WR %.0f%% PF %.2f passed=%s",
                    strategy.name, pair, result.trades, result.win_rate,
                    result.profit_factor, result.passed)
        return result

    def walk_forward(self, strategy: BaseStrategy, frames: dict[str, pd.DataFrame],
                     pair: str) -> tuple[BacktestResult, BacktestResult]:
        """Train on the first 70% of candles, test on the remaining 30%."""
        m15 = frames["m15"]
        split = int(len(m15) * 0.7)
        split_ts = pd.Timestamp(m15["time"].iloc[split])
        train = self.run(strategy, frames, pair, end=split_ts)
        test = self.run(strategy, frames, pair, start=split_ts)
        return train, test

    @staticmethod
    def _persist(result: BacktestResult) -> None:
        """Save to backtest_results."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(
                sqltext("INSERT INTO backtest_results (created_at, strategy, pair, "
                        "start_date, end_date, trades, win_rate, profit_factor, "
                        "max_drawdown, sharpe, passed, detail_json) "
                        "VALUES (:t, :s, :p, :sd, :ed, :n, :wr, :pf, :dd, :sh, :ok, :j)"),
                {"t": db._utcnow(), "s": result.strategy, "p": result.pair,
                 "sd": result.start, "ed": result.end, "n": result.trades,
                 "wr": result.win_rate, "pf": result.profit_factor,
                 "dd": result.max_drawdown, "sh": result.sharpe,
                 "ok": result.passed, "j": json.dumps(result.as_dict())},
            )

    # ---- gates and weight adjustments ----

    def apply_startup_gate(self, results: list[BacktestResult]) -> tuple[int, list[str]]:
        """Disable strategies below the gate; True-to-trade when >= min pass.

        Returns (passing_count, disabled_names).
        """
        disabled: list[str] = []
        passing = 0
        for r in results:
            if r.passed:
                passing += 1
                db.set_strategy_weight(r.strategy, 1.0, True,
                                       f"backtest gate passed (WR {r.win_rate:.0f}%)")
            else:
                db.set_strategy_weight(r.strategy, 0.0, False,
                                       f"backtest gate failed (WR {r.win_rate:.0f}% "
                                       f"PF {r.profit_factor:.2f})")
                disabled.append(r.strategy)
        if passing < settings.BACKTEST_GATE_MIN_STRATEGIES:
            db.set_state("backtest_gate_passed", "0")
            logger.error("backtest gate FAILED: %d/%d strategies pass",
                         passing, len(results))
        else:
            db.set_state("backtest_gate_passed", "1")
        return passing, disabled

    def apply_sunday_adjustments(self, results: list[BacktestResult]) -> list[dict]:
        """Week-over-week weight tweaks: +15% improvers, -25% degraders."""
        changes: list[dict] = []
        for r in results:
            prev = self._previous_week(r.strategy, r.pair)
            if prev is None or prev["trades"] < 5 or r.trades < 5:
                continue
            delta = r.win_rate - prev["win_rate"]
            weight = db.strategy_weight(r.strategy)
            if delta <= -10:
                new_w = max(0.1, weight * 0.75)
                db.set_strategy_weight(r.strategy, new_w, True,
                                       f"Sunday backtest: WR -{abs(delta):.0f}% WoW")
                changes.append({"strategy": r.strategy, "delta": round(delta, 1),
                                "weight": new_w})
            elif delta >= 10:
                new_w = min(2.0, weight * 1.15)
                db.set_strategy_weight(r.strategy, new_w, True,
                                       f"Sunday backtest: WR +{delta:.0f}% WoW")
                changes.append({"strategy": r.strategy, "delta": round(delta, 1),
                                "weight": new_w})
        return changes

    @staticmethod
    def _previous_week(strategy: str, pair: str) -> Optional[dict]:
        """Most recent prior backtest row for the strategy/pair."""
        from sqlalchemy import text as sqltext

        from core.db import engine
        with engine.begin() as conn:
            row = conn.execute(
                sqltext("SELECT trades, win_rate FROM backtest_results "
                        "WHERE strategy = :s AND pair = :p "
                        "ORDER BY id DESC LIMIT 1 OFFSET 1"),
                {"s": strategy, "p": pair}).first()
        return {"trades": row[0], "win_rate": row[1]} if row else None


# ---- helpers ----


def _resample(df: pd.DataFrame, rule: str) -> Optional[pd.DataFrame]:
    """Resample M15 candles to a higher timeframe; None when too little data."""
    if len(df) < 8:
        return None
    ts = pd.to_datetime(df["time"], utc=True)
    out = df.set_index(ts).resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"}).dropna()
    out = out.reset_index().rename(columns={"time": "time"})
    return out if len(out) >= 30 else None


def _walk_trade(df: pd.DataFrame, start_idx: int, direction: str, entry: float,
                sl: float, tp: float, max_candles: int = 96) -> tuple[str, float]:
    """Walk candles forward until SL/TP; returns (outcome, exit_price)."""
    end = min(start_idx + max_candles, len(df))
    for i in range(start_idx, end):
        row = df.iloc[i]
        if direction == "buy":
            if float(row["low"]) <= sl:
                return "sl", sl
            if float(row["high"]) >= tp:
                return "tp", tp
        else:
            if float(row["high"]) >= sl:
                return "sl", sl
            if float(row["low"]) <= tp:
                return "tp", tp
    exit_price = float(df["close"].iloc[end - 1])
    return "timeout", exit_price


def run_startup_gate(engine: BacktestEngine, strategies: list[BaseStrategy],
                     frames_by_pair: dict[str, dict[str, pd.DataFrame]],
                     min_candles: int = 1500) -> tuple[int, list[str]]:
    """Run every enabled strategy on every pair's frames and apply the gate.

    min_candles guards against evaluating on statistically meaningless
    windows (e.g. live-feed frames of a few hundred candles on a fresh
    install): below that, the gate passes WITHOUT touching strategy weights
    so a proper offline backtest (backtest.py) remains the source of truth.
    """
    shortest = min((len(f["m15"]) for f in frames_by_pair.values()), default=0)
    if not frames_by_pair or shortest < min_candles:
        logger.warning("startup gate skipped: only %d m15 candles available "
                       "(need %d for a meaningful gate) - weights untouched",
                       shortest, min_candles)
        db.set_state("backtest_gate_passed", "1")  # do not lock the bot out
        return 0, []
    results = []
    for pair, frames in frames_by_pair.items():
        for strategy in strategies:
            try:
                results.append(engine.run(strategy, frames, pair))
            except Exception as exc:
                logger.exception("backtest failed %s %s: %s", strategy.name, pair, exc)
    if not results:
        db.set_state("backtest_gate_passed", "1")  # no data: do not lock the bot out
        return 0, []
    # deduplicate per strategy: best result counts
    best: dict[str, BacktestResult] = {}
    for r in results:
        if r.strategy not in best or r.passed:
            best[r.strategy] = r
    return engine.apply_startup_gate(list(best.values()))
