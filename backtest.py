"""Backtest runner: replay all 15 strategies over historical M15 candles.

Usage:
    python backtest.py                       # EURUSD default, 50 days
    python backtest.py --pairs EURUSD,GBPUSD --days 55
    python backtest.py --strategies london_breakout,ema_trend_rider
    python backtest.py --walk-forward        # 70/30 train/test split
    python backtest.py --gate                # apply the startup backtest gate

The simulation walks every M15 candle from index 220, builds the unified
MarketContext, asks each strategy for a signal, fills at the NEXT candle open
(no lookahead), and resolves the trade by walking forward until SL/TP (max 96
candles = 24h). Fixed 1% risk sizing; results persist to backtest_results and
print a summary table with WR / PF / maxDD / Sharpe per strategy.
"""

from __future__ import annotations

import argparse

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description="Unified strategy backtester")
    parser.add_argument("--pairs", default="EURUSD",
                        help="comma-separated pairs (default EURUSD)")
    parser.add_argument("--days", type=int, default=50,
                        help="days of M15 history (max 59, yfinance limit)")
    parser.add_argument("--strategies", default="",
                        help="comma-separated strategy names (default: all 15)")
    parser.add_argument("--risk", type=float, default=1.0,
                        help="risk %% per trade (default 1.0)")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="starting balance (default 10000)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="70/30 train/test split per strategy")
    parser.add_argument("--gate", action="store_true",
                        help="apply the startup backtest gate afterwards")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the data cache and re-download")
    return parser.parse_args()


def select_strategies(names: str) -> list:
    """Instantiate the requested strategies (all when blank)."""
    from strategies.library import all_strategies

    strategies = all_strategies()
    if not names:
        return strategies
    wanted = {n.strip() for n in names.split(",") if n.strip()}
    found = [s for s in strategies if s.name in wanted]
    missing = wanted - {s.name for s in found}
    if missing:
        print(f"⚠️ unknown strategies: {', '.join(sorted(missing))}")
    return found


def run_all(pairs: list[str], days: int, strategies: list, risk: float,
            balance: float, walk_forward: bool, use_cache: bool) -> dict:
    """Run every strategy on every pair; returns results grouped by strategy."""
    from backtesting.backtest_engine import BacktestEngine
    from backtesting.data_loader import load_frames

    engine = BacktestEngine(risk_per_trade=risk, start_balance=balance)
    results: dict[str, list] = {}

    for pair in pairs:
        print(f"\n📥 loading {pair} ({days}d of M15)...")
        try:
            frames = load_frames(pair, days=days, use_cache=use_cache)
        except Exception as exc:
            print(f"  ✗ data unavailable: {exc}")
            continue
        print(f"  ✓ {len(frames['m15'])} m15 / {len(frames['h1'])} h1 / "
              f"{len(frames['h4'])} h4 / {len(frames['d1'])} d1 candles")

        for strategy in strategies:
            label = f"{strategy.name} {pair}"
            try:
                if walk_forward:
                    train, test = engine.walk_forward(strategy, frames, pair)
                    results.setdefault(strategy.name, []).extend([train, test])
                    _print_result(label + " [train]", train)
                    _print_result(label + " [test ]", test)
                else:
                    result = engine.run(strategy, frames, pair)
                    results.setdefault(strategy.name, []).append(result)
                    _print_result(label, result)
            except Exception as exc:
                logger.exception("backtest crashed for %s: %s", label, exc)
                print(f"  ✗ {label}: {exc}")
    return results


def _print_result(label: str, r) -> None:
    """One-line result row."""
    status = "✓" if r.passed else ("·" if r.trades else "–")
    print(f"  {status} {label:38s} trades {r.trades:4d}  WR {r.win_rate:5.1f}%  "
          f"PF {r.profit_factor:5.2f}  DD {r.max_drawdown:4.1f}%  "
          f"Sharpe {r.sharpe:5.2f}  PnL {r.total_pnl:+8.2f}"
          + (f"  ({r.note})" if r.note and r.trades == 0 else ""))


def summarize(results: dict[str, list]) -> None:
    """Aggregate per-strategy summary + gate verdict."""
    print("\n" + "=" * 100)
    print("AGGREGATE PER STRATEGY (across pairs)")
    print("=" * 100)
    print(f"{'strategy':26s} {'trades':>7s} {'WR%':>6s} {'PF':>6s} {'PnL':>10s} {'gate':>6s}")
    passing = 0
    for name in sorted(results):
        runs = results[name]
        trades = sum(r.trades for r in runs)
        if not trades:
            print(f"{name:26s} {0:7d} {'-':>6s} {'-':>6s} {'-':>10s}   ✗ no trades")
            continue
        wins = sum(r.wins for r in runs)
        wr = wins / trades * 100.0
        pnl = sum(r.total_pnl for r in runs)
        # profit factor: trade-weighted average of per-run PFs
        pf_values = [r.profit_factor for r in runs if r.trades]
        pf = (sum(r.profit_factor * r.trades for r in runs if r.trades)
              / sum(r.trades for r in runs if r.trades)) if pf_values else 0.0
        pf = sum(pf_values) / len(pf_values) if pf_values else 0.0
        passed = wr >= settings.BACKTEST_GATE_WIN_RATE and pf >= settings.BACKTEST_GATE_PROFIT_FACTOR
        passing += passed
        print(f"{name:26s} {trades:7d} {wr:6.1f} {pf:6.2f} {pnl:+10.2f}   "
              f"{'✓' if passed else '✗'}")
    print("-" * 100)
    print(f"gate: {passing}/{len(results)} strategies above "
          f"{settings.BACKTEST_GATE_WIN_RATE:.0f}% WR / "
          f"{settings.BACKTEST_GATE_PROFIT_FACTOR:.1f} PF "
          f"(need {settings.BACKTEST_GATE_MIN_STRATEGIES})")


def main() -> None:
    """Entry point."""
    args = parse_args()
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    strategies = select_strategies(args.strategies)
    if not strategies:
        print("no strategies selected")
        return

    print(f"🔬 backtesting {len(strategies)} strategies on {', '.join(pairs)} "
          f"({args.days}d, risk {args.risk}%, balance ${args.balance:,.0f})")

    db.init_db()
    db.set_state("last_backtest_at", db._utcnow().isoformat())

    results = run_all(pairs, args.days, strategies, args.risk, args.balance,
                      args.walk_forward, not args.refresh)

    summarize(results)

    if args.gate:
        from backtesting.backtest_engine import BacktestEngine

        print("\n🚦 applying startup backtest gate...")
        # pick each strategy's best passing run across pairs
        best: dict[str, object] = {}
        for runs in results.values():
            for r in runs:
                if r.strategy not in best or (r.passed and not best[r.strategy].passed):
                    best[r.strategy] = r
        passing_count, disabled = BacktestEngine().apply_startup_gate(
            list(best.values()))
        print(f"   gate {'PASSED' if passing_count >= settings.BACKTEST_GATE_MIN_STRATEGIES else 'FAILED'}: "
              f"{passing_count} strategies pass")
        if disabled:
            print(f"   disabled: {', '.join(disabled)}")


if __name__ == "__main__":
    main()
