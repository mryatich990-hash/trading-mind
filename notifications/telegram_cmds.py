"""Operator command implementations for the Telegram bot.

Each function takes a list of string args and returns a reply string.
They read only from the DB and system_state so they are safe to call from
the polling thread without blocking the trading loop.
"""

from __future__ import annotations

import json
from typing import Optional

from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["register_all", "cmd_status", "cmd_trades", "cmd_balance"]

_engine_refs: dict = {}


def wire(engine_refs: dict) -> None:
    """Give commands access to live services (risk manager, engines...)."""
    _engine_refs.update(engine_refs)


def cmd_status(args: list[str]) -> str:
    """/status — bot state, open trades, today PnL."""
    running = db.get_state("running", "0") == "1"
    breakers = db.unresolved_breakers()
    mode = db.get_state("trade_mode", "demo")
    open_trades = db.open_trades()
    day = db.daily_pnl()
    lines = [
        f"🤖 Bot: {'RUNNING' if running else 'STOPPED'} | mode: {mode.upper()}",
        f"Open trades: {len(open_trades)} | Today PnL: ${day:+.2f}",
    ]
    if breakers:
        lines.append(f"🚨 Breakers: {', '.join(breakers)}")
    for t in open_trades[:5]:
        lines.append(f"  • {t['pair']} {t['direction']} {t['lots']} lots "
                     f"@ {float(t['entry_price']):.5f} ({t['strategy']})")
    return "\n".join(lines)


def cmd_trades(args: list[str]) -> str:
    """/trades — last 10 closed trades."""
    rows = db.closed_trades(limit=10)
    if not rows:
        return "No closed trades yet."
    lines = ["Last 10 closed trades:"]
    for r in rows:
        pnl = float(r.get("pnl_usd") or 0)
        flag = "✅" if pnl > 0 else "❌"
        lines.append(f"{flag} {r['pair']} {r['direction']} {pnl:+.2f} USD "
                     f"({r.get('strategy', '')})")
    return "\n".join(lines)


def cmd_balance(args: list[str]) -> str:
    """/balance — balance and equity from the active broker."""
    broker = _engine_refs.get("broker")
    if broker is not None:
        try:
            acct = broker.account()
            return (f"💰 Balance: ${acct['balance']:.2f} | Equity: ${acct['equity']:.2f}\n"
                    f"Margin level: {acct['margin_level']:.0f}% | "
                    f"Free: ${acct['margin_free']:.2f}")
        except Exception as exc:
            logger.warning("balance via broker failed: %s", exc)
    return f"💰 Today PnL: ${db.daily_pnl():+.2f} (broker unavailable)"


def cmd_pairs(args: list[str]) -> str:
    """/pairs — configured pairs and weights of strategies covering them."""
    from config import settings
    lines = ["Trading pairs:"]
    for p in settings.TRADING_PAIRS:
        lines.append(f"  • {p}")
    return "\n".join(lines)


def cmd_research(args: list[str]) -> str:
    """/research — last completed research cycle."""
    rows = db.recent_research(limit=1)
    if not rows:
        return "No research cycles yet."
    r = rows[0]
    return (f"🔬 {r['pair']} {r['result'].upper()} (conf {r['confluence_score']}, "
            f"conv {r['conviction']}): {r['reason']}")


def cmd_stats(args: list[str]) -> str:
    """/stats — full performance stats since start."""
    rows = db.closed_trades(limit=10000)
    if not rows:
        return "No trades since start."
    wins = [r for r in rows if float(r["pnl_usd"] or 0) > 0]
    losses = [r for r in rows if float(r["pnl_usd"] or 0) <= 0]
    total_pnl = sum(float(r["pnl_usd"] or 0) for r in rows)
    gross_win = sum(float(r["pnl_usd"] or 0) for r in wins)
    gross_loss = abs(sum(float(r["pnl_usd"] or 0) for r in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    return (f"📈 Stats since start:\n"
            f"Trades: {len(rows)} | Win rate: {len(wins)/len(rows)*100:.0f}%\n"
            f"PnL: ${total_pnl:+.2f} | Profit factor: {pf:.2f}\n"
            f"Best: ${max((float(r['pnl_usd'] or 0) for r in rows), default=0):+.2f} | "
            f"Worst: ${min((float(r['pnl_usd'] or 0) for r in rows), default=0):+.2f}")


def cmd_cot(args: list[str]) -> str:
    """/cot — current COT positioning."""
    try:
        from institutional.cot_reader import COTReader
        snaps = COTReader().snapshots()
        lines = ["COT positioning (commercial bias):"]
        for cur, s in sorted(snaps.items()):
            lines.append(f"  • {cur}: {s.bias.upper()} "
                         f"(comm pctile {s.commercial_pctile:.0f}%)")
        return "\n".join(lines) if len(lines) > 1 else "COT data unavailable."
    except Exception as exc:
        return f"COT unavailable: {exc}"


def cmd_sentiment(args: list[str]) -> str:
    """/sentiment — retail sentiment snapshot."""
    try:
        from institutional.retail_sentiment import RetailSentimentReader
        snaps = RetailSentimentReader().snapshots()
        if not snaps:
            return "Retail sentiment unavailable (no API key or data)."
        lines = ["Retail sentiment (contrarian view):"]
        for pair, s in sorted(snaps.items()):
            lines.append(f"  • {pair}: {s.pct_long:.0f}% long → {s.contrarian_bias.upper()}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Sentiment unavailable: {exc}"


def cmd_vix(args: list[str]) -> str:
    """/vix — VIX and regime summary."""
    try:
        from institutional.intermarket_analyzer import IntermarketAnalyzer
        snap = IntermarketAnalyzer().snapshot()
        regime = "HALT" if snap.vix_halt else ("high" if snap.vix >= 30 else
                                               "elevated" if snap.vix >= 20 else "calm")
        return (f"🌡️ VIX: {snap.vix:.1f} ({regime}) | VIX3M: {snap.vix3m:.1f}\n"
                f"USD bias: {snap.usd_bias:+.0f} | DXY: {snap.dxy_trend}\n"
                f"Size scale: {snap.vix_position_scale:.1f}x")
    except Exception as exc:
        return f"VIX unavailable: {exc}"


def cmd_pause(args: list[str]) -> str:
    """/pause — pause trading, research continues."""
    db.set_state("trading_paused", "1")
    db.audit("operator", "pause", "via telegram")
    return "⏸ Trading paused. Research continues in background."


def cmd_resume(args: list[str]) -> str:
    """/resume — resume trading and clear breakers."""
    db.set_state("trading_paused", "0")
    db.set_state("dd_halt_active", "0")
    db.resolve_breakers()
    db.audit("operator", "resume", "via telegram")
    return "▶️ Trading resumed. All breakers cleared."


def cmd_close(args: list[str]) -> str:
    """/close — close all open trades immediately."""
    tm = _engine_refs.get("trade_manager")
    if tm is None:
        return "Trade manager not wired in this process."
    count = 0
    for t in db.open_trades():
        try:
            price = tm._current_price(t["pair"], t["direction"])
            tm.close(t, price, "Manual")
            count += 1
        except Exception as exc:
            logger.error("manual close failed: %s", exc)
    return f"🛑 Closed {count} trades."


def cmd_demo(args: list[str]) -> str:
    """/demo — switch to demo mode."""
    db.set_state("trade_mode", "demo")
    db.audit("operator", "mode_demo", "via telegram")
    return "🧪 Switched to DEMO mode."


def cmd_live(args: list[str]) -> str:
    """/live — switch to live mode (only after gates passed)."""
    if db.get_state("demo_gate_passed", "0") != "1":
        return "⛔ Demo gate not passed yet (60 trades @ 52% WR, PF 1.3). " \
               "Live mode locked."
    if db.get_state("backtest_gate_passed", "0") != "1":
        return "⛔ Backtest gate not passed yet. Live mode locked."
    db.set_state("trade_mode", "live")
    db.audit("operator", "mode_live", "via telegram")
    return "🔴 Switched to LIVE mode."


def cmd_enable(args: list[str]) -> str:
    """/enable {strategy} — re-enable a strategy."""
    if not args:
        return "Usage: /enable {strategy}"
    name = args[0]
    weight = db.strategy_weight(name) or 1.0
    db.set_strategy_weight(name, max(weight, 1.0), True, "enabled via telegram")
    return f"✅ {name} enabled."


def cmd_disable(args: list[str]) -> str:
    """/disable {strategy} — disable a strategy."""
    if not args:
        return "Usage: /disable {strategy}"
    db.set_strategy_weight(args[0], 0.0, False, "disabled via telegram")
    return f"🚫 {args[0]} disabled."


def cmd_backtest(args: list[str]) -> str:
    """/backtest {strategy} {pair} {days} — run an on-demand backtest."""
    if len(args) < 2:
        return "Usage: /backtest {strategy} {pair} [days]"
    name, pair = args[0], args[1].upper()
    days = int(args[2]) if len(args) > 2 else 30
    try:
        import pandas as pd
        from backtesting.backtest_engine import BacktestEngine
        from data.market_data_engine import MarketDataEngine
        from strategies.strategy_selector import StrategySelector
        selector = StrategySelector()
        strategy = next((s for s in selector.strategies if s.name == name), None)
        if strategy is None:
            return f"Unknown strategy: {name}"
        frames = MarketDataEngine().get_frames(pair)
        start = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        engine = BacktestEngine()
        result = engine.run(strategy, frames, pair, start=start)
        r = result.as_dict()
        return (f"🧪 Backtest {name} {pair} ({days}d):\n"
                f"Trades: {r['trades']} | WR: {r['win_rate']}% | PF: {r['profit_factor']}\n"
                f"MaxDD: {r['max_drawdown']}% | Sharpe: {r['sharpe']} | "
                f"Passed: {'✅' if r['passed'] else '❌'}")
    except Exception as exc:
        return f"Backtest failed: {exc}"


def cmd_kelly(args: list[str]) -> str:
    """/kelly — current Kelly fraction."""
    from risk.kelly_criterion import KellyCriterion
    result = KellyCriterion().compute(db.closed_trades(limit=60))
    return (f"🎲 Kelly: WR {result.win_rate}% | avg RR {result.avg_rr}\n"
            f"Full: {result.full_kelly_pct:.2f}% | Half: {result.half_kelly_pct:.2f}%\n"
            f"Applied: {result.applied_pct:.2f}% per trade"
            f"{' (capped)' if result.capped else ''}")


def cmd_montecarlo(args: list[str]) -> str:
    """/montecarlo — run and show the Monte Carlo simulation."""
    from risk.monte_carlo import MonteCarloEngine
    result = MonteCarloEngine().run(db.closed_trades(limit=200))
    if result.simulations == 0:
        return "Monte Carlo needs closed trades first."
    return (f"🎰 Monte Carlo ({result.simulations} sims, 100 trades):\n"
            f"P(DD 10%): {result.prob_dd10}% | P(DD 20%): {result.prob_dd20}%\n"
            f"P(DD 30%): {result.prob_dd30}% | P(ruin): {result.prob_ruin}%\n"
            f"Outcomes p5/p50/p95: {result.p5_outcome}% / {result.p50_outcome}% / "
            f"{result.p95_outcome}%")


def cmd_heatmap(args: list[str]) -> str:
    """/heatmap — session performance heatmap summary."""
    try:
        from ml.session_optimizer import SessionOptimizer
        cells = SessionOptimizer().heatmap()
        if not cells:
            return "Heatmap empty (needs 25+ trades)."
        best = sorted(cells, key=lambda c: -c["win_rate"])[:5]
        lines = ["🔥 Best hours:"]
        for c in best:
            lines.append(f"  • {c['strategy']} {c['hour']:02d}:00 dow{c['day_of_week']}: "
                         f"{c['win_rate']:.0f}% ({c['trades']} trades)")
        return "\n".join(lines)
    except Exception as exc:
        return f"Heatmap unavailable: {exc}"


def cmd_version(args: list[str]) -> str:
    """/version — strategy versions and A/B status."""
    ab_running = db.get_state("ab_running", "0") == "1"
    prompt_ab = db.get_state("ab_prompt_running", "0") == "1"
    from sqlalchemy import text as sqltext
    from core.db import engine
    with engine.begin() as conn:
        rows = conn.execute(sqltext(
            "SELECT strategy, weight, enabled FROM strategy_weights "
            "ORDER BY strategy")).all()
    lines = [f"⚙️ A/B test: {'RUNNING' if ab_running else 'idle'} | "
             f"prompt A/B: {'RUNNING' if prompt_ab else 'idle'}",
             "Strategy weights:"]
    for name, weight, enabled in rows or []:
        lines.append(f"  • {name}: {weight:.2f} {'✅' if enabled else '🚫'}")
    return "\n".join(lines)


COMMANDS = {
    "status": cmd_status, "trades": cmd_trades, "balance": cmd_balance,
    "pairs": cmd_pairs, "research": cmd_research, "stats": cmd_stats,
    "cot": cmd_cot, "sentiment": cmd_sentiment, "vix": cmd_vix,
    "pause": cmd_pause, "resume": cmd_resume, "close": cmd_close,
    "demo": cmd_demo, "live": cmd_live, "enable": cmd_enable,
    "disable": cmd_disable, "backtest": cmd_backtest, "kelly": cmd_kelly,
    "montecarlo": cmd_montecarlo, "heatmap": cmd_heatmap, "version": cmd_version,
}


def register_all(bot) -> None:
    """Attach every command to the bot."""
    for name, fn in COMMANDS.items():
        bot.register(name, fn)
