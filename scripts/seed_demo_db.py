#!/usr/bin/env python3
"""Seed a throwaway demo database so the dashboard can be previewed safely.

Creates demo_trading.db from config/schema.sql and fills it with ~45 days of
realistic demo data: closed/open trades, research cycles, COT, sentiment,
intermarket snapshots (VIX series), market regime, strategy weights from the
real gate run, session stats, Monte Carlo, and ML metrics.

Usage:
    python scripts/seed_demo_db.py            # (re)create demo_trading.db
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text as sqltext  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "demo_trading.db")

rng = random.Random(42)

GATE = {  # strategy -> (weight, enabled, reason) -- mirrors the real gate run
    "asian_range_fade": (1.0, True, "backtest gate passed (WR 75%)"),
    "cot_extreme": (0.0, False, "backtest gate failed (WR 0% PF 0.00)"),
    "ema_trend_rider": (1.0, True, "backtest gate passed (WR 50%)"),
    "fvg_fill": (0.0, False, "backtest gate failed (WR 23% PF 0.45)"),
    "harmonic_prz": (0.0, False, "backtest gate failed (WR 0% PF 0.00)"),
    "liquidity_sweep": (1.0, True, "backtest gate passed (WR 66%)"),
    "london_breakout": (0.0, False, "backtest gate failed (WR 38% PF 1.41)"),
    "macd_momentum": (1.0, True, "backtest gate passed (WR 67%)"),
    "news_spike_fade": (0.0, False, "backtest gate failed (WR 0% PF 0.00)"),
    "ny_reversal": (1.0, True, "backtest gate passed (WR 75%)"),
    "order_block_sniper": (1.0, True, "backtest gate passed (WR 57%)"),
    "rsi_divergence": (1.0, True, "backtest gate passed (WR 70%)"),
    "vpoc_magnet": (0.0, False, "backtest gate failed (WR 0% PF 0.00)"),
    "vwap_reversion": (0.0, False, "backtest gate failed (WR 0% PF 0.00)"),
    "wyckoff_springthrust": (0.0, False, "backtest gate failed (WR 0% PF 0.00)"),
}


def apply_schema(engine) -> None:
    """Apply config/schema.sql statement by statement (SQLite-safe)."""
    with open(os.path.join(ROOT, "config", "schema.sql"), encoding="utf-8") as fh:
        statements = [s.strip() for s in fh.read().split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            try:
                conn.execute(sqltext(stmt))
            except Exception:
                pass  # skip Postgres-specific statements (ENUMs, indexes on missing cols)


def ts(days_ago: float) -> str:
    """ISO timestamp N days ago."""
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def seed_trades(conn) -> tuple[int, float]:
    """~45 days of closed trades + 2 open. Returns (count, final balance)."""
    strategies = [s for s, (_, en, _) in GATE.items() if en]
    skill = {"asian_range_fade": 1.35, "ny_reversal": 1.30, "rsi_divergence": 1.25,
             "macd_momentum": 1.15, "ema_trend_rider": 1.05, "liquidity_sweep": 1.0,
             "order_block_sniper": 0.95}
    balance = 10_000.0
    rows = []
    for d in range(45, -1, -1):
        n = rng.choice((0, 1, 2, 2, 3))
        for k in range(n):
            strat = rng.choice(strategies)
            sk = skill.get(strat, 1.0)
            win = rng.random() < min(0.78, 0.52 * sk)
            risk = 100.0  # 1% of 10k, flat for simplicity
            rr = rng.uniform(1.4, 2.4)
            pnl = round(risk * rr * (1.05 if win else -1.0), 2)
            balance += pnl
            hour = rng.choice((8, 9, 10, 11, 14, 15, 16, 17, 20))
            opened = (datetime.now(timezone.utc).replace(
                hour=hour, minute=rng.randint(0, 59), second=0, microsecond=0)
                - timedelta(days=d))
            closed = opened + timedelta(minutes=rng.choice((15, 30, 45, 60, 90, 120)))
            if closed > datetime.now(timezone.utc):
                closed = datetime.now(timezone.utc) - timedelta(minutes=rng.randint(5, 55))
            pair = rng.choice(("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"))
            pip_size = 0.01 if pair == "USDJPY" else (0.01 if pair == "XAUUSD" else 0.0001)
            pip_val = rng.uniform(0.0008, 0.0025) if pip_size == 0.0001 else rng.uniform(0.015, 0.05)
            entry = {"EURUSD": 1.085, "GBPUSD": 1.270, "USDJPY": 152.5, "XAUUSD": 2650.0}[pair]
            direction = rng.choice(("buy", "sell"))
            entry = round(entry * (1 + rng.uniform(-0.01, 0.01)), 5)
            sl_dist = pip_val * rng.uniform(15, 35)
            sl = entry - sl_dist if direction == "buy" else entry + sl_dist
            tp = entry + 1.8 * sl_dist if direction == "buy" else entry - 1.8 * sl_dist
            exit_price = (tp if win else sl) * (1 + rng.uniform(-0.0002, 0.0002))
            pips = (exit_price - entry) / pip_size * (1 if direction == "buy" else -1)
            conviction = rng.randint(74, 96)
            rows.append({
                "pair": pair, "direction": direction, "lots": round(rng.uniform(0.1, 0.5), 2),
                "entry_price": round(entry, 5), "sl": round(sl, 5), "tp": round(tp, 5),
                "opened_at": opened.isoformat(), "closed_at": closed.isoformat(),
                "status": "closed", "strategy": strat,
                "session": rng.choice(("London", "NewYork", "Overlap", "Asia")),
                "confluence_score": rng.randint(3, 6),
                "groq_conviction": conviction,
                "groq_reasoning": (
                    f"{strat.replace('_', ' ').title()} setup on {pair}: confluence ≥3, "
                    f"regime filter passed, Groq conviction {conviction}/100. "
                    f"Risk 1% ({round(sl_dist / pip_size):.0f} pip stop). "
                    + ("Target reached; partial at TP1, runner to TP2."
                       if win else "Invalidated early; stop hit at full risk.")),
                "exit_price": round(exit_price, 5),
                "pips": round(pips, 1), "pnl_usd": pnl,
                "rr_achieved": round(rr, 2) if win else 0.0,
                "slippage_pips": round(rng.uniform(0, 0.8), 2),
                "signal_hash": f"demo{d:02d}{k:02d}{rng.randrange(10**8):08d}",
                "mode": "demo",
            })
    # two open trades
    for pair, direction, entry in (("EURUSD", "buy", 1.0842), ("XAUUSD", "sell", 2655.4)):
        sl_dist = 0.0022 if pair == "EURUSD" else 3.5
        rows.append({
            "pair": pair, "direction": direction, "lots": 0.25,
            "entry_price": entry, "sl": entry - sl_dist if direction == "buy" else entry + sl_dist,
            "tp": entry + 2 * sl_dist if direction == "buy" else entry - 2 * sl_dist,
            "opened_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "status": "open", "strategy": rng.choice(strategies),
            "session": "Overlap", "confluence_score": 4, "groq_conviction": 81,
            "groq_reasoning": "Live position opened after research consensus.",
            "mode": "demo",
        })
    # executemany requires identical keys on every row; fill the rest with None
    all_cols = sorted({c for r in rows for c in r})
    for r in rows:
        for c in all_cols:
            r.setdefault(c, None)
    cols = all_cols
    placeholders = ", ".join(f":{c}" for c in cols)
    col_list = ", ".join(cols)
    conn.execute(sqltext(
        f"INSERT INTO trades ({col_list}) VALUES ({placeholders})"), rows)
    return len(rows), balance


def seed_research(conn) -> None:
    """Research cycle feed rows."""
    reasons = ["confluence ≥ 3", "session filter active", "conviction above threshold",
               "ATR within range", "risk guard approved"]
    rows = []
    for i in range(5):
        accepted = rng.random() < 0.5
        rows.append({
            "pair": rng.choice(("EURUSD", "GBPUSD", "XAUUSD")),
            "result": "accepted" if accepted else "rejected",
            "reason": rng.choice(reasons),
            "confluence_score": rng.randint(2, 6),
            "conviction": rng.randint(58, 94),
            "research_json": "{}",
            "created_at": ts(i * 0.25),
        })
    conn.execute(sqltext(
        "INSERT INTO research_cycles (pair, result, reason, confluence_score, "
        "conviction, research_json, created_at) VALUES (:pair, :result, :reason, "
        ":confluence_score, :conviction, :research_json, :created_at)"), rows)


def seed_institutional(conn) -> None:
    """COT, retail sentiment, intermarket snapshots with a VIX series."""
    cot = [("EUR", 41200, 118400, 62.0), ("GBP", -18300, -52100, 38.0),
           ("JPY", -87600, -143200, 71.0), ("AUD", 15400, 22900, 55.0),
           ("CHF", 6800, 12400, 48.0), ("CAD", -22900, -38100, 44.0)]
    conn.execute(sqltext(
        "INSERT INTO cot_reports (report_date, currency, commercial_net, "
        "noncommercial_net, commercial_pctile) VALUES (:d, :c, :cm, :nc, :p)"),
        [{"d": ts(3)[:10], "c": c, "cm": cm, "nc": nc, "p": p}
         for c, cm, nc, p in cot])
    retail = [("EURUSD", 38.0), ("GBPUSD", 61.0), ("USDJPY", 72.0), ("XAUUSD", 44.0)]
    conn.execute(sqltext(
        "INSERT INTO retail_sentiment (pair, pct_long, pct_short) VALUES (:p, :l, :s)"),
        [{"p": p, "l": l, "s": round(100 - l, 1)} for p, l in retail])
    snap = []
    for i in range(120):
        t = ts(120 - i) if False else (datetime.now(timezone.utc)
                                       - timedelta(hours=120 - i))
        vix = max(11.0, 16.5 + 4.2 * math.sin(i / 9.0) + rng.uniform(-0.8, 0.8))
        snap.append({"t": t.isoformat(),
                     "usd": round(rng.uniform(-0.5, 0.5), 2),
                     "eur": round(rng.uniform(-0.5, 0.5), 2),
                     "gbp": round(rng.uniform(-0.5, 0.5), 2),
                     "jpy": round(rng.uniform(-0.5, 0.5), 2),
                     "xau": round(rng.uniform(-0.5, 0.5), 2),
                     "dxy": rng.choice(("up", "down", "flat")),
                     "vix": round(vix, 2)})
    conn.execute(sqltext(
        "INSERT INTO intermarket_snapshots (created_at, usd_bias, eur_bias, gbp_bias, "
        "jpy_bias, xau_bias, dxy_trend, vix) VALUES (:t, :usd, :eur, :gbp, :jpy, "
        ":xau, :dxy, :vix)"), snap)


def seed_regime_and_more(conn) -> None:
    """Regime, strategy weights, session stats, Monte Carlo, ML metrics."""
    conn.execute(sqltext(
        "INSERT INTO market_regime (regime, confidence, recommended, avoid) "
        "VALUES (:r, :c, :rec, :av)"),
        {"r": "trending", "c": 72,
         "rec": json.dumps(["ema_trend_rider", "macd_momentum", "ny_reversal"]),
         "av": json.dumps(["vwap_reversion"])})
    conn.execute(sqltext(
        "INSERT INTO strategy_weights (strategy, weight, enabled, reason) "
        "VALUES (:s, :w, :e, :r)"),
        [{"s": s, "w": w, "e": 1 if en else 0, "r": reason}
         for s, (w, en, reason) in GATE.items()])
    # session heatmap cells, aggregated from the seeded trades just like
    # ml/session_optimizer.py does (hour x weekday per strategy)
    rows = conn.execute(sqltext(
        "SELECT strategy, opened_at, pnl_usd FROM trades "
        "WHERE status = 'closed'")).mappings().all()
    stats: dict[tuple, list] = {}
    for r in rows:
        try:
            o = datetime.fromisoformat(str(r["opened_at"]).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        key = (str(r["strategy"]), o.hour, o.weekday())
        stats.setdefault(key, [0, 0])
        stats[key][0] += 1
        stats[key][1] += 1 if float(r["pnl_usd"] or 0) > 0 else 0
    conn.execute(sqltext(
        "INSERT INTO session_heatmap (updated_at, strategy, hour, day_of_week, "
        "trades, wins, win_rate, blacklisted) VALUES (:t, :s, :h, :d, :n, :w, :r, :b)"),
        [{"t": ts(0), "s": s, "h": h, "d": d, "n": n, "w": w,
          "r": round(w / n * 100.0, 1), "b": n >= 10 and w / n < 0.4}
         for (s, h, d), (n, w) in stats.items()])
    conn.execute(sqltext(
        "INSERT INTO monte_carlo_results (simulations, prob_dd10, prob_dd20, "
        "prob_dd30, prob_ruin) VALUES (2000, 0.18, 0.06, 0.02, 0.004)"))
    conn.execute(sqltext(
        "INSERT INTO ml_metrics (n_trades, train_accuracy, test_accuracy, features_json) "
        "VALUES (412, 0.68, 0.63, :f)"),
        {"f": json.dumps([["session", 0.24], ["confluence_score", 0.19],
                          ["atr_pips", 0.17], ["h1_trend", 0.15],
                          ["rsi_14", 0.13], ["vix", 0.12]])})


def main() -> None:
    """(Re)create the demo DB and seed everything."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    apply_schema(engine)
    with engine.begin() as conn:
        n_trades, balance = seed_trades(conn)
        seed_research(conn)
        seed_institutional(conn)
        seed_regime_and_more(conn)
        conn.execute(sqltext(
            "INSERT INTO system_state (key, value) VALUES ('running', '1')"))
        conn.execute(sqltext(
            "INSERT INTO system_state (key, value) VALUES ('trade_mode', 'demo')"))
        conn.execute(sqltext(
            "INSERT INTO system_state (key, value) VALUES ('high_watermark', :hw)"),
            {"hw": str(round(balance * 1.04, 2))})
        conn.execute(sqltext(
            "INSERT INTO system_state (key, value) "
            "VALUES ('last_backtest_at', :v)"), {"v": ts(1)})
    print(f"demo DB ready: {DB_PATH}")
    print(f"  trades: {n_trades} closed + 2 open, final balance ≈ ${balance:,.2f}")


if __name__ == "__main__":
    main()
