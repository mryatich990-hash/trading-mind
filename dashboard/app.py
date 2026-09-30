"""Unified Flask dashboard: one mobile-first page, ten live sections.

All data flows through JSON endpoints refreshed by dashboard.js every
DASHBOARD_REFRESH_SEC seconds (open trades every 10s). Control endpoints
cover pause/resume/close-all. The dashboard is read-only regarding logs.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
app.secret_key = settings.SECRET_KEY


def _eat(ts) -> str:
    """Render a UTC timestamp in EAT (UTC+3)."""
    if ts is None:
        return "-"
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return str(ts)[:8]


@app.route("/")
def index() -> str:
    """Render the unified dashboard page."""
    return render_template("index.html", refresh_sec=settings.DASHBOARD_REFRESH_SEC)


@app.route("/health")
def health() -> tuple[str, int]:
    """Health endpoint for the watchdog (200 when DB answers)."""
    try:
        db.get_state("running", "")
        return "ok", 200
    except Exception:
        return "db down", 500


@app.route("/webhook/tv", methods=["POST"])
def webhook_tv() -> tuple[str, int]:
    """TradingView alerts (same pipeline as execution/tv_webhook.py).

    TradingView expects a quick 2xx, so we ack immediately and process in
    the background, mirroring the standalone server's behavior.
    """
    from execution.tv_webhook import get_server

    server = get_server()
    try:
        body = request.get_data() or b""
    except Exception:
        body = b""
    threading.Thread(target=server._process, args=(body, request.remote_addr or ""),
                     name="tv-alert", daemon=True).start()
    return "", 204


# ---- section APIs ----


@app.route("/api/status")
def api_status() -> tuple:
    """Section 1: status bar + section 2: account overview."""
    running = db.get_state("running", "0") == "1"
    paused = db.get_state("trading_paused", "0") == "1"
    breakers = db.unresolved_breakers()
    mode = db.get_state("trade_mode", "demo")
    status = "HALTED" if breakers else ("PAUSED" if paused else
                                        "RUNNING" if running else "OFFLINE")
    regime = _latest_regime()
    vix = _latest_vix()
    session = _current_session()
    open_trades = db.open_trades()
    day_rows = db.trades_today()
    wins = sum(1 for r in day_rows if r["status"] == "closed"
               and float(r["pnl_usd"] or 0) > 0)
    losses = sum(1 for r in day_rows if r["status"] == "closed"
                 and float(r["pnl_usd"] or 0) <= 0)
    return jsonify({
        "status": status, "mode": mode, "breakers": breakers,
        "session": session, "regime": regime, "vix": vix,
        "open_trades": len(open_trades),
        "open_pnl_est": round(len(open_trades) * 0, 2),
        "trades_today": len(day_rows), "wins_today": wins, "losses_today": losses,
        "daily_pnl": db.daily_pnl(), "daily_target": settings.MAX_TRADES_PER_DAY,
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/equity")
def api_equity() -> tuple:
    """Equity curve across closed trades (section 2)."""
    rows = db.closed_trades(limit=500)
    rows = list(reversed(rows))
    balance = 10000.0
    points = []
    for r in rows:
        balance += float(r["pnl_usd"] or 0)
        points.append({"t": _eat(r.get("closed_at")), "v": round(balance, 2)})
    watermark = float(db.get_state("high_watermark", "0") or 0)
    dd = 0.0
    if watermark > 0 and points:
        dd = max(0.0, (watermark - points[-1]["v"]) / watermark * 100.0)
    return jsonify({"points": points, "high_watermark": watermark,
                    "drawdown_pct": round(dd, 2)})


@app.route("/api/open_trades")
def api_open_trades() -> tuple:
    """Section 3: open trades table."""
    trades = []
    for t in db.open_trades():
        trades.append({
            "id": t["id"], "pair": t["pair"], "direction": t["direction"],
            "lots": t["lots"], "entry": float(t["entry_price"]),
            "sl": float(t["sl"]), "tp": float(t["tp"]),
            "strategy": t["strategy"], "session": t["session"],
            "opened": _eat(t["opened_at"] or t.get("created_at")),
        })
    return jsonify({"trades": trades, "refresh_sec": 10})


@app.route("/api/research")
def api_research() -> tuple:
    """Section 4: research live feed."""
    rows = db.recent_research(limit=5)
    for r in rows:
        r["time"] = _eat(r.pop("created_at", None))
    return jsonify({"cycles": rows})


@app.route("/api/institutional")
def api_institutional() -> tuple:
    """Section 5: COT, retail sentiment, intermarket bias, VIX, DXY."""
    from sqlalchemy import text as sqltext
    from core.db import engine
    out = {"cot": [], "retail": [], "biases": {}, "vix_series": []}
    try:
        with engine.begin() as conn:
            cot = conn.execute(sqltext(
                "SELECT currency, commercial_net, noncommercial_net, commercial_pctile "
                "FROM cot_reports WHERE report_date = (SELECT MAX(report_date) "
                "FROM cot_reports)")).all()
            out["cot"] = [{"currency": r[0], "commercial_net": r[1],
                           "noncommercial_net": r[2], "pctile": r[3]} for r in cot]
            retail = conn.execute(sqltext(
                "SELECT rs.pair, rs.pct_long FROM retail_sentiment rs "
                "WHERE rs.id = (SELECT MAX(r2.id) FROM retail_sentiment r2 "
                "WHERE r2.pair = rs.pair)")).all()
            out["retail"] = [{"pair": r[0], "pct_long": r[1]} for r in retail]
            snap = conn.execute(sqltext(
                "SELECT usd_bias, eur_bias, gbp_bias, jpy_bias, xau_bias, dxy_trend, vix "
                "FROM intermarket_snapshots ORDER BY id DESC LIMIT 1")).first()
            if snap:
                out["biases"] = {"USD": snap[0], "EUR": snap[1], "GBP": snap[2],
                                 "JPY": snap[3], "XAU": snap[4]}
                out["dxy_trend"] = snap[5]
                out["vix"] = snap[6]
            vix_rows = conn.execute(sqltext(
                "SELECT created_at, vix FROM intermarket_snapshots "
                "ORDER BY id DESC LIMIT 120")).all()
            out["vix_series"] = [{"t": _eat(r[0]), "v": r[1]} for r in reversed(vix_rows)]
    except Exception as exc:
        logger.warning("institutional api failed: %s", exc)
    return jsonify(out)


@app.route("/api/strategies")
def api_strategies() -> tuple:
    """Section 6: strategy performance table."""
    from sqlalchemy import text as sqltext
    from core.db import engine
    out = []
    try:
        with engine.begin() as conn:
            rows = conn.execute(sqltext(
                "SELECT w.strategy, COALESCE(w.weight, 1.0), COALESCE(w.enabled, TRUE) "
                "FROM strategy_weights w ORDER BY w.strategy")).all()
            names = [r[0] for r in rows]
            week_stats: dict[str, dict] = {}
            stat_rows = conn.execute(sqltext(
                "SELECT strategy, COUNT(*), SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END), "
                "SUM(pnl_usd) FROM trades WHERE status='closed' "
                "AND closed_at >= CURRENT_DATE - 7 GROUP BY strategy")).all()
            for r in stat_rows:
                week_stats[r[0]] = {"signals": r[1], "wins": r[2] or 0,
                                    "pnl": float(r[3] or 0)}
            for name in names or []:
                s = week_stats.get(name, {"signals": 0, "wins": 0, "pnl": 0.0})
                wr = (s["wins"] / s["signals"] * 100.0) if s["signals"] else 0.0
                weight = next((r[1] for r in rows if r[0] == name), 1.0)
                enabled = next((r[2] for r in rows if r[0] == name), True)
                out.append({"name": name, "signals": s["signals"], "wins": s["wins"],
                            "win_rate": round(wr, 1), "pnl": round(s["pnl"], 2),
                            "weight": weight, "enabled": bool(enabled)})
    except Exception as exc:
        logger.warning("strategies api failed: %s", exc)
    return jsonify({"strategies": out})


@app.route("/api/heatmap")
def api_heatmap() -> tuple:
    """Section 7: session heatmap cells."""
    try:
        from ml.session_optimizer import SessionOptimizer
        return jsonify({"cells": SessionOptimizer().heatmap()})
    except Exception as exc:
        return jsonify({"cells": [], "error": str(exc)})


@app.route("/api/montecarlo")
def api_montecarlo() -> tuple:
    """Section 8: latest Monte Carlo results."""
    from sqlalchemy import text as sqltext
    from core.db import engine
    try:
        with engine.begin() as conn:
            row = conn.execute(sqltext(
                "SELECT simulations, prob_dd10, prob_dd20, prob_dd30, prob_ruin, "
                "created_at FROM monte_carlo_results ORDER BY id DESC LIMIT 1")).first()
        if row:
            return jsonify({"simulations": row[0], "prob_dd10": row[1],
                            "prob_dd20": row[2], "prob_dd30": row[3],
                            "prob_ruin": row[4], "run_at": _eat(row[5])})
    except Exception as exc:
        logger.warning("montecarlo api failed: %s", exc)
    return jsonify({})


@app.route("/api/ml")
def api_ml() -> tuple:
    """Section 9: ML model status."""
    from sqlalchemy import text as sqltext
    from core.db import engine
    out = {}
    try:
        with engine.begin() as conn:
            row = conn.execute(sqltext(
                "SELECT trained_at, n_trades, train_accuracy, test_accuracy, features_json "
                "FROM ml_metrics ORDER BY id DESC LIMIT 1")).first()
            if row:
                out = {"trained_at": _eat(row[0]), "n_trades": row[1],
                       "train_accuracy": row[2], "test_accuracy": row[3],
                       "importances": json.loads(row[4] or "[]")}
    except Exception as exc:
        logger.warning("ml api failed: %s", exc)
    out["ab_running"] = db.get_state("ab_running", "0") == "1"
    return jsonify(out)


@app.route("/api/history")
def api_history() -> tuple:
    """Section 10: trade history (last 100, filterable)."""
    pair = request.args.get("pair", "").upper()
    outcome = request.args.get("outcome", "")
    rows = db.closed_trades(limit=100)
    out = []
    for r in rows:
        if pair and r["pair"] != pair:
            continue
        pnl = float(r["pnl_usd"] or 0)
        if outcome == "win" and pnl <= 0:
            continue
        if outcome == "loss" and pnl > 0:
            continue
        out.append({
            "pair": r["pair"], "direction": r["direction"], "lots": r["lots"],
            "entry": r["entry_price"], "exit": r["exit_price"], "pips": r["pips"],
            "pnl": pnl, "strategy": r["strategy"], "session": r["session"],
            "confluence": r["confluence_score"], "conviction": r["groq_conviction"],
            "closed": _eat(r.get("closed_at")),
            "reasoning": (r.get("groq_reasoning") or "")[:300],
        })
    return jsonify({"trades": out})


# ---- upgrades (9) ----


@app.route("/api/upgrades")
def api_upgrades() -> tuple:
    """One endpoint powering all seven new panels (deep learning, NLP, tick,
    scalping, stat-arb, analytics, system health)."""
    out: dict = {}
    try:
        from upgrades_registry import get_registry

        registry = get_registry()
        out = registry.status() if registry else {"error": "registry not initialized"}
    except Exception as exc:
        out = {"error": str(exc)[:200]}
    return jsonify(out)


# ---- controls ----


@app.route("/api/control", methods=["POST"])
def api_control() -> tuple:
    """Pause / resume / close-all / recompute-montecarlo."""
    action = (request.json or {}).get("action", "")
    if action == "pause":
        db.set_state("trading_paused", "1")
        db.audit("operator", "pause", "dashboard")
        return jsonify({"ok": True})
    if action == "resume":
        db.set_state("trading_paused", "0")
        db.resolve_breakers()
        db.audit("operator", "resume", "dashboard")
        return jsonify({"ok": True})
    if action == "close_all":
        tm = _get_trade_manager()
        closed = 0
        if tm is not None:
            for t in db.open_trades():
                try:
                    price = tm._current_price(t["pair"], t["direction"])
                    tm.close(t, price, "Manual")
                    closed += 1
                except Exception as exc:
                    logger.error("dashboard close failed: %s", exc)
        return jsonify({"ok": True, "closed": closed})
    if action == "montecarlo":
        from risk.monte_carlo import MonteCarloEngine
        result = MonteCarloEngine().run(db.closed_trades(limit=200))
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext(
                "INSERT INTO monte_carlo_results (created_at, simulations, prob_dd10, "
                "prob_dd20, prob_dd30, prob_ruin) VALUES (:t, :s, :a, :b, :c, :d)"),
                {"t": db._utcnow(), "s": result.simulations, "a": result.prob_dd10,
                 "b": result.prob_dd20, "c": result.prob_dd30, "d": result.prob_ruin})
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "unknown action"}), 400


@app.route("/api/force_trade", methods=["POST"])
def api_force_trade() -> tuple:
    """Operator-forced end-to-end test trade (EURUSD 0.01 market order).

    Web process only writes the trigger state key; the engine worker picks
    it up on its next cycle and runs the real risk+execution pipeline.
    """
    if (request.json or {}).get("action") != "force_trade":
        return jsonify({"ok": False, "error": "body must be {action: force_trade}"}), 400
    db.set_state("force_trade_requested", "1")
    db.audit("operator", "force_trade_requested", "dashboard endpoint")
    return jsonify({"ok": True, "detail": "engine will execute within one cycle (~60s)"})


def _get_trade_manager():
    """Trade manager instance when the app runs inside the bot process."""
    try:
        from execution.trade_manager import TradeManager
        return TradeManager(broker=None)
    except Exception:
        return None


# ---- shared helpers ----


def _latest_regime() -> str:
    """Latest market regime row."""
    from sqlalchemy import text as sqltext
    from core.db import engine
    try:
        with engine.begin() as conn:
            row = conn.execute(sqltext(
                "SELECT regime FROM market_regime ORDER BY id DESC LIMIT 1")).first()
            return row[0] if row else "unknown"
    except Exception:
        return "unknown"


def _latest_vix() -> float:
    """Latest VIX value."""
    from sqlalchemy import text as sqltext
    from core.db import engine
    try:
        with engine.begin() as conn:
            row = conn.execute(sqltext(
                "SELECT vix FROM intermarket_snapshots ORDER BY id DESC LIMIT 1")).first()
            return float(row[0]) if row and row[0] else 0.0
    except Exception:
        return 0.0


def _current_session() -> str:
    """Current trading session name."""
    from strategies.strategy_selector import session_of
    return session_of(datetime.now(timezone.utc))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.PORT, debug=False)
