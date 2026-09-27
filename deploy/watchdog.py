"""Overnight watchdog: keep-alive pings + Telegram alerts for the cloud engine.

Runs on the LOCAL PC (systemd user service trading-watchdog.service) and:
  1. Pings the Render /health endpoint every 5 minutes (keeps free tier warm).
  2. Watches the cloud Supabase for the first paper trade and Telegram-alerts once.
  3. Alerts on: weekend halt lifting, engine unreachable (3x), DB down (3x),
     and recovery after any incident. Everything logged to logs/watchdog.log.

Secrets come from the local .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # standalone service: load creds explicitly

RENDER_HEALTH = os.environ.get("RENDER_HEALTH_URL", "https://trading-bot-f3yn.onrender.com/health")
RENDER_STATUS = RENDER_HEALTH.replace("/health", "/status")
SUPA_URL = os.environ.get("SUPABASE_URL", "https://xhrfmptvkansdqkpouri.supabase.co")
SUPA_ANON = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

PING_INTERVAL = 300          # 5 min keep-alive
STATUS_INTERVAL = 60         # watch the cloud every minute
FAIL_THRESHOLD = 3           # consecutive failures before alerting
STATE_FILE = Path("/tmp/trading_watchdog_state.json")


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} {msg}", flush=True)


def load_state() -> dict:
    try:
        import json

        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    try:
        import json

        STATE_FILE.write_text(json.dumps(state))
    except Exception:
        pass


def telegram(text: str) -> None:
    """Fire-and-forget Telegram message (local bot owns the chat)."""
    if not TG_TOKEN or not TG_CHAT:
        log("telegram skipped (no creds)")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        log(f"telegram -> {r.status_code}")
    except Exception as exc:
        log(f"telegram failed: {exc}")


def main() -> None:
    state = load_state()
    fails = 0
    last_ping = 0.0
    last_status = 0.0
    was_halted = None
    log(f"watchdog started | render={RENDER_HEALTH}")

    while True:
        now = time.time()

        # ---- keep-alive ping every 5 min ----
        if now - last_ping >= PING_INTERVAL:
            last_ping = now
            try:
                r = requests.get(RENDER_HEALTH, timeout=25)
                if r.status_code == 200:
                    log("keep-alive ping OK")
                else:
                    log(f"keep-alive ping HTTP {r.status_code}")
            except Exception as exc:
                log(f"keep-alive ping failed: {str(exc)[:80]}")

        # ---- cloud status watch every minute ----
        if now - last_status >= STATUS_INTERVAL:
            last_status = now
            status, health = {}, {}
            ok = True
            try:
                status = requests.get(RENDER_STATUS, timeout=25).json()
            except Exception as exc:
                ok = False
                log(f"status unreachable: {str(exc)[:80]}")
            try:
                health = requests.get(RENDER_HEALTH, timeout=25).json()
            except Exception:
                pass

            engine_alive = health.get("engine") == "alive"
            db_ok = health.get("database") == "connected"
            halted = "weekend" in (status.get("halted_by") or [])

            # --- failure alerting ---
            if not ok or not engine_alive or not db_ok:
                fails += 1
                if fails == FAIL_THRESHOLD:
                    telegram(
                        f"🔴 <b>Cloud engine problem</b>\n"
                        f"engine: {health.get('engine', '?')} | db: {health.get('database', '?')}\n"
                        f"consecutive fails: {fails}\n"
                        f"Supervisor auto-restarts it; check {RENDER_HEALTH}"
                    )
            else:
                if fails >= FAIL_THRESHOLD:
                    telegram("🟢 <b>Cloud engine recovered</b> — all checks green again.")
                fails = 0

            # --- weekend halt lift ---
            if was_halted and not halted:
                telegram(
                    "🌍 <b>Weekend halt lifted</b> — cloud engine is now scanning "
                    "for setups. First paper trade will be alerted here."
                )
                log("halt lifted -> notified")
            was_halted = halted

            # --- first trade alert ---
            if status.get("trades_today", 0) > 0 and not state.get("first_trade_alerted"):
                trade = {}
                try:
                    tr = requests.get(
                        f"{SUPA_URL}/rest/v1/trades?select=pair,direction,lots,entry_price,strategy,mode&order=id.desc&limit=1",
                        headers={"apikey": SUPA_ANON, "Authorization": f"Bearer {SUPA_ANON}"},
                        timeout=15,
                    ).json()
                    trade = tr[0] if tr else {}
                except Exception:
                    pass
                telegram(
                    "🎯 <b>FIRST CLOUD TRADE!</b>\n"
                    f"pair: {trade.get('pair', '?')} | side: {trade.get('direction', '?')}\n"
                    f"lots: {trade.get('lots', '?')} @ {trade.get('entry_price', '?')}\n"
                    f"strategy: {trade.get('strategy', '?')} | mode: {trade.get('mode', '?')}\n"
                    "Dashboard: https://trading-bot-dashboard-five-zeta.vercel.app"
                )
                state["first_trade_alerted"] = True
                save_state(state)
                log("FIRST TRADE -> notified")

            # --- first research entry alert (engine processing markets) ---
            if not state.get("first_research_alerted") and not halted:
                try:
                    rr = requests.get(
                        f"{SUPA_URL}/rest/v1/research_cycles?select=id,pair,result&order=id.desc&limit=1",
                        headers={"apikey": SUPA_ANON, "Authorization": f"Bearer {SUPA_ANON}"},
                        timeout=15,
                    ).json()
                    if rr:
                        state["first_research_alerted"] = True
                        save_state(state)
                        telegram(
                            f"🔬 <b>Cloud research running</b>\n"
                            f"first cycle: {rr[0].get('pair')} -> {rr[0].get('result')}"
                        )
                        log("first research -> notified")
                except Exception:
                    pass

        time.sleep(5)


if __name__ == "__main__":
    main()
