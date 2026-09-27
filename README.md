# 🏛️ Institutional AI Forex Trading System

Fully autonomous institutional-grade Forex/index/gold trading system combining
**15 strategies** (SMC, volume profile, VWAP, Wyckoff, harmonics, COT), a
**Groq LLM decision brain with 3x consensus + numeric verification**,
**Kelly-criterion risk sizing**, **ML loss filtering**, and a **10-step
research pipeline** that runs before every trade. Runs locally (SQLite +
systemd) or deploys free to **Render + Supabase + Vercel**.

> ⚠️ **Trading foreign exchange carries substantial risk.** This bot can lose
> money rapidly. Always run on a demo/practice account until the demo gate
> (60 trades, ≥52% WR, PF ≥1.3) passes. Nothing here is financial advice.

---

## Architecture

```
 M15 candle close (10 pairs)
        │
        ▼
 StrategySelector ── scores all 15 strategies, picks best ≥70
        │
        ▼
 ResearchEngine (10 steps, 60-90s, never skipped)
   1. Macro context (intermarket/COT/retail/yields)
   2. News gate (red-event window abort)
   3. HTF bias (daily + H4 must agree)
   4. Institutional structure (OB/FVG/VPOC/VWAP/Wyckoff/Elliott)
   5. Volume & order flow (delta, HVN, absorption)
   6. M15 entry checklist (≥8/10)
   7. Pattern/harmonic bonus
   8. Historical setup match (poor history → 9/10 required)
   9. Groq 3x consensus + verifier (any skip wins, conviction ≥74)
 10. RiskManager final gate (Kelly sizing, breakers, correlation)
        │
        ▼
 ExecutionEngine ── MT5 → OANDA → shadow; duplicate-hash guard
        │
        ▼
 TradeManager (TP1 35%+BE, TP2 35%, trail 15/25p, invalidation,
               weekend/swap protection, one half-size re-entry)
```

**Safety stack:** circuit breakers (feed/MT5/Groq/VIX/ATR/margin/weekend),
drawdown tiers 0-3-5-8-10%, correlation filter (≤0.70, EURUSD+GBPUSD ≤0.5
lots), anti-martingale sizing, loss-streak pauses, max 2 open trades / 8 per
day / 3% daily loss.

**Gates:** backtest gate (5+ strategies ≥45% WR, PF ≥1.2) before demo; demo
gate (60 trades ≥52% WR, PF ≥1.3) before `/live`.

---

## File structure

```
trading-bot/
├── main.py                  # entry point: 10-check startup + async loop
├── config/                  # settings.py (env), schema.sql (26 tables)
├── core/                    # db.py, event_bus.py, graceful_shutdown.py
├── data/                    # market data failover, indicators, VWAP, volume profile
├── strategies/              # unified base + 15 strategies + selector
├── strategy_engine/         # prompt-3 strategy package (10 strategies)
├── research/                # macro, HTF, entry validator, matcher, engine
├── risk/                    # risk manager, Kelly, drawdown, correlation, breakers, Monte Carlo
├── execution/               # MT5/OANDA connectors, engine, trade manager
├── backtesting/             # backtest engine + gates, shadow trader
├── ml/                      # RandomForest filter, A/B tester, session optimizer
├── institutional/           # intermarket, COT, retail sentiment, central banks
├── news/                    # calendar engine, sentiment
├── notifications/           # Telegram bot + 21 operator commands
├── dashboard/               # Flask app: 10-section mobile-first UI
├── ai/                      # Groq brain, verifier, prompt builder
├── tests/                   # pytest suite (unified system)
├── Dockerfile / railway.toml / Procfile / .env.example
```

---

## The 15 strategies

| # | Strategy | Style |
|---|----------|-------|
| 1 | London Breakout | Asian range break + retest |
| 2 | NY Reversal | 12:00 reversal pattern |
| 3 | Order Block Sniper | OB retest entries |
| 4 | FVG Fill | Fair value gap fills |
| 5 | EMA Trend Rider | H1 pullback continuation |
| 6 | Liquidity Sweep | Stop-hunt reversals |
| 7 | RSI Divergence | Momentum divergence |
| 8 | MACD Momentum | Histogram acceleration |
| 9 | News Spike Fade | Spike exhaustion (market order) |
| 10 | Asian Range Fade | Range extremes fade |
| 11 | VWAP Reversion | 2σ band fade to VWAP |
| 12 | VPOC Magnet | LVN transit to VPOC |
| 13 | Wyckoff Spring/Upthrust | Accumulation events |
| 14 | Harmonic PRZ | Gartley/Bat reversals |
| 15 | COT Extreme | Commercial positioning swings |

---

## Deployment (free: Render + Supabase + Vercel)

Three services, $0/month, independent failure domains: dashboard down → bot
keeps trading; bot down → dashboard still shows last known data.

| Service | Runs | Tier |
|---|---|---|
| **Render** (`render.yaml`) | Bot engine (`main.py` via `deploy/run_engine.py`) + health endpoint (`health.py`) in ONE web service | Free |
| **Supabase** | PostgreSQL (all 24 tables) + realtime change feeds | Free |
| **Vercel** (`dashboard-nextjs/`) | Next.js realtime dashboard | Free |
| **UptimeRobot** | Pings `/health` every 5 min, keeps the free service warm | Free |

> Why one Render web service instead of worker+web: Render's **worker tier has
> no free plan**. The free **web** tier hosts both the health endpoint
> (UptimeRobot target) and the engine supervisor. If the service ever cold
> starts, `deploy/start_render.sh` relaunches the engine automatically.

### STEP 1 — Supabase (database + realtime)

1. supabase.com → **New project** (pick a nearby region, e.g. Frankfurt).
2. **Settings → API**: copy
   - Project URL → `SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_URL`
   - `service_role` key → `SUPABASE_SERVICE_KEY` (**secret**, engine only)
   - `anon` key → `NEXT_PUBLIC_SUPABASE_ANON_KEY` (public-safe, read-only via RLS)
3. **Settings → Database → Connection pooling** → copy the URI (port **6543**)
   and make it SQLAlchemy-compatible by switching the scheme:
   `postgresql://…` → `postgresql+psycopg2://…?pgbouncer=true&sslmode=require`
   → this is `DATABASE_URL` and `SUPABASE_DB_URL` on Render.
4. **SQL Editor** → run **`supabase/realtime_rls.sql`** (realtime publication
   + row level security). The engine auto-creates all tables on first boot;
   optionally run `config/schema.sql` first to inspect them.
5. Realtime is enabled for: `trades`, `research_cycles`, `circuit_breakers`.

### STEP 2 — Render (bot engine)

1. render.com → **New + → Blueprint** → connect this repo (reads `render.yaml`).
2. Fill the env vars it lists (`SUPABASE_*`, `GROQ_API_KEY`, `TELEGRAM_*`,
   `CTRADER_*`, data API keys). `DEMO_MODE=true` is preset.
3. **Deploy**. First boot applies the schema to Supabase automatically, then
   trades on the **paper broker** until broker credentials are added.
4. Your health URL: `https://trading-bot.onrender.com/health`.

### STEP 3 — UptimeRobot (keeps it awake)

1. uptimerobot.com (free) → **Add New Monitor → HTTP**
2. URL: your `/health` URL · Interval: **5 minutes** · add your alert email
3. Free Render web services sleep after ~15 min idle; these pings keep it
   warm and the engine self-restarts on any cold start.

### STEP 4 — Vercel (dashboard)

1. vercel.com → **Add New → Project** → import repo → **Root Directory:
   `dashboard-nextjs`**
2. Env vars: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`,
   `SUPABASE_SERVICE_KEY`, `DASHBOARD_KEY` (any long random string — set the
   **same value on Render**; it gates `/api/pause` and `/api/close`)
3. **Deploy**. Dashboard shows RUNNING + live trades/research via Supabase
   realtime (no polling); **Pause** and **Close all** buttons write state keys
   the engine consumes within one cycle (~60 s).

### STEP 5 — Verify

1. Open the dashboard on your phone → status badge **RUNNING**
2. Telegram → `/status` → bot responds
3. Supabase → **Table editor** → data flowing into `trades`, `research_cycles`
4. New research entries appear on the dashboard instantly (realtime push)

**Local development is unchanged**: SQLite default (`DATABASE_URL=sqlite:///trading_system.db`),
systemd unit in `deploy/trading-bot.service`, Flask dashboard on :5000.
Supabase/psycopg2 only activate when `DATABASE_URL` points at Postgres.

---

## Deploy to Railway (legacy option)

### 1. Create the project

1. Push this repo to GitHub.
2. [Railway](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
   Railway builds the `Dockerfile` (per `railway.toml`).
3. **Add PostgreSQL**: right-click canvas → **Database → Add PostgreSQL**.
   `DATABASE_URL` is injected automatically.

### 2. Environment variables

Paste from `.env.example`. Required for trading:

| Variable | Notes |
|---|---|
| `GROQ_API_KEY` | free at console.groq.com |
| `OANDA_API_KEY` / `OANDA_ENV=practice` | recommended broker on Railway (v20 REST) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | alerts + 21 operator commands |
| `SECRET_KEY` | `python -c "import secrets;print(secrets.token_hex(32))"` |
| `DEMO_MODE=true` | stays demo until the demo gate passes |
| `ML_MODEL_DIR=/data/models` | persistent volume mount point |

> **MT5 note:** the `MetaTrader5` package only runs on Windows/macOS. On
> Railway the bot automatically falls back to OANDA, then to shadow mode.

### cTrader broker (Linux-native, Kenya-friendly)

Preferred live/demo broker path on Linux servers (works with Pepperstone,
IC Markets, FXPIG — cTrader accounts):

1. Create an API application at <https://openapi.ctrader.com> (Applications
   → Add), set any redirect URI, note **Client ID + Secret**.
2. Put `CTRADER_CLIENT_ID` / `CTRADER_CLIENT_SECRET` in `.env`, then:

```bash
python -m execution.ctrader_connector geturl     # open URL, approve, copy ?code=
python -m execution.ctrader_connector token CODE # prints tokens -> paste into .env
python -m execution.ctrader_connector accounts   # lists your account ids
```

3. Set `CTRADER_ACCOUNT_ID` (and `CTRADER_ENV=live` only for a funded live
   account). Restart the bot — startup check 3 reports `cTrader ok`.

Broker chain: MT5 → OANDA → cTrader → internal paper broker (simulated
fills) → shadow. The first healthy broker wins; without any credentials the
built-in paper broker keeps the full trade lifecycle running on virtual
money.

---

### 3. Persistent volume

Service → **Settings → Volumes** → mount at `/data`. The ML model persists at
`/data/models/rf_model.joblib`.

### 4. Deploy & verify

Watch **Deployments → Logs** for the startup checklist:

```
startup check database              PASS schema ok, state readable
startup check broker                PASS OANDA reachable
startup check data_feeds            PASS EURUSD m15 ok via alpha_vantage
startup check groq_api              PASS groq reachable
...
startup checks: 10/10 passed
```

Open the service URL for the dashboard; health endpoint: `/health`.

### 5. TradingView alerts (optional external signals)

The bot accepts TradingView alerts via webhook and runs them through the same
research -> risk -> execution pipeline as native signals.

1. Set `TV_WEBHOOK_SECRET` (any random string) in the environment.
2. In TradingView, create an alert on your chart -> Notifications -> Webhook URL:
   `https://<your-app>/webhook/tv`
3. Alert message (raw JSON):

```json
{"secret": "<TV_WEBHOOK_SECRET>", "pair": "OANDA:EURUSD", "action": "buy",
 "price": 1.08450, "sl": 1.08200, "tp": 1.09000}
```

- `action`: `buy` / `sell` / `exit` (exit closes that pair's open trades)
- `pair` accepts exchange prefixes and aliases (`GOLD`->XAUUSD, `US100`->NAS100)
- optional `"strategy": "my_indicator"` labels the signal `tv_my_indicator`
- Locally the standalone receiver also listens on `TV_WEBHOOK_PORT` (default 5555).

Alerts are deduplicated (same pair/direction/price within 15 min) and rejected
without the correct secret. TradingView webhooks require a paid TV plan.

### 6. TradingView on the FREE plan: email bridge

No paid plan? Alert **emails are free**. The bot polls your inbox over IMAP
and feeds alert mails into the same pipeline:

1. Gmail: enable 2FA -> create an **App Password** (myaccount.google.com/apppasswords)
2. Set in `.env`: `BRIDGE_EMAIL_USER`, `BRIDGE_EMAIL_PASSWORD` (the app password),
   `BRIDGE_IMAP_HOST` (Gmail: `imap.gmail.com`)
3. TradingView alert -> Notifications -> **Email** (your inbox) with the SAME
   JSON message as the webhook (including the `secret` field)

The bridge marks mails as read, so alerts fire exactly once. Poll interval:
`BRIDGE_POLL_SEC` (default 20s).

---

## Upgrades (UPGRADES 1-9)

All modules are additions-only, toggleable via env vars, and degrade
gracefully when optional dependencies are missing:

| Upgrade | Module | Toggle | Optional deps |
|---|---|---|---|
| 1. Deep learning | `deep_learning/` (LSTM + XGBoost + RL, ensemble voting gate) | `DEEP_LEARNING_ENABLED` | tensorflow, xgboost, stable-baselines3 (NumPy/sklearn/bandit fallbacks) |
| 2. NLP sentiment | `nlp/` (FinBERT, central banks, Twitter, Google Trends) | `NLP_ENABLED`, `GOOGLE_TRENDS_ENABLED` | transformers, pytrends, `TWITTER_BEARER_TOKEN` (lexicon fallback) |
| 3. Tick data | `tick/` (velocity, imbalance, stop hunts, velocity breaker) | `TICK_DATA_ENABLED` | MetaTrader5 (synthetic pseudo-ticks otherwise) |
| 4. Scalping | `scalping/scalping_engine.py` (kill zones, DL-gated) | `SCALPING_ENABLED` | - |
| 5. Stat arb | `arbitrage/stat_arb_engine.py` (correlation + triangular) | `STAT_ARB_ENABLED` | - |
| 6. Analytics | `analytics/` (Sharpe/Sortino/Calmar, MAE/MFE, quality) | always on | - |
| 7. Infrastructure | `infrastructure/` (Redis cache, hot reload, versioning, profiler, deps) | `REDIS_URL` (memory fallback) | redis, pytrends |
| 8. Smart SL | `risk/smart_sl_engine.py` (ATR/structure stops, dynamic TP) | always on | - |
| 9. Dashboard | `/api/upgrades` endpoint powering 7 new panels | always on | - |

Model files persist to `DL_MODELS_DIR` (default `models/`). The ensemble
voting gate runs on every candidate trade: all three models must agree on
direction, a skip vote requires extra research confluence, and per-model
accuracy below `DL_MIN_ACCURACY` (52%) over 50 predictions triggers
retraining. Hot reload: edit `config/runtime_overrides.json` to change risk
%, max trades, pairs or confluence live (audited + Telegram alert).

---

## Local development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill in keys
python main.py                    # engine + dashboard on :5000
pytest tests/ -q                  # unified test suite
pytest tests/ market-intelligence/tests/ strategy-engine/tests/ -q   # everything
```

Without broker keys the bot runs in **shadow mode**: full research pipeline and
event bus operate, orders are recorded but never sent.

---

## Telegram commands

`/status /trades /balance /pairs /research /stats /cot /sentiment /vix
/pause /resume /close /demo /live /enable /disable /backtest /kelly
/montecarlo /heatmap /version` — all chat-ID authorized; unauthorized access
is logged and rejected.

`/live` is hard-gated on both the backtest gate and demo gate passing.

---

## Dashboard sections

1. Status bar (state, session, regime, VIX) · 2. Account + equity curve ·
3. Open trades (10s refresh) · 4. Research live feed · 5. Institutional data
(COT/retail/DXY/VIX) · 6. Strategy performance · 7. Session heatmap ·
8. Monte Carlo risk · 9. ML model status · 10. Trade history w/ Groq reasoning.
Controls: pause / resume / close-all / run Monte Carlo.

---

## Disclaimer

Educational software provided as-is. Verify every component on **demo** first.
You are solely responsible for any trading losses.
