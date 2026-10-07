"""Unified trading system configuration (master prompt).

Single source of truth for every module. Values come from environment
variables (python-dotenv) with production-safe defaults.
"""

import os

from dotenv import load_dotenv

for _candidate in (
    os.path.join(os.path.dirname(__file__), "..", ".env"),
    os.path.join(os.path.dirname(__file__), ".env"),
    ".env",
):
    if os.path.exists(_candidate):
        load_dotenv(_candidate)
        break
else:
    load_dotenv()


def env_str(name: str, default: str = "") -> str:
    """Read a string env var."""
    return os.getenv(name, default).strip()


def env_int(name: str, default: int) -> int:
    """Read an int env var with fallback."""
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    """Read a float env var with fallback."""
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    """Read a bool env var (1/true/yes)."""
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def env_list(name: str, default: str) -> list[str]:
    """Read a comma-separated env var."""
    return [p.strip() for p in os.getenv(name, default).split(",") if p.strip()]


# ---- identity / deployment ----
ENVIRONMENT = env_str("ENVIRONMENT", "production")
DATABASE_URL = env_str("DATABASE_URL", "sqlite:///trading_system.db").replace(
    "postgres://", "postgresql://", 1
)
PORT = env_int("PORT", 5000)
SECRET_KEY = env_str("SECRET_KEY", "dev-secret-change-me")
TZ_DISPLAY_OFFSET_HOURS = env_float("TZ_DISPLAY_OFFSET_HOURS", 3.0)  # EAT = UTC+3

# ---- brokers ----
MT5_LOGIN = env_str("MT5_LOGIN")
MT5_PASSWORD = env_str("MT5_PASSWORD")
MT5_SERVER = env_str("MT5_SERVER")
OANDA_API_KEY = env_str("OANDA_API_KEY")
OANDA_ACCOUNT_ID = env_str("OANDA_ACCOUNT_ID")
OANDA_ENV = env_str("OANDA_ENV", "practice")

# ---- internal paper broker (brokerless demo trading) ----
PAPER_BROKER_ENABLED = env_bool("PAPER_BROKER_ENABLED", True)
PAPER_START_BALANCE = env_float("PAPER_START_BALANCE", 10000.0)

# ---- cTrader Open API (Linux-native broker for Pepperstone/IC Markets etc.) ----
CTRADER_CLIENT_ID = env_str("CTRADER_CLIENT_ID")
CTRADER_CLIENT_SECRET = env_str("CTRADER_CLIENT_SECRET")
CTRADER_REDIRECT_URI = env_str("CTRADER_REDIRECT_URI", "https://openapi.ctrader.com/apps")
CTRADER_ACCESS_TOKEN = env_str("CTRADER_ACCESS_TOKEN")
CTRADER_REFRESH_TOKEN = env_str("CTRADER_REFRESH_TOKEN")
CTRADER_ACCOUNT_ID = env_str("CTRADER_ACCOUNT_ID")
CTRADER_ENV = env_str("CTRADER_ENV", "demo")

# ---- AI ----
GROQ_API_KEY = env_str("GROQ_API_KEY")
GROQ_MODEL = env_str("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_VERIFY_THRESHOLD = env_int("GROQ_VERIFY_THRESHOLD", 82)
GROQ_MIN_CONVICTION = env_int("GROQ_MIN_CONVICTION", 74)
GROQ_MAX_REJECTIONS = env_int("GROQ_MAX_REJECTIONS", 5)

# ---- news / data feeds ----
NEWS_API_KEY = env_str("NEWS_API_KEY")
# NOTE: Alpha Vantage REMOVED as a data feed (25 req/day free tier caused a
# 54+ hour feed outage). Feed chain: MT5 (primary, guarded) -> yfinance
# (secondary) -> Twelve Data (tertiary).
TWELVE_DATA_API_KEY = env_str("TWELVE_DATA_API_KEY")
FRED_API_KEY = env_str("FRED_API_KEY")
MYFXBOOK_API_KEY = env_str("MYFXBOOK_API_KEY")
FOREXFACTORY_RSS = env_str(
    "FOREXFACTORY_RSS", "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
)
REUTERS_RSS = env_str("REUTERS_RSS", "https://news.google.com/rss/search?q=reuters+forex")
BLOOMBERG_RSS = env_str("BLOOMBERG_RSS", "https://news.google.com/rss/search?q=bloomberg+markets")
FXSTREET_RSS = env_str("FXSTREET_RSS", "https://news.google.com/rss/search?q=fxstreet+forex")
CFTC_API_BASE = env_str("CFTC_API_BASE", "https://publicreporting.cftc.gov/resource")

# ---- Telegram ----
TELEGRAM_BOT_TOKEN = env_str("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = env_str("TELEGRAM_CHAT_ID")

# ---- trading gates ----
DEMO_MODE = env_bool("DEMO_MODE", True)
MIN_CONFLUENCE = env_int("MIN_CONFLUENCE", 8)
MAX_CONFLUENCE = env_int("MAX_CONFLUENCE", 10)

# ---- first-trade pilot (one-time; auto-expires on the first fill) ----
# While the trades table is empty, a setup short of MIN_CONFLUENCE may trade IF
# the Groq consensus conviction is exceptional. The expiry condition is the
# database state itself (no trade rows -> active; first row -> voided forever),
# so there is no flag to clean up. Steady-state gates are untouched.
FIRST_TRADE_PILOT_ENABLED = env_bool("FIRST_TRADE_PILOT_ENABLED", False)
FIRST_TRADE_PILOT_CONFLUENCE = env_int("FIRST_TRADE_PILOT_CONFLUENCE", 7)
FIRST_TRADE_PILOT_CONVICTION = env_int("FIRST_TRADE_PILOT_CONVICTION", 80)

# ---- TEMPORARY trade-unblock window (operator-ordered, auto-expires) ----
# One env-controlled window loosens THREE gates at once: confluence floor,
# Groq conviction floor, Groq verifier score floor, and suppresses all
# circuit breakers EXCEPT the capital-protection allowlist. Expiry is
# computed per call from TEMP_WINDOW_STARTED_AT + TEMP_WINDOW_HOURS, so the
# window ends by itself with no cleanup commit or restart.
TEMP_WINDOW_HOURS = env_float("TEMP_WINDOW_HOURS", 24.0)
TEMP_WINDOW_STARTED_AT = env_str("TEMP_WINDOW_STARTED_AT", "")  # ISO UTC
TEMP_MIN_CONFLUENCE = env_int("TEMP_MIN_CONFLUENCE", 0)   # 0 = inactive
TEMP_GROQ_MIN_CONVICTION = env_int("TEMP_GROQ_MIN_CONVICTION", 0)
TEMP_GROQ_VERIFY_THRESHOLD = env_int("TEMP_GROQ_VERIFY_THRESHOLD", 0)
# breakers that STAY active during the window (capital protection only)
TEMP_BREAKERS_ALLOWLIST = env_str(
    "TEMP_BREAKERS_ALLOWLIST", "daily_loss,drawdown_halt,margin_halt")


def _temp_window_active() -> bool:
    """True while the temporary unblock window is live."""
    if not TEMP_WINDOW_STARTED_AT:
        return False
    try:
        from datetime import datetime, timezone

        start = datetime.fromisoformat(TEMP_WINDOW_STARTED_AT)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        return 0.0 <= elapsed < TEMP_WINDOW_HOURS * 3600.0
    except Exception:
        return False


def temp_min_confluence() -> int:
    """Temporary confluence floor cap (0 = no cap)."""
    return TEMP_MIN_CONFLUENCE if _temp_window_active() else 0


def temp_groq_min_conviction() -> int:
    """Temporary Groq consensus conviction floor (0 = steady state)."""
    return TEMP_GROQ_MIN_CONVICTION if _temp_window_active() else 0


def temp_groq_verify_threshold() -> int:
    """Temporary verifier score floor (0 = steady state)."""
    return TEMP_GROQ_VERIFY_THRESHOLD if _temp_window_active() else 0


def effective_groq_verify_threshold() -> int:
    """Verifier floor actually in force right now."""
    temp = temp_groq_verify_threshold()
    return temp if temp else GROQ_VERIFY_THRESHOLD


def temp_breaker_allowlist() -> list[str]:
    """Breakers that remain ACTIVE during the window ([] = window closed)."""
    if not _temp_window_active():
        return []
    return [s.strip() for s in TEMP_BREAKERS_ALLOWLIST.split(",") if s.strip()]
MAX_DAILY_LOSS_PCT = env_float("MAX_DAILY_LOSS_PCT", 3.0)
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 1.0)
MAX_RISK_PER_TRADE_PCT = env_float("MAX_RISK_PER_TRADE_PCT", 1.5)
MIN_RISK_PER_TRADE_PCT = env_float("MIN_RISK_PER_TRADE_PCT", 0.25)
MAX_OPEN_TRADES = env_int("MAX_OPEN_TRADES", 2)
MAX_TOTAL_OPEN_RISK_PCT = env_float("MAX_TOTAL_OPEN_RISK_PCT", 3.0)
MAX_TRADES_PER_DAY = env_int("MAX_TRADES_PER_DAY", 8)
MIN_ACCOUNT_BALANCE = env_float("MIN_ACCOUNT_BALANCE", 50.0)
KELLY_CAP_PCT = env_float("KELLY_CAP_PCT", 1.5)
KELLY_FLOOR_PCT = env_float("KELLY_FLOOR_PCT", 0.25)
# Absolute per-trade lots cap regardless of risk math. A tiny SL (e.g. Groq
# returning a 2-4 pip stop) otherwise explodes lots_for_risk into 1.5-30 lot
# trades ($250k+ notional on a $10k account). 0.75 lots at the worst observed
# 2-pip SL loses ~1.4% — still inside MAX_RISK_PER_TRADE_PCT.
MAX_ABS_LOTS = env_float("MAX_ABS_LOTS", 0.75)
# Reject stops tighter than N x modeled spread: a stop inside transaction
# noise is nearly guaranteed to be tagged by spread/slippage, and risk-based
# sizing on it explodes lots. 3x keeps strategy structural stops (8-20 pips)
# fully live while killing the 2-4 pip Groq stops (EURJPY #4).
MIN_SL_SPREAD_MULT = env_float("MIN_SL_SPREAD_MULT", 3.0)
# A strategy stop derived from a structural level (EMA/band) can land on the
# WRONG side of entry after a deep pullback (GBPJPY #5: buy with SL 21.9 pips
# ABOVE entry, broker "stopped out" in profit). When research must fall back
# to the structural stop, it re-anchors it this many ATR(M15) away on the
# correct side instead of trusting a nonsensical level.
STRUCTURAL_SL_BUFFER_MULT = env_float("STRUCTURAL_SL_BUFFER_MULT", 2.5)
# A deferred limit fill (price never returned to the zone) is re-armed for
# this many minutes; after that the setup is stale and dropped (NAS100 #428:
# confluence-9 approval lost to an OOM restart with no pending-fill state).
PENDING_LIMIT_TTL_MIN = env_float("PENDING_LIMIT_TTL_MIN", 45.0)

# ---- selector ----
SELECTOR_MIN_SCORE = env_int("SELECTOR_MIN_SCORE", 70)
SELECTOR_RETRAIN_TRADES = env_int("SELECTOR_RETRAIN_TRADES", 25)

# ---- circuit breakers ----
# M15 candles close every 15 min; Yahoo/broker feeds legitimately lag several
# minutes behind. A feed is "truly stale" only after 10 min without updates.
STALENESS_LIMIT_SEC = env_int("STALENESS_LIMIT_SEC", 600)
# When the cloud (Render) instance is the primary trader, the local copy runs
# in shadow mode: full monitoring/research/alerts, but NEW trades disabled so
# the two engines never double-trade the same signal.
LOCAL_SHADOW_MODE = env_bool("LOCAL_SHADOW_MODE", False)
# Feed dead this long (continuously) = full HALT (before that: observation).
FEED_DEAD_LIMIT_SEC = env_int("FEED_DEAD_LIMIT_SEC", 1800)
VIX_HALT = env_float("VIX_HALT", 40.0)
VIX_REDUCE_60 = env_float("VIX_REDUCE_60", 30.0)
VIX_REDUCE_30 = env_float("VIX_REDUCE_30", 20.0)
ATR_EXPLOSION_MULT = env_float("ATR_EXPLOSION_MULT", 2.5)
MARGIN_ALERT_PCT = env_float("MARGIN_ALERT_PCT", 500.0)
MARGIN_REDUCE_PCT = env_float("MARGIN_REDUCE_PCT", 300.0)
MARGIN_HALT_PCT = env_float("MARGIN_HALT_PCT", 200.0)
DRAWDOWN_HALT_PCT = env_float("DRAWDOWN_HALT_PCT", 10.0)

# ---- pairs (priority order from the master prompt) ----
TRADING_PAIRS = env_list(
    "TRADING_PAIRS",
    "EURUSD,GBPUSD,XAUUSD,USDJPY,GBPJPY,EURJPY,NAS100,US30,USDCHF,AUDUSD",
)

# ---- engine cadence (seconds) ----
CYCLE_INTERVAL_SEC = env_int("CYCLE_INTERVAL_SEC", 60)
MANAGE_INTERVAL_SEC = env_int("MANAGE_INTERVAL_SEC", 60)
HEALTH_INTERVAL_SEC = env_int("HEALTH_INTERVAL_SEC", 180)

# ---- ML ----
ML_MODEL_DIR = env_str("ML_MODEL_DIR", "/data/models")
ML_RETRAIN_EVERY = env_int("ML_RETRAIN_EVERY", 25)
ML_LOSS_SKIP_THRESHOLD = env_float("ML_LOSS_SKIP_THRESHOLD", 0.60)
ML_MIN_TRAIN = env_int("ML_MIN_TRAIN", 30)

# ---- backtest gates ----
BACKTEST_GATE_MIN_STRATEGIES = env_int("BACKTEST_GATE_MIN_STRATEGIES", 5)
BACKTEST_GATE_WIN_RATE = env_float("BACKTEST_GATE_WIN_RATE", 45.0)
BACKTEST_GATE_PROFIT_FACTOR = env_float("BACKTEST_GATE_PROFIT_FACTOR", 1.2)
DEMO_GATE_TRADES = env_int("DEMO_GATE_TRADES", 60)
DEMO_GATE_WIN_RATE = env_float("DEMO_GATE_WIN_RATE", 52.0)
DEMO_GATE_PROFIT_FACTOR = env_float("DEMO_GATE_PROFIT_FACTOR", 1.3)

# ---- TradingView webhook ----
TV_WEBHOOK_SECRET = env_str("TV_WEBHOOK_SECRET")
TV_WEBHOOK_PORT = env_int("TV_WEBHOOK_PORT", 5555)

# ---- TradingView email bridge (free-plan alternative to webhooks) ----
BRIDGE_IMAP_HOST = env_str("BRIDGE_IMAP_HOST", "imap.gmail.com")
BRIDGE_EMAIL_USER = env_str("BRIDGE_EMAIL_USER")
BRIDGE_EMAIL_PASSWORD = env_str("BRIDGE_EMAIL_PASSWORD")
BRIDGE_IMAP_FOLDER = env_str("BRIDGE_IMAP_FOLDER", "INBOX")
BRIDGE_POLL_SEC = env_int("BRIDGE_POLL_SEC", 20)
BRIDGE_ALLOWED_SENDERS = env_str(
    "BRIDGE_ALLOWED_SENDERS",
    "tradingview.com,investing.com,finviz.com,stockcharts.com")

# ---- upgrade toggles ----
DEEP_LEARNING_ENABLED = env_bool("DEEP_LEARNING_ENABLED", True)
NLP_ENABLED = env_bool("NLP_ENABLED", True)
TICK_DATA_ENABLED = env_bool("TICK_DATA_ENABLED", True)
SCALPING_ENABLED = env_bool("SCALPING_ENABLED", True)
STAT_ARB_ENABLED = env_bool("STAT_ARB_ENABLED", True)
GOOGLE_TRENDS_ENABLED = env_bool("GOOGLE_TRENDS_ENABLED", True)

# deep learning
DL_MODELS_DIR = env_str("DL_MODELS_DIR", "models")
DL_RETRAIN_CANDLES = env_int("DL_RETRAIN_CANDLES", 200)
DL_RETRAIN_TRADES = env_int("DL_RETRAIN_TRADES", 30)
DL_MIN_ACCURACY = env_float("DL_MIN_ACCURACY", 52.0)

# scalping
SCALP_RISK_PCT = env_float("SCALP_RISK_PCT", 0.5)
SCALP_MAX_PER_SESSION = env_int("SCALP_MAX_PER_SESSION", 5)
SCALP_TP_PIPS = (8.0, 12.0)
SCALP_SL_PIPS = (5.0, 7.0)
SCALP_TIME_LIMIT_MIN = 15

# stat arb
STATARB_MAX_POSITIONS = env_int("STATARB_MAX_POSITIONS", 1)
STATARB_RISK_PCT = env_float("STATARB_RISK_PCT", 0.5)
STATARB_MAX_HOLD_MIN = 240

# nlp
TWITTER_BEARER_TOKEN = env_str("TWITTER_BEARER_TOKEN")
HUGGINGFACE_TOKEN = env_str("HUGGINGFACE_TOKEN")

# infrastructure
REDIS_URL = env_str("REDIS_URL")
HOT_RELOAD_SEC = env_int("HOT_RELOAD_SEC", 30)
PROFILE_BUDGETS = {"data_fetch": 5.0, "indicators": 2.0, "groq": 15.0,
                   "total_research": 90.0}

# ---- dashboard ----
DASHBOARD_REFRESH_SEC = env_int("DASHBOARD_REFRESH_SEC", 30)
