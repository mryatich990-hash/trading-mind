"""Unified market data engine: MT5 -> yfinance -> Twelve Data failover.

Feed chain (primary first):
  1. MT5 terminal feed  - free, unlimited, broker-direct (guarded import;
     skipped automatically on hosts without the MetaTrader5 package, e.g.
     Linux/Render).
  2. YFinance feed      - free, no API key, no hard rate limit (TTL-cached).
  3. Twelve Data        - key-limited (800/day free), used only as a last
     resort for confirmation.

Alpha Vantage has been REMOVED entirely (25 req/day free tier killed the
feed for 54+ hours). Every candle batch is validated (freshness, OHLC logic,
volume, freeze, anomalies) before use; feed switches are audited and the
current feed status is exposed via get_feed_status() for AutoRecovery.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["FeedError", "CandleData", "MarketDataEngine", "validate_ohlcv",
           "pip_size", "fx_market_closed"]

TD_SYMBOLS = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "XAUUSD": "XAU/USD",
              "USDJPY": "USD/JPY", "GBPJPY": "GBP/JPY", "EURJPY": "EUR/JPY",
              "USDCHF": "USD/CHF", "AUDUSD": "AUD/USD"}


def pip_size(pair: str) -> float:
    """Pip size per instrument (indices use 1.0 point)."""
    pair = pair.upper()
    if pair in ("USDJPY", "GBPJPY", "EURJPY"):
        return 0.01
    if pair == "XAUUSD":
        return 0.1
    if pair in ("NAS100", "US30"):
        return 1.0
    return 0.0001


class FeedError(Exception):
    """Feed failure or invalid payload."""


def fx_market_closed(now: Optional[datetime] = None) -> bool:
    """True inside the weekend FX close (Fri 20:00 UTC - Sun 21:00 UTC).

    Mirrors risk/circuit_breakers.py's weekend window so the data layer and
    the breakers agree on when stale candles are expected, not alarming.
    """
    t = now or datetime.now(timezone.utc)
    weekday, hour = t.weekday(), t.hour
    return ((weekday == 4 and hour >= 20) or weekday in (5, 6)
            or (weekday == 6 and hour < 21))


def validate_ohlcv(df: pd.DataFrame, pair: str, max_age_sec: float) -> Optional[str]:
    """Validate candles: freshness, OHLC logic, volume, freeze, anomalies."""
    if df is None or len(df) < 30:
        return "insufficient rows"
    market_closed = fx_market_closed()
    last_ts = pd.Timestamp(df["time"].iloc[-1])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(timezone.utc)
    age = (datetime.now(timezone.utc) - last_ts.to_pydatetime()).total_seconds()
    if age > max_age_sec and not market_closed:
        return f"stale: last candle {age:.0f}s old"
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)) or np.any(h < l):
        return "OHLC logic violation"
    no_volume_feed = float(df["volume"].sum()) == 0.0
    if not no_volume_feed and np.all(df["volume"].to_numpy(float)[-6:] == 0):
        return "zero volume on recent candles"
    if np.any(~np.isfinite(o)) or np.any(~np.isfinite(h)) or np.any(~np.isfinite(l)) or np.any(~np.isfinite(c)):
        return "NaN/inf in OHLC"
    # data freeze: 3 identical consecutive candles (expected while closed)
    closes_tail = df["close"].tail(3).to_numpy(float)
    if len(set(closes_tail.tolist())) == 1 and not market_closed:
        return "data freeze detected"
    # anomaly: candle range > 5x 20-period average ATR
    from data.indicator_engine import atr
    if len(df) > 25:
        atr_avg = float(atr(df, 14).rolling(20).mean().iloc[-1])
        if atr_avg > 0:
            rng = (h[-1] - l[-1])
            if rng > 5 * atr_avg:
                return f"candle anomaly: range {rng:.5f} > 5x ATR {atr_avg:.5f}"
    return None


class MT5Feed:
    """PRIMARY feed: MetaTrader 5 terminal (broker-direct, unlimited).

    Guarded import: when the MetaTrader5 package is not installed (Linux,
    Render cloud), ``configured`` is False and the feed is skipped silently.
    """

    name = "mt5"

    @property
    def configured(self) -> bool:
        """False when the MetaTrader5 package is unavailable on this OS."""
        try:
            import MetaTrader5  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self) -> None:
        self.mt5 = None
        self._connected = False

    def _ensure(self) -> bool:
        """Initialize/login MT5 once (and re-initialize after a drop)."""
        if self._connected:
            return True
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:
            raise FeedError("MetaTrader5 unavailable") from exc
        kwargs = {"server": settings.MT5_SERVER} if settings.MT5_SERVER else {}
        if not mt5.initialize(**kwargs):
            raise FeedError(f"initialize failed: {mt5.last_error()}")
        if settings.MT5_LOGIN:
            if not mt5.login(int(settings.MT5_LOGIN), password=settings.MT5_PASSWORD,
                             server=settings.MT5_SERVER):
                raise FeedError(f"login failed: {mt5.last_error()}")
        self.mt5 = mt5
        self._connected = True
        return True

    def reconnect(self) -> bool:
        """Force a terminal re-initialization (AutoRecovery path)."""
        self._connected = False
        self.mt5 = None
        try:
            return self._ensure()
        except Exception:
            return False

    def connected(self) -> bool:
        """True when the terminal responds (AutoRecovery/health path)."""
        try:
            self._ensure()
            return self.mt5.terminal_info() is not None
        except Exception:
            return False

    def tick_age_sec(self, pair: str) -> Optional[float]:
        """Age of the last tick in seconds; None when unavailable."""
        try:
            self._ensure()
            tick = self.mt5.symbol_info_tick(pair.upper())
            if tick is None or not tick.time:
                return None
            return max(0.0, time.time() - tick.time)
        except Exception:
            return None

    def fetch(self, pair: str, timeframe_min: int, count: int) -> pd.DataFrame:
        """Fetch candles from the terminal."""
        self._ensure()
        tf_map = {1: "M1", 5: "M5", 15: "M15", 60: "H1", 240: "H4", 1440: "D1"}
        tf = getattr(self.mt5, f"TIMEFRAME_{tf_map.get(timeframe_min, 'M15')}")
        rates = self.mt5.copy_rates_from_pos(pair.upper(), tf, 0, max(count, 30))
        if rates is None or not len(rates):
            raise FeedError(f"no rates for {pair}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.rename(columns={"tick_volume": "volume"})[
            ["time", "open", "high", "low", "close", "volume"]
        ].tail(max(count, 30)).reset_index(drop=True)


class TwelveDataFeed:
    """TERTIARY feed: Twelve Data (800 requests/day free; last resort)."""

    name = "twelve_data"

    @property
    def configured(self) -> bool:
        """False when no API key is set."""
        return bool(settings.TWELVE_DATA_API_KEY)

    def __init__(self) -> None:
        self.session = requests.Session()

    def fetch(self, pair: str, timeframe_min: int, count: int) -> pd.DataFrame:
        """Fetch OHLCV rows."""
        if not settings.TWELVE_DATA_API_KEY:
            raise FeedError("no API key")
        symbol = TD_SYMBOLS.get(pair.upper())
        if not symbol:
            raise FeedError(f"no mapping for {pair}")
        resp = self.session.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": symbol, "interval": f"{timeframe_min}min",
                    "outputsize": min(max(count, 30), 5000),
                    "apikey": settings.TWELVE_DATA_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") == "error":
            raise FeedError(str(payload.get("message"))[:150])
        rows = [{"time": pd.Timestamp(v["datetime"], tz="UTC"),
                 "open": float(v["open"]), "high": float(v["high"]),
                 "low": float(v["low"]), "close": float(v["close"]),
                 "volume": float(v.get("volume", 1) or 1)}
                for v in reversed(payload.get("values", []))]
        if not rows:
            raise FeedError("empty values")
        return pd.DataFrame(rows).reset_index(drop=True)


class YFinanceFeed:
    """SECONDARY feed: Yahoo Finance (free, keyless, TTL-cached).

    15m candles exist for the last ~59 days only; higher timeframes map to
    longer lookbacks. FX symbols report volume=0, which validate_ohlcv may
    flag depending on its strictness.
    """

    name = "yfinance"

    _MAP = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
        "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X",
        "USDCAD": "USDCAD=X", "GBPJPY": "GBPJPY=X", "EURJPY": "EURJPY=X",
        "XAUUSD": "GC=F", "NAS100": "^NDX", "US30": "^DJI",
    }
    _INTERVALS = {1: ("1m", 7), 5: ("5m", 40), 15: ("15m", 59),
                  60: ("60m", 120), 240: ("60m", 180), 1440: ("1d", 730)}

    _CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
    _CACHE_TTL = 90.0  # seconds; keeps request volume far below Yahoo throttling

    def __init__(self) -> None:
        self.session = requests.Session()

    def fetch(self, pair: str, timeframe_min: int, count: int) -> pd.DataFrame:
        """Fetch OHLCV rows via yfinance download() (cached, one retry)."""
        import yfinance as yf

        symbol = self._MAP.get(pair.upper())
        if not symbol:
            raise FeedError(f"no yfinance mapping for {pair}")
        cache_key = f"{pair}_{timeframe_min}_{count}"
        now = time.monotonic()
        hit = YFinanceFeed._CACHE.get(cache_key)
        if hit is not None and now - hit[0] < YFinanceFeed._CACHE_TTL:
            return hit[1].copy()
        interval, days = self._INTERVALS.get(timeframe_min, ("15m", 59))
        if interval == "1m":
            days = min(days, 7)
        df = None
        for attempt in (1, 2):
            df = yf.download(symbol, period=f"{days}d", interval=interval,
                             progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                break
            if attempt == 1:
                time.sleep(2.0)  # Yahoo intermittently returns empty frames
        if df is None or df.empty:
            raise FeedError(f"yfinance returned no data for {pair}")
        if isinstance(df.columns, pd.MultiIndex):  # yfinance >= 0.2.5
            df.columns = df.columns.get_level_values(0)
        # Adj Close duplicates Close after renaming -> drop it first
        if "Adj Close" in df.columns:
            df = df.drop(columns=["Adj Close"])
        df = df.reset_index()  # index name: Datetime (intraday) / Date (daily)
        df = df.rename(columns={"Datetime": "time", "Date": "time", "index": "time",
                                "Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        missing = {"time", "open", "high", "low", "close"} - set(df.columns)
        if missing:
            raise FeedError(f"yfinance missing columns {sorted(missing)} for {pair}")
        ts = pd.to_datetime(df["time"], utc=True)
        out = pd.DataFrame({
            "time": ts,
            "open": pd.to_numeric(df["open"], errors="coerce").astype(float),
            "high": pd.to_numeric(df["high"], errors="coerce").astype(float),
            "low": pd.to_numeric(df["low"], errors="coerce").astype(float),
            "close": pd.to_numeric(df["close"], errors="coerce").astype(float),
            "volume": pd.to_numeric(df.get("volume"), errors="coerce").fillna(0).astype(float),
        }).dropna(subset=("open", "high", "low", "close"))
        # Yahoo FX daily bars occasionally contain garbage rows (high below
        # open/close etc.); drop them instead of failing validation wholesale.
        o = out["open"].to_numpy(float)
        h = out["high"].to_numpy(float)
        l = out["low"].to_numpy(float)
        c = out["close"].to_numpy(float)
        good = (h >= np.maximum(o, c)) & (l <= np.minimum(o, c)) & (h >= l)
        if good.any() and good.mean() < 0.75:
            raise FeedError(f"yfinance data too corrupt for {pair} "
                            f"({int((~good).sum())} bad rows)")
        out = out[good]
        out = out.tail(max(count, 30)).reset_index(drop=True)
        if out.empty:
            raise FeedError(f"yfinance empty frame for {pair}")
        # keep the cache bounded (only successful fetches are stored)
        if len(YFinanceFeed._CACHE) > 64:
            oldest = min(YFinanceFeed._CACHE, key=lambda k: YFinanceFeed._CACHE[k][0])
            del YFinanceFeed._CACHE[oldest]
        YFinanceFeed._CACHE[cache_key] = (now, out.copy())
        return out


@dataclass
class CandleData:
    """Validated candle bundle for one pair/timeframe."""

    pair: str
    timeframe_min: int
    df: pd.DataFrame
    source: str
    bid: float = 0.0
    ask: float = 0.0
    fetch_seconds: float = 0.0

    @property
    def last_close(self) -> float:
        """Latest close."""
        return float(self.df["close"].iloc[-1])

    @property
    def spread_pips(self) -> float:
        """Spread in pips (0 when unavailable)."""
        if self.ask <= 0 or self.bid <= 0:
            return 0.0
        return round((self.ask - self.bid) / pip_size(self.pair), 1)


class MarketDataEngine:
    """Failover engine with validation, switch auditing and status reporting."""

    def __init__(self, feeds: Optional[list] = None, failover_deadline_sec: float = 5.0) -> None:
        # PRIMARY -> SECONDARY -> TERTIARY. MT5 first: it is the free,
        # unlimited, broker-direct feed wherever the terminal exists (Windows
        # VPS / local Windows). On Linux hosts its `configured` is False and
        # the engine transparently leans on yfinance (secondary).
        self.feeds = feeds if feeds is not None else [
            MT5Feed(), YFinanceFeed(), TwelveDataFeed(),
        ]
        self.failover_deadline_sec = failover_deadline_sec
        self.consecutive_failures = 0
        self.active_feed = "none"
        self.last_update: Optional[datetime] = None
        self._latencies: list[float] = []
        self._cache: dict[str, CandleData] = {}

    # ---- status (consumed by AutoRecovery + /health) ----

    def get_feed_status(self) -> dict:
        """Current failover posture for monitoring/auto-recovery."""
        return {
            "active_feed": self.active_feed,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "consecutive_failures": self.consecutive_failures,
            "feeds_configured": [f.name for f in self.feeds
                                 if getattr(f, "configured", True)],
            "mt5_connected": self._mt5_connected(),
        }

    def _mt5_connected(self) -> bool:
        """MT5 terminal reachable (False when the package is unavailable)."""
        for feed in self.feeds:
            if feed.name == "mt5":
                try:
                    return bool(feed.connected())
                except Exception:
                    return False
        return False

    # ---- fetching with failover ----

    def get_candles(self, pair: str, timeframe_min: int = 15, count: int = 300) -> CandleData:
        """Fetch validated candles with feed failover inside the deadline."""
        deadline = time.monotonic() + self.failover_deadline_sec
        errors: list[str] = []
        freshness = max(settings.STALENESS_LIMIT_SEC, timeframe_min * 60 * 2)
        previous_source = self.active_feed
        for attempt in range(len(self.feeds)):
            feed = self.feeds[attempt % len(self.feeds)]
            if not getattr(feed, "configured", True):
                continue  # unconfigured feed: skip silently (no warning spam)
            started = time.monotonic()
            try:
                df = feed.fetch(pair, timeframe_min, count)
                err = validate_ohlcv(df, pair, freshness)
                if err:
                    raise FeedError(err)
                latency = time.monotonic() - started
                self._record_latency(latency)
                candle = CandleData(
                    pair=pair.upper(), timeframe_min=timeframe_min, df=df,
                    source=feed.name, fetch_seconds=round(latency, 3),
                )
                self.consecutive_failures = 0
                self._cache[f"{pair}_{timeframe_min}"] = candle
                self._mark_feed_ok(feed.name, previous_source)
                return candle
            except Exception as exc:
                errors.append(f"{feed.name}: {exc}")
                logger.warning("feed %s failed %s %dm: %s", feed.name, pair, timeframe_min, exc)
                if time.monotonic() >= deadline:
                    break
        self.consecutive_failures += 1
        db.log_feed_health("data_feeds", False, "; ".join(errors)[:255])
        cached = self._cache.get(f"{pair}_{timeframe_min}")
        if cached is not None:
            return cached
        raise FeedError(f"all feeds failed for {pair}: {errors}")

    def _mark_feed_ok(self, feed_name: str, previous_source: str) -> None:
        """Record the active feed; audit-log every feed switch to Supabase."""
        self.active_feed = feed_name
        self.last_update = datetime.now(timezone.utc)
        if previous_source != feed_name:
            logger.warning("FEED SWITCH: %s -> %s", previous_source, feed_name)
            try:
                db.audit("data", "feed_switch",
                         f"{previous_source} -> {feed_name} "
                         f"(failures={self.consecutive_failures})")
                db.log_feed_health("data_feeds", True, f"active feed {feed_name}")
            except Exception as exc:
                logger.debug("feed switch audit skipped: %s", exc)

    def _record_latency(self, seconds: float) -> None:
        """Track rolling latency; alert above 3s average."""
        self._latencies.append(seconds)
        if len(self._latencies) > 50:
            del self._latencies[:-50]
        avg = sum(self._latencies) / len(self._latencies)
        if avg > 3.0 and len(self._latencies) >= 10:
            logger.error("feed latency high: %.2fs average", avg)

    def get_price(self, pair: str) -> dict:
        """Latest bid/ask approximation from the last candle."""
        candle = self.get_candles(pair, 15, 50)
        spread = pip_size(pair)
        last = float(candle.last_close)
        return {"bid": round(last - spread / 2, 6), "ask": round(last + spread / 2, 6),
                "spread_pips": 1.0}

    def get_frames(self, pair: str) -> dict[str, pd.DataFrame]:
        """Fetch the standard timeframe bundle used by the research engine."""
        return {
            "m15": self.get_candles(pair, 15, 300).df,
            "h1": self.get_candles(pair, 60, 300).df,
            "h4": self.get_candles(pair, 240, 300).df,
            "d1": self.get_candles(pair, 1440, 260).df,
            "m1": self.get_candles(pair, 1, 60).df,
        }
