"""Historical data loader for backtesting: yfinance M15 candles.

yfinance provides at most ~60 days of 15-minute history per request (per
Yahoo's API limits), so the loader chunks requests across the requested range,
deduplicates, normalizes to the bot's standard OHLCV frame
(time/open/high/low/close/volume, UTC) and caches to Parquet-free CSV.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from config import settings
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["SYMBOL_MAP", "load_m15_history", "load_frames", "resample"]

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data_cache")

# bot pair -> yfinance symbol (indices use proxies: US30 via DJI index, NAS100 via NDX)
SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "GBPJPY": "GBPJPY=X",
    "EURJPY": "EURJPY=X",
    "XAUUSD": "GC=F",       # gold futures as proxy for spot gold
    "NAS100": "^NDX",
    "US30": "^DJI",
}

MAX_15M_DAYS = 59  # yfinance hard limit is 60 days per request


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's multi-index columns and standardize."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "close", "Volume": "volume",
    })
    keep = ["open", "high", "low", "close", "volume"]
    for col in keep:
        if col not in df.columns:
            df[col] = 0.0
    df["volume"] = df["volume"].fillna(0.0)
    df = df[keep].reset_index()
    time_col = df.columns[0]
    df = df.rename(columns={time_col: "time"})
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["time"] = ts
    df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    # collapse duplicated column names (Close + Adj Close both -> close)
    df = df.loc[:, ~df.columns.duplicated()]
    return df[["time", "open", "high", "low", "close", "volume"]]


def _cache_path(pair: str) -> str:
    return os.path.join(CACHE_DIR, f"{pair.upper()}_m15.csv")


def load_m15_history(pair: str, days: int = 55,
                     use_cache: bool = True) -> pd.DataFrame:
    """Load M15 candles for a pair over the last ``days`` days (max 59)."""
    days = min(days, MAX_15M_DAYS)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = _cache_path(pair)

    if use_cache and os.path.exists(cache):
        age_hours = (datetime.now().timestamp() - os.path.getmtime(cache)) / 3600.0
        if age_hours < 12:
            df = pd.read_csv(cache)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            logger.info("loaded %d cached m15 rows for %s", len(df), pair)
            return df

    symbol = SYMBOL_MAP.get(pair.upper(), f"{pair.upper()}=X")
    frames = []
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=MAX_15M_DAYS), end)
        try:
            import yfinance as yf
            raw = yf.download(symbol, start=cursor, end=chunk_end, interval="15m",
                              progress=False, auto_adjust=False)
            if raw is not None and len(raw):
                frames.append(_normalize(raw))
        except Exception as exc:
            logger.warning("yfinance chunk %s..%s failed for %s: %s",
                           cursor.date(), chunk_end.date(), pair, exc)
        cursor = chunk_end

    if not frames:
        raise RuntimeError(f"no historical data for {pair} ({symbol})")

    df = pd.concat(frames).drop_duplicates(subset=["time"]).sort_values("time")
    df = df.reset_index(drop=True)
    df.to_csv(cache, index=False)
    logger.info("downloaded %d m15 rows for %s (%.1f days)", len(df), pair, days)
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an M15 OHLCV frame to a higher timeframe (1h/4h/1D)."""
    ts = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True))
    out = df.set_index(ts).resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"}).dropna()
    out = out.reset_index()
    out = out.rename(columns={"index": "time"})  # some pandas versions name it 'index'
    return out[["time", "open", "high", "low", "close", "volume"]]


def load_frames(pair: str, days: int = 55, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Load the standard frame bundle (m1 absent in backtests; m15 is primary)."""
    m15 = load_m15_history(pair, days, use_cache)
    return {
        "m15": m15,
        "h1": resample(m15, "1h"),
        "h4": resample(m15, "4h"),
        "d1": resample(m15, "1D"),
    }
