"""Intermarket analyzer: bonds, equities, commodities, VIX and synthetic DXY.

Produces a per-currency bias score (-100..+100) consumed by the research engine
and passed to Groq. Runs hourly; failures degrade to neutral scores gracefully.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["IntermarketSnapshot", "IntermarketAnalyzer"]

FRED_SERIES = {"yield_10y": "DGS10", "yield_2y": "DGS2"}
VIX_REDUCE_30 = settings.VIX_REDUCE_30
VIX_REDUCE_60 = settings.VIX_REDUCE_60
VIX_HALT = settings.VIX_HALT


@dataclass
class IntermarketSnapshot:
    """Combined cross-market state."""

    usd_bias: float = 0.0
    eur_bias: float = 0.0
    gbp_bias: float = 0.0
    jpy_bias: float = 0.0
    xau_bias: float = 0.0
    aud_bias: float = 0.0
    dxy_trend: str = "neutral"
    vix: float = 0.0
    vix3m: float = 0.0
    yield_10y: float = 0.0
    yield_2y: float = 0.0
    yield_spread: float = 0.0
    spx_direction: str = "flat"
    vix_position_scale: float = 1.0   # position size multiplier from VIX
    vix_halt: bool = False
    notes: list[str] = field(default_factory=list)

    def bias_for(self, currency: str) -> float:
        """Bias score for a currency or 'XAU'."""
        return float(getattr(self, f"{currency.lower()}_bias", 0.0))

    def pair_bias(self, pair: str) -> float:
        """Base-bias minus quote-bias for a pair (XAUUSD -> XAU vs USD)."""
        pair = pair.upper()
        if pair == "XAUUSD":
            return self.bias_for("XAU") - self.bias_for("USD")
        if pair in ("NAS100", "US30"):
            return self.bias_for("USD") * -0.5
        base, quote = pair[:3], pair[3:]
        return self.bias_for(base) - self.bias_for(quote)

    def as_dict(self) -> dict:
        """Serialize for DB and Groq prompt."""
        return {
            "usd_bias": self.usd_bias, "eur_bias": self.eur_bias,
            "gbp_bias": self.gbp_bias, "jpy_bias": self.jpy_bias,
            "xau_bias": self.xau_bias, "aud_bias": self.aud_bias,
            "dxy_trend": self.dxy_trend, "vix": self.vix, "vix3m": self.vix3m,
            "yield_10y": self.yield_10y, "yield_2y": self.yield_2y,
            "yield_spread": self.yield_spread, "spx_direction": self.spx_direction,
            "vix_position_scale": self.vix_position_scale, "vix_halt": self.vix_halt,
        }


class IntermarketAnalyzer:
    """Fetches cross-market data and computes currency bias scores."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self._cache: Optional[tuple[float, IntermarketSnapshot]] = None
        self._ttl = 3600.0

    # ---- fetch helpers with backoff ----

    def _get_json(self, url: str, params: dict, retries: int = 3) -> Optional[dict]:
        """GET JSON with exponential backoff; None on failure."""
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                logger.warning("GET %s attempt %d failed: %s", url.split('/')[2], attempt, exc)
                time.sleep(2 ** attempt)
        return None

    def fetch_yields(self) -> tuple[float, float]:
        """(10y, 2y) Treasury yields from FRED."""
        if not settings.FRED_API_KEY:
            return 0.0, 0.0
        out = []
        for key, series in FRED_SERIES.items():
            data = self._get_json(
                "https://api.stlouisfed.org/fred/series/observations",
                {"series_id": series, "api_key": settings.FRED_API_KEY,
                 "file_type": "json", "sort_order": "desc", "limit": 5},
            )
            value = 0.0
            if data:
                for obs in reversed(data.get("observations", [])):
                    try:
                        value = float(obs["value"])
                        break
                    except (KeyError, ValueError):
                        continue
            out.append(value)
        return out[0], out[1]

    def fetch_series_last(self, symbol: str, n: int = 25) -> list[float]:
        """Last n closes of a daily series from Alpha Vantage (empty on failure)."""
        if not settings.ALPHA_VANTAGE_API_KEY:
            return []
        data = self._get_json(
            "https://www.alphavantage.co/query",
            {"function": "TIME_SERIES_DAILY", "symbol": symbol,
             "outputsize": "compact", "apikey": settings.ALPHA_VANTAGE_API_KEY},
        )
        if not data:
            return []
        key = next((k for k in data if "Time Series" in str(k)), None)
        if key is None:
            return []
        series = data[key]
        closes = []
        for ts in sorted(series.keys())[-n:]:
            try:
                closes.append(float(series[ts]["4. close"]))
            except (KeyError, ValueError, TypeError):
                continue
        return closes

    def fetch_vix(self) -> tuple[float, float]:
        """(VIX, VIX3M) latest values."""
        vix = self.fetch_series_last("VIX", 5)
        vix3m = self.fetch_series_last("VIX3M", 5)
        return (vix[-1] if vix else 0.0, vix3m[-1] if vix3m else 0.0)

    # ---- main ----

    def snapshot(self, force: bool = False) -> IntermarketSnapshot:
        """Build the combined snapshot (cached 1 hour)."""
        now = time.monotonic()
        if not force and self._cache and now - self._cache[0] < self._ttl:
            return self._cache[1]

        snap = IntermarketSnapshot()
        y10, y2 = self.fetch_yields()
        snap.yield_10y, snap.yield_2y = y10, y2
        snap.yield_spread = round(y10 - y2, 3)
        snap.vix, snap.vix3m = self.fetch_vix()

        spx = self.fetch_series_last("SPY", 6)
        if len(spx) >= 2:
            snap.spx_direction = "up" if spx[-1] > spx[-2] else "down"
        oil = self.fetch_series_last("CL=F", 6)
        copper = self.fetch_series_last("HG=F", 6)

        # --- USD bias from yields ---
        if y10 and y2:
            if y10 > y2:  # positive spread + rising yields = USD supportive
                snap.usd_bias += 20 if snap.yield_spread > 0 else 0
            else:
                snap.usd_bias -= 20  # inverted curve = risk-off, eventual USD weakness
        if spx and len(spx) >= 6:
            spx_change = (spx[-1] / spx[-6] - 1) * 100
            if spx_change > 1:
                snap.usd_bias -= 10  # risk-on weakens USD vs safe havens
                snap.jpy_bias -= 25; snap.xau_bias -= 15
                snap.gbp_bias += 20; snap.aud_bias += 25; snap.eur_bias += 15
            elif spx_change < -1:
                snap.usd_bias += 15  # risk-off USD bid
                snap.jpy_bias += 30; snap.xau_bias += 20
                snap.gbp_bias -= 20; snap.aud_bias -= 25; snap.eur_bias -= 15
        if oil and len(oil) >= 6:
            oil_change = (oil[-1] / oil[-6] - 1) * 100
            if oil_change > 3:
                snap.aud_bias += 10
            elif oil_change < -3:
                snap.aud_bias -= 10
        if copper and len(copper) >= 6:
            cu_change = (copper[-1] / copper[-6] - 1) * 100
            if cu_change > 3:
                snap.aud_bias += 10; snap.jpy_bias -= 10
            elif cu_change < -3:
                snap.aud_bias -= 10; snap.jpy_bias += 10

        # --- synthetic DXY trend (EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2%, CHF 3.6%) ---
        eurusd = self.fetch_series_last("EURUSD=X", 30)
        if len(eurusd) >= 10:
            trend_slope = float(np.polyfit(range(10), eurusd[-10:], 1)[0])
            snap.dxy_trend = "falling" if trend_slope > 0 else "rising" if trend_slope < 0 else "neutral"
            if snap.dxy_trend == "falling":   # EUR up = DXY down
                snap.usd_bias -= 15; snap.eur_bias += 15
            elif snap.dxy_trend == "rising":
                snap.usd_bias += 15; snap.eur_bias -= 15

        # --- VIX position scaling ---
        if snap.vix >= VIX_HALT:
            snap.vix_halt = True
            snap.vix_position_scale = 0.0
            snap.notes.append("VIX >= 40: all trading halted")
        elif snap.vix >= VIX_REDUCE_60:
            snap.vix_position_scale = 0.4
            snap.notes.append("VIX 30-40: sizes -60%, Gold/JPY only recommended")
        elif snap.vix >= VIX_REDUCE_30:
            snap.vix_position_scale = 0.7
            snap.notes.append("VIX 20-30: sizes -30%")

        for k in ("usd_bias", "eur_bias", "gbp_bias", "jpy_bias", "xau_bias", "aud_bias"):
            setattr(snap, k, round(max(-100.0, min(100.0, getattr(snap, k))), 1))

        self._persist(snap)
        self._cache = (now, snap)
        logger.info("intermarket snapshot: USD %.0f EUR %.0f JPY %.0f VIX %.1f scale %.1f",
                    snap.usd_bias, snap.eur_bias, snap.jpy_bias, snap.vix, snap.vix_position_scale)
        return snap

    @staticmethod
    def _persist(snap: IntermarketSnapshot) -> None:
        """Store snapshot for the dashboard."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK, _utcnow
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext(
                "INSERT INTO intermarket_snapshots (created_at, usd_bias, eur_bias, gbp_bias, "
                "jpy_bias, xau_bias, dxy_trend, vix, vix3m, yield_10y, yield_2y, yield_spread, "
                "spx_direction) VALUES (:t, :usd, :eur, :gbp, :jpy, :xau, :dxy, :vix, :vix3m, "
                ":y10, :y2, :spread, :spx)"
            ), {"t": db._utcnow(), "usd": snap.usd_bias, "eur": snap.eur_bias,
                "gbp": snap.gbp_bias, "jpy": snap.jpy_bias, "xau": snap.xau_bias,
                "dxy": snap.dxy_trend, "vix": snap.vix, "vix3m": snap.vix3m,
                "y10": snap.yield_10y, "y2": snap.yield_2y,
                "spread": snap.yield_spread, "spx": snap.spx_direction})
