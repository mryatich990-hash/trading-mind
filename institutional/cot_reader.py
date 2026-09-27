"""COT report reader: CFTC Commitment of Traders as a directional bias filter.

Fetches weekly reports from the CFTC public API, tracks commercial /
non-commercial / retail net positioning per currency, computes 52-week
percentiles and produces alignment signals worth +15 confluence.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["COTSnapshot", "COTReader"]

CFTC_URL = f"{settings.CFTC_API_BASE}/6cfc513d-9549-46c5-9b64-1cf0472d23d9.json"
# futures contract -> currency
CONTRACT_CURRENCY = {
    "EURO FX": "EUR", "BRITISH POUND STERLING": "GBP",
    "JAPANESE YEN": "JPY", "SWISS FRANC": "CHF",
    "AUSTRALIAN DOLLAR": "AUD", "NEW ZEALAND DOLLAR": "NZD",
    "CANADIAN DOLLAR": "CAD",
}


@dataclass
class COTSnapshot:
    """Positioning snapshot for one currency."""

    currency: str
    report_date: str
    commercial_net: float
    noncommercial_net: float
    retail_net: float
    open_interest: float
    commercial_pctile: float   # 0-100 vs last 52 weeks
    noncommercial_pctile: float
    wow_change_commercial: float

    @property
    def bias(self) -> str:
        """'buy' / 'sell' / 'neutral' from the commercial-vs-spec extreme rule."""
        if self.commercial_pctile >= 80 and self.noncommercial_pctile <= 20:
            return "buy"
        if self.commercial_pctile <= 20 and self.noncommercial_pctile >= 80:
            return "sell"
        return "neutral"


class COTReader:
    """Fetches and scores CFTC positioning data."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self._cache: Optional[tuple[datetime, dict[str, COTSnapshot]]] = None
        self._cache_ttl = timedelta(hours=12)

    def fetch_report(self, retries: int = 3) -> pd.DataFrame:
        """Download the latest COT futures-only report with backoff."""
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(
                    CFTC_URL,
                    params={
                        "$select": "report_date_as_yyyy_mm_dd,market_and_exchange_names,"
                                   "open_interest_all,noncomm_positions_long_all,"
                                   "noncomm_positions_short_all,comm_positions_long_all,"
                                   "comm_positions_short_all,nonrept_positions_long_all,"
                                   "nonrept_positions_short_all",
                        "$limit": 200,
                        "$order": "report_date_as_yyyy_mm_dd DESC",
                    },
                    timeout=25,
                )
                resp.raise_for_status()
                rows = resp.json()
                if rows:
                    df = pd.DataFrame(rows)
                    logger.info("COT report fetched: %d rows", len(df))
                    return df
            except Exception as exc:
                logger.warning("COT fetch attempt %d failed: %s", attempt, exc)
                time.sleep(2 ** attempt)
        raise RuntimeError("COT report unavailable after retries")

    @staticmethod
    def _pctile(series: pd.Series, value: float) -> float:
        """Percentile rank of value within series (0-100)."""
        clean = series.dropna()
        if not len(clean):
            return 50.0
        return round(float((clean < value).mean() * 100.0), 1)

    def snapshots(self, force: bool = False) -> dict[str, COTSnapshot]:
        """Parse the latest report into per-currency snapshots (cached 12h)."""
        now = datetime.now(timezone.utc)
        if not force and self._cache and now - self._cache[0] < self._cache_ttl:
            return self._cache[1]

        df = self.fetch_report()
        df.columns = [c.lower() for c in df.columns]
        snapshots: dict[str, COTSnapshot] = {}
        for _, row in df.iterrows():
            name = str(row.get("market_and_exchange_names", "")).upper()
            currency = next((c for key, c in CONTRACT_CURRENCY.items() if key in name), None)
            if currency is None or currency in snapshots:
                continue
            try:
                comm_net = float(row["comm_positions_long_all"]) - float(row["comm_positions_short_all"])
                noncomm_net = (float(row["noncomm_positions_long_all"])
                               - float(row["noncomm_positions_short_all"]))
                retail_net = (float(row["nonrept_positions_long_all"])
                              - float(row["nonrept_positions_short_all"]))
                oi = float(row["open_interest_all"])
            except (KeyError, TypeError, ValueError):
                continue

            # history for percentiles (same contract across reports)
            history = df[df["market_and_exchange_names"] == row["market_and_exchange_names"]]
            hist_comm, hist_noncomm = [], []
            for _, hrow in history.iterrows():
                try:
                    hist_comm.append(float(hrow["comm_positions_long_all"])
                                     - float(hrow["comm_positions_short_all"]))
                    hist_noncomm.append(float(hrow["noncomm_positions_long_all"])
                                        - float(hrow["noncomm_positions_short_all"]))
                except (TypeError, ValueError):
                    continue
            wow = 0.0
            if len(hist_comm) > 1:
                wow = comm_net - hist_comm[1]

            snap = COTSnapshot(
                currency=currency,
                report_date=str(row.get("report_date_as_yyyy_mm_dd", ""))[:10],
                commercial_net=comm_net, noncommercial_net=noncomm_net,
                retail_net=retail_net, open_interest=oi,
                commercial_pctile=self._pctile(pd.Series(hist_comm), comm_net),
                noncommercial_pctile=self._pctile(pd.Series(hist_noncomm), noncomm_net),
                wow_change_commercial=wow,
            )
            snapshots[currency] = snap
            self._persist(snap)

        if snapshots:
            self._cache = (now, snapshots)
        logger.info("COT snapshots: %s", {k: v.bias for k, v in snapshots.items()})
        return snapshots

    @staticmethod
    def _persist(snap: COTSnapshot) -> None:
        """Upsert into cot_reports."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(sqltext(
                "DELETE FROM cot_reports WHERE report_date = :d AND currency = :c"
            ), {"d": snap.report_date, "c": snap.currency})
            conn.execute(sqltext(
                "INSERT INTO cot_reports (report_date, currency, commercial_net, "
                "noncommercial_net, retail_net, open_interest, commercial_pctile, "
                "noncommercial_pctile) VALUES (:d, :c, :com, :nc, :ret, :oi, :cp, :np)"
            ), {"d": snap.report_date, "c": snap.currency, "com": snap.commercial_net,
                "nc": snap.noncommercial_net, "ret": snap.retail_net,
                "oi": snap.open_interest, "cp": snap.commercial_pctile,
                "np": snap.noncommercial_pctile})

    def bias_for_pair(self, pair: str) -> str:
        """COT bias for a pair from base vs quote currency positioning.

        'buy' when base bias is buy (or quote bias is sell), 'sell' for the
        inverse, 'neutral' otherwise or when data is missing.
        """
        snaps = self.snapshots()
        pair = pair.upper()
        base, quote = pair[:3], pair[3:]
        if pair == "XAUUSD":
            base, quote = "XAU", "USD"
        if pair in ("NAS100", "US30"):
            return "neutral"
        base_bias = snaps[base].bias if base in snaps else "neutral"
        quote_bias = snaps[quote].bias if quote in snaps else "neutral"
        if base_bias == "buy" or quote_bias == "sell":
            return "buy"
        if base_bias == "sell" or quote_bias == "buy":
            return "sell"
        return "neutral"
