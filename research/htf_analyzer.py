"""HTFAnalyzer: Research Step 3 (higher timeframe bias) + Step 4 (institutional structure).

Daily and H4 must agree or no trade is possible. Step 4 collects order blocks,
FVGs, VPOC position, VWAP position, Wyckoff phase, Elliott wave count, harmonic
patterns and nearest unswept liquidity into one structure read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.logging_utils import get_logger
from data.market_data_engine import pip_size
from data.volume_profile import VolumeProfileEngine
from data.vwap_engine import VWAPEngine
from structure.harmonic_detector import HarmonicDetector
from structure.elliott_wave import ElliottWaveAnalyzer
from structure.pattern_detector import PatternDetector
from structure.smart_money_detector import SmartMoneyDetector
from structure.wyckoff_analyzer import WyckoffAnalyzer

logger = get_logger(__name__)

__all__ = ["HTFResult", "HTFAnalyzer"]


@dataclass
class HTFResult:
    """Step 3 + 4 output."""

    daily_trend: str = "range"        # up / down / range
    h4_bias: str = "ranging"          # bullish / bearish / ranging
    htf_agree: bool = False
    direction_ok: bool = False        # proposed direction matches both HTFs
    daily_vwap_position: str = "at"   # above / below / at
    h4_ema20: float = 0.0
    h4_ema50: float = 0.0
    h4_ema200: float = 0.0
    h4_bos: str = "none"
    h4_bos_price: float = 0.0
    h4_bos_ago: int = 0
    wyckoff_phase: str = "ranging"
    wyckoff_event: str = ""
    wyckoff_range_high: float = 0.0
    wyckoff_range_low: float = 0.0
    elliott_wave: str = "no count"
    ob_kind: str = "none"
    ob_high: float = 0.0
    ob_low: float = 0.0
    ob_strength: float = 0.0
    ob_distance_pips: float = 0.0
    fvg_kind: str = "none"
    fvg_high: float = 0.0
    fvg_low: float = 0.0
    fvg_age_hours: float = 0.0
    vpoc_prev: float = 0.0
    vpoc_position: str = "at"
    hvn_levels: list = field(default_factory=list)
    buyside_liquidity: float = 0.0
    buyside_pips: float = 0.0
    sellside_liquidity: float = 0.0
    sellside_pips: float = 0.0
    pattern_name: str = "none"
    pattern_target: float = 0.0
    harmonic_name: str = "none"
    harmonic: Optional[object] = None
    vwap_daily: float = 0.0
    vwap_sigma2_upper: float = 0.0
    vwap_sigma2_lower: float = 0.0
    vwap_sigma3_upper: float = 0.0
    vwap_sigma3_lower: float = 0.0
    notes: list[str] = field(default_factory=list)

    def htf_direction(self) -> str:
        """Combined HTF direction for a trade ('buy'/'sell'/'' when unclear)."""
        if not self.htf_agree:
            return ""
        if self.daily_trend == "up" and self.h4_bias == "bullish":
            return "buy"
        if self.daily_trend == "down" and self.h4_bias == "bearish":
            return "sell"
        return ""


class HTFAnalyzer:
    """Reads daily + H4 + institutional structure for the research engine."""

    def __init__(self) -> None:
        self.smc = SmartMoneyDetector()
        self.wyckoff = WyckoffAnalyzer()
        self.elliott = ElliottWaveAnalyzer()
        self.patterns = PatternDetector()
        self.harmonics = HarmonicDetector()
        self.vwap = VWAPEngine()
        self.vpoc = VolumeProfileEngine()

    def analyze(self, pair: str, frames: dict[str, pd.DataFrame],
                direction: str) -> HTFResult:
        """Full HTF + structure read (never raises)."""
        r = HTFResult()
        m15, h1 = frames["m15"], frames["h1"]
        h4, daily = frames["h4"], frames["d1"]
        now = pd.Timestamp(m15["time"].iloc[-1])
        price = float(m15["close"].iloc[-1])

        # ---- step 3: higher timeframe bias ----
        c_d, c_h4 = daily["close"], h4["close"]
        if len(c_d) >= 60:
            ema50d = c_d.ewm(span=50, adjust=False).mean().iloc[-1]
            ema200d = c_d.ewm(span=200, adjust=False).mean().iloc[-1] if len(c_d) >= 200 else ema50d
            r.daily_trend = "up" if ema50d > ema200d else "down"
        if len(c_h4) >= 200:
            e20 = float(c_h4.ewm(span=20, adjust=False).mean().iloc[-1])
            e50 = float(c_h4.ewm(span=50, adjust=False).mean().iloc[-1])
            e200 = float(c_h4.ewm(span=200, adjust=False).mean().iloc[-1])
            r.h4_ema20, r.h4_ema50, r.h4_ema200 = round(e20, 5), round(e50, 5), round(e200, 5)
            r.h4_bias = "bullish" if e50 > e200 else "bearish"
        r.htf_agree = (r.daily_trend == "up" and r.h4_bias == "bullish") or \
                      (r.daily_trend == "down" and r.h4_bias == "bearish")
        want = "buy" if direction == "buy" else "sell"
        r.direction_ok = r.htf_agree and r.htf_direction() == want

        try:
            snap_vwap = self.vwap.compute(m15, now)
            r.daily_vwap_position = snap_vwap.position_vs_daily
            r.vwap_daily = snap_vwap.daily
            r.vwap_sigma2_upper, r.vwap_sigma2_lower = snap_vwap.sigma2_upper, snap_vwap.sigma2_lower
            r.vwap_sigma3_upper, r.vwap_sigma3_lower = snap_vwap.sigma3_upper, snap_vwap.sigma3_lower
        except Exception as exc:
            r.notes.append(f"vwap failed: {exc}")

        # ---- step 4: institutional structure ----
        try:
            h4_struct = self.smc.analyze(h4, "H4", pair)
            if h4_struct.last_bos:
                r.h4_bos = h4_struct.last_bos["direction"]
                r.h4_bos_price = round(float(h4_struct.last_bos["price"]), 5)
                r.h4_bos_ago = int(h4_struct.last_bos["candles_ago"])
        except Exception as exc:
            r.notes.append(f"smc h4 failed: {exc}")

        try:
            h1_struct = self.smc.analyze(h1, "H1", pair)
            ob = h1_struct.nearest_ob
            if ob is not None:
                r.ob_kind = ob.kind
                r.ob_high, r.ob_low = round(ob.high, 5), round(ob.low, 5)
                r.ob_strength = ob.strength
                r.ob_distance_pips = h1_struct.nearest_ob_distance_pips
            fvg = h1_struct.nearest_fvg or self.smc.find_fvgs(m15)[0:1]
            gaps = self.smc.find_fvgs(m15)
            fvg = gaps[0] if gaps else None
            if fvg is not None:
                r.fvg_kind = fvg.kind
                r.fvg_high, r.fvg_low = round(fvg.high, 5), round(fvg.low, 5)
                r.fvg_age_hours = fvg.age_hours
            # nearest unswept liquidity above and below
            unswept_above = [lv.price for lv in h1_struct.liquidity
                             if not lv.swept and lv.price > price]
            unswept_below = [lv.price for lv in h1_struct.liquidity
                             if not lv.swept and lv.price < price]
            pip = pip_size(pair)
            if unswept_above:
                r.buyside_liquidity = round(min(unswept_above), 5)
                r.buyside_pips = round((min(unswept_above) - price) / pip, 1)
            if unswept_below:
                r.sellside_liquidity = round(max(unswept_below), 5)
                r.sellside_pips = round((price - max(unswept_below)) / pip, 1)
        except Exception as exc:
            r.notes.append(f"smc h1 failed: {exc}")

        try:
            wy = self.wyckoff.analyze(h1)
            r.wyckoff_phase = wy.phase
            r.wyckoff_event = wy.event
            r.wyckoff_range_high, r.wyckoff_range_low = round(wy.range_high, 5), round(wy.range_low, 5)
        except Exception as exc:
            r.notes.append(f"wyckoff failed: {exc}")

        try:
            wave = self.elliott.analyze(h4)
            if wave.wave:
                r.elliott_wave = f"wave {wave.wave} ({wave.phase})"
        except Exception as exc:
            r.notes.append(f"elliott failed: {exc}")

        try:
            pat = self.patterns.detect(h1)
            if pat is not None:
                r.pattern_name = pat.name
                r.pattern_target = pat.target
        except Exception as exc:
            r.notes.append(f"pattern failed: {exc}")

        try:
            harm = self.harmonics.detect(h1)
            if harm is not None:
                r.harmonic_name = harm.name
                r.harmonic = harm
        except Exception as exc:
            r.notes.append(f"harmonic failed: {exc}")

        try:
            # previous day VPOC (profile computed on the frame minus today)
            ts = pd.to_datetime(m15["time"], utc=True)
            today = ts.iloc[-1].normalize()
            prev_days = m15[ts < today]
            if len(prev_days) >= 8:
                prev_profile = self.vpoc.compute(prev_days, pair, today - pd.Timedelta(days=1))
                r.vpoc_prev = prev_profile.poc
                if prev_profile.poc > price * 1.0005:
                    r.vpoc_position = "above"
                elif prev_profile.poc < price * 0.9995:
                    r.vpoc_position = "below"
                r.hvn_levels = prev_profile.hvn
        except Exception as exc:
            r.notes.append("vpoc failed: %s" % exc)

        logger.info("htf %s: daily=%s h4=%s agree=%s dir_ok=%s", pair, r.daily_trend,
                    r.h4_bias, r.htf_agree, r.direction_ok)
        return r
