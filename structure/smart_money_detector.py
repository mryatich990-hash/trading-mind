"""SmartMoneyDetector: structure, order blocks, FVGs, liquidity and Power of 3.

Consolidates and extends the prompt-2 detector: OB strength scores (>3.0
tradeable), 50% mitigation tracking, FVG age limits, round-number liquidity and
daily Power-of-3 phase detection.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.logging_utils import get_logger
from data.market_data_engine import pip_size

logger = get_logger(__name__)

__all__ = ["OrderBlock", "FVG", "LiquidityLevel", "StructureSnapshot", "SmartMoneyDetector"]


@dataclass
class OrderBlock:
    """Order block with strength and mitigation state."""

    kind: str  # bullish_ob / bearish_ob
    high: float
    low: float
    time: pd.Timestamp
    strength: float  # impulse size / OB size; >3.0 tradeable
    mitigated: bool


@dataclass
class FVG:
    """Fair value gap."""

    kind: str  # bullish_fvg / bearish_fvg
    high: float
    low: float
    time: pd.Timestamp
    age_hours: float
    fill_status: str  # unfilled / partially / fully


@dataclass
class LiquidityLevel:
    """A liquidity pool or level."""

    kind: str  # equal_highs / equal_lows / prev_day_high / prev_day_low / asian_high / asian_low / round_number
    price: float
    swept: bool


@dataclass
class StructureSnapshot:
    """Complete structure read for one timeframe."""

    timeframe: str
    trend_label: str  # higher highs and higher lows / lower... / ranging
    last_bos: Optional[dict] = None
    last_choch: Optional[dict] = None
    nearest_ob: Optional[OrderBlock] = None
    nearest_ob_distance_pips: float = 0.0
    nearest_fvg: Optional[FVG] = None
    liquidity: list[LiquidityLevel] = field(default_factory=list)
    power_of_3_phase: str = ""  # accumulation / manipulation / distribution


class SmartMoneyDetector:
    """Detects SMC structure on any timeframe."""

    def __init__(self, zigzag_pct: float = 0.15, ob_min_strength: float = 3.0,
                 fvg_max_hours: float = 6.0, eq_tolerance_pips: float = 3.0) -> None:
        self.zigzag_pct = zigzag_pct
        self.ob_min_strength = ob_min_strength
        self.fvg_max_hours = fvg_max_hours
        self.eq_tolerance_pips = eq_tolerance_pips

    # ---- swings ----

    def zigzag(self, df: pd.DataFrame) -> list[dict]:
        """ZigZag pivots [{index, price, kind}]."""
        if len(df) < 5:
            return []
        highs, lows = df["high"].to_numpy(float), df["low"].to_numpy(float)
        swings: list[dict] = []
        direction = 0
        ext_idx, ext_high, ext_low = 0, highs[0], lows[0]
        for i in range(1, len(df)):
            if direction >= 0 and highs[i] > ext_high:
                ext_high, ext_idx = highs[i], i
            if direction <= 0 and lows[i] < ext_low:
                ext_low, ext_idx = lows[i], i
            if direction >= 0:
                if (ext_high - lows[i]) / ext_high * 100 >= self.zigzag_pct and (direction == 1 or ext_high != highs[0]):
                    if direction == 1 or not swings:
                        swings.append({"index": ext_idx, "price": ext_high, "kind": "high"})
                    direction = -1
                    ext_low, ext_idx = lows[i], i
            if direction <= 0:
                if (highs[i] - ext_low) / ext_low * 100 >= self.zigzag_pct and (direction == -1 or True):
                    if direction == -1 or (swings and swings[-1]["kind"] == "high"):
                        swings.append({"index": ext_idx, "price": ext_low, "kind": "low"})
                    direction = 1
                    ext_high, ext_idx = highs[i], i
        if direction == 1:
            swings.append({"index": ext_idx, "price": ext_high, "kind": "high"})
        elif direction == -1:
            swings.append({"index": ext_idx, "price": ext_low, "kind": "low"})
        return sorted(swings, key=lambda s: s["index"])

    @staticmethod
    def trend_label(swings: list[dict]) -> str:
        """Classify the last two highs and lows."""
        highs = [s for s in swings if s["kind"] == "high"][-2:]
        lows = [s for s in swings if s["kind"] == "low"][-2:]
        if len(highs) < 2 or len(lows) < 2:
            return "ranging"
        if highs[-1]["price"] > highs[-2]["price"] and lows[-1]["price"] > lows[-2]["price"]:
            return "higher highs and higher lows"
        if highs[-1]["price"] < highs[-2]["price"] and lows[-1]["price"] < lows[-2]["price"]:
            return "lower highs and lower lows"
        return "ranging"

    def detect_bos_choch(self, df: pd.DataFrame, swings: list[dict]) -> tuple[Optional[dict], Optional[dict]]:
        """Close-beyond-swing BOS and opposing CHoCH detection."""
        closes = df["close"].to_numpy(float)
        n = len(df)
        bos: Optional[dict] = None
        choch: Optional[dict] = None
        last_dir: Optional[str] = None
        for swing in swings:
            for i in range(swing["index"] + 1, n):
                broke_up = swing["kind"] == "high" and closes[i] > swing["price"]
                broke_down = swing["kind"] == "low" and closes[i] < swing["price"]
                if broke_up or broke_down:
                    ev = {"direction": "bullish" if broke_up else "bearish",
                          "price": swing["price"], "candles_ago": n - 1 - i}
                    if last_dir and last_dir != ev["direction"]:
                        choch = ev
                    else:
                        bos = ev
                    last_dir = ev["direction"]
                    break
        return bos, choch

    # ---- order blocks ----

    def find_order_blocks(self, df: pd.DataFrame, impulse_atr_mult: float = 2.0) -> list[OrderBlock]:
        """OBs with strength = impulse / OB size, mitigation at 50% of body."""
        from data.indicator_engine import atr as atr_fn

        a = float(atr_fn(df, 14).iloc[-1]) if len(df) > 20 else 0.0
        if a <= 0:
            return []
        n = len(df)
        closes, opens = df["close"].to_numpy(float), df["open"].to_numpy(float)
        obs: list[OrderBlock] = []
        for i in range(n - 4, 1, -1):
            body_low, body_high = min(opens[i], closes[i]), max(opens[i], closes[i])
            ob_size = max(body_high - body_low, 1e-9)
            impulse = float(closes[min(i + 3, n - 1)]) - closes[i]
            if abs(impulse) < impulse_atr_mult * a:
                continue
            mitigated = bool((df["low"].iloc[i + 1:] < (body_high + body_low) / 2).any()
                             and (df["high"].iloc[i + 1:] > (body_high + body_low) / 2).any())
            strength = round(abs(impulse) / ob_size, 2)
            if strength < self.ob_min_strength:
                continue
            obs.append(OrderBlock(
                kind="bullish_ob" if impulse > 0 else "bearish_ob",
                high=float(df["high"].iloc[i]), low=float(df["low"].iloc[i]),
                time=df["time"].iloc[i], strength=strength, mitigated=mitigated,
            ))
            if len(obs) >= 4:
                break
        return obs

    # ---- FVGs ----

    def find_fvgs(self, df: pd.DataFrame, now: Optional[pd.Timestamp] = None) -> list[FVG]:
        """FVGs younger than the max age with fill status."""
        now = now or df["time"].iloc[-1]
        gaps: list[FVG] = []
        n = len(df)
        for i in range(n - 1, 2, -1):
            c1_high, c1_low = float(df["high"].iloc[i - 2]), float(df["low"].iloc[i - 2])
            c3_high, c3_low = float(df["high"].iloc[i]), float(df["low"].iloc[i])
            t0 = df["time"].iloc[i - 2]
            age = (now - t0).total_seconds() / 3600.0
            if age > self.fvg_max_hours:
                break
            if c3_low > c1_high:
                kind, high, low = "bullish_fvg", c3_low, c1_high
            elif c3_high < c1_low:
                kind, high, low = "bearish_fvg", c1_low, c3_high
            else:
                continue
            later = df.iloc[i + 1:]
            if len(later) and (later["low"] <= low).any() and (later["high"] >= low).any():
                fill = "partially" if (later["close"] > high).any() else "fully"
                if (later["low"] <= low).all():
                    fill = "fully"
            else:
                fill = "unfilled"
            gaps.append(FVG(kind=kind, high=round(high, 6), low=round(low, 6),
                            time=t0, age_hours=round(age, 1), fill_status=fill))
            if len(gaps) >= 4:
                break
        return gaps

    # ---- liquidity ----

    def liquidity_map(self, df: pd.DataFrame, pair: str) -> list[LiquidityLevel]:
        """Equal highs/lows (3 pip), session extremes, round numbers, swept status."""
        tol = self.eq_tolerance_pips * pip_size(pair)
        levels: list[LiquidityLevel] = []
        window = df.tail(96)
        now_price = float(df["close"].iloc[-1])

        def cluster(values: list[float], kind: str) -> None:
            vals = sorted(values)
            group: list[float] = []
            for v in vals:
                if group and abs(v - group[-1]) <= tol:
                    group.append(v)
                else:
                    if len(group) >= 2:
                        avg = sum(group) / len(group)
                        swept = (now_price > avg) if kind == "equal_highs" else (now_price < avg)
                        levels.append(LiquidityLevel(kind, round(avg, 6), swept))
                    group = [v]
            if len(group) >= 2:
                avg = sum(group) / len(group)
                swept = (now_price > avg) if kind == "equal_highs" else (now_price < avg)
                levels.append(LiquidityLevel(kind, round(avg, 6), swept))

        cluster([float(x) for x in window["high"].tail(48)], "equal_highs")
        cluster([float(x) for x in window["low"].tail(48)], "equal_lows")

        # session extremes
        ts = pd.to_datetime(df["time"], utc=True)
        today = ts.iloc[-1].normalize()
        asian = df[(ts >= today) & (ts < today + pd.Timedelta(hours=7))]
        if len(asian):
            levels.append(LiquidityLevel("asian_high", round(float(asian["high"].max()), 6),
                                         now_price > float(asian["high"].max())))
            levels.append(LiquidityLevel("asian_low", round(float(asian["low"].min()), 6),
                                         now_price < float(asian["low"].min())))
        prev = df[(ts >= today - pd.Timedelta(days=1)) & (ts < today)]
        if len(prev):
            levels.append(LiquidityLevel("prev_day_high", round(float(prev["high"].max()), 6),
                                         now_price > float(prev["high"].max())))
            levels.append(LiquidityLevel("prev_day_low", round(float(prev["low"].min()), 6),
                                         now_price < float(prev["low"].min())))

        # round numbers every 50 pips near price
        step = 50 * pip_size(pair)
        base = round(now_price / step) * step
        for offset in (-step, step):
            level = round(base + offset, 6)
            kind = "round_number"
            swept = abs(now_price - level) < tol
            levels.append(LiquidityLevel(kind, level, swept))
        return levels

    # ---- Power of 3 ----

    @staticmethod
    def power_of_3_phase(now: pd.Timestamp) -> str:
        """Accumulation 00-07, manipulation 07-08, distribution 08-17 UTC."""
        h = now.hour
        if h < 7:
            return "accumulation"
        if h < 8:
            return "manipulation"
        if h < 17:
            return "distribution"
        return "post"

    # ---- main ----

    def analyze(self, df: pd.DataFrame, timeframe: str, pair: str) -> StructureSnapshot:
        """Full structure read."""
        swings = self.zigzag(df)
        bos, choch = self.detect_bos_choch(df, swings)
        obs = self.find_order_blocks(df)
        price = float(df["close"].iloc[-1])
        nearest_ob: Optional[OrderBlock] = None
        nearest_dist = 0.0
        for ob in obs:
            if ob.low <= price <= ob.high:
                dist = 0.0
            elif price < ob.low:
                dist = (ob.low - price) / pip_size(pair)
            else:
                dist = (price - ob.high) / pip_size(pair)
            if nearest_ob is None or dist < nearest_dist:
                nearest_ob, nearest_dist = ob, round(dist, 1)
        gaps = self.find_fvgs(df)
        nearest_fvg = gaps[0] if gaps else None
        return StructureSnapshot(
            timeframe=timeframe,
            trend_label=self.trend_label(swings),
            last_bos=bos, last_choch=choch,
            nearest_ob=nearest_ob, nearest_ob_distance_pips=nearest_dist,
            nearest_fvg=nearest_fvg,
            liquidity=self.liquidity_map(df, pair),
            power_of_3_phase=self.power_of_3_phase(df["time"].iloc[-1]),
        )
