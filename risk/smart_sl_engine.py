"""Smart SL engine (UPGRADE 8): volatility- and structure-aware stops.

Replaces fixed-pip stops:
- ATR-based: SL = entry ± ATR14 x multiplier (multiplier by strategy class)
- Structure-based: beyond the last significant swing with a 3-pip buffer
- Whichever is SMALLER wins (better RR while still beyond invalidation)
- Dynamic TP: reassessed on momentum every 30 minutes; never moved against a
  winning position's favor.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from strategies.base_strategy import atr, pip_size

logger = logging.getLogger(__name__)

STRATEGY_MULTIPLIERS = {
    "scalping": 1.0, "scalp": 1.0,
    "order_block": 1.5, "fvg": 1.5, "liquidity": 1.5,
    "trend": 2.0, "ema": 2.0, "macd": 2.0, "breakout": 2.0, "reversal": 2.0,
    "swing": 3.0, "cot": 3.0, "divergence": 2.5,
}
DEFAULT_MULTIPLIER = 2.0
STRUCTURE_BUFFER_PIPS = 3.0


def multiplier_for(strategy: str) -> float:
    """ATR multiplier by strategy class."""
    lowered = (strategy or "").lower()
    for key, mult in STRATEGY_MULTIPLIERS.items():
        if key in lowered:
            return mult
    return DEFAULT_MULTIPLIER


def swing_low(df: pd.DataFrame, lookback: int = 12) -> Optional[float]:
    """Most recent significant swing low (local minimum over the lookback)."""
    lows = df["low"].astype(float).tail(lookback)
    if len(lows) < 5:
        return None
    idx = int(lows.values.argmin())
    # require at least 2 candles on each side for "significance"
    if idx < 2 or idx > len(lows) - 3:
        return None
    return float(lows.iloc[idx])


def swing_high(df: pd.DataFrame, lookback: int = 12) -> Optional[float]:
    """Most recent significant swing high."""
    highs = df["high"].astype(float).tail(lookback)
    if len(highs) < 5:
        return None
    idx = int(highs.values.argmax())
    if idx < 2 or idx > len(highs) - 3:
        return None
    return float(highs.iloc[idx])


class SmartSLEngine:
    """Computes dynamic SL/TP and manages TP adjustment over a trade's life."""

    def stop_distance(self, pair: str, strategy: str, entry: float,
                      m15_df: pd.DataFrame, direction: str) -> dict:
        """Return {sl_price, distance_pips, basis} — smaller of ATR/structure."""
        pip = pip_size(pair)
        mult = multiplier_for(strategy)
        atr_value = float(atr(m15_df).iloc[-1]) if len(m15_df) > 20 else 0.0
        atr_dist = atr_value * mult
        candidates = []
        if atr_dist > 0:
            candidates.append((atr_dist, f"atr_{mult}x"))
        if direction == "buy":
            swing = swing_low(m15_df)
            if swing is not None and swing < entry:
                candidates.append((entry - swing + STRUCTURE_BUFFER_PIPS * pip,
                                   "structure_low"))
            sl = entry - min(c[0] for c in candidates)
        else:
            swing = swing_high(m15_df)
            if swing is not None and swing > entry:
                candidates.append((swing - entry + STRUCTURE_BUFFER_PIPS * pip,
                                   "structure_high"))
            sl = entry + min(c[0] for c in candidates)
        chosen = min(candidates, key=lambda c: c[0])
        return {"sl_price": round(sl, 6), "distance_pips": round(chosen[0] / pip, 1),
                "basis": chosen[1], "multiplier": mult}

    def initial_tp(self, pair: str, entry: float, sl_distance_pips: float,
                   m15_df: pd.DataFrame, direction: str,
                   min_rr: float = 1.5) -> dict:
        """TP at the next liquidity swing, at least min_rr x the stop."""
        pip = pip_size(pair)
        min_dist = sl_distance_pips * min_rr * pip
        if direction == "buy":
            target = swing_high(m15_df)
            dist = (target - entry) if (target and target > entry + min_dist) else min_dist
            tp = entry + dist
        else:
            target = swing_low(m15_df)
            dist = (entry - target) if (target and target < entry - min_dist) else min_dist
            tp = entry - dist
        return {"tp_price": round(tp, 6), "rr": round(dist / (sl_distance_pips * pip), 2)}

    def adjust_tp(self, trade: dict, m15_df: pd.DataFrame,
                  current_price: float) -> Optional[dict]:
        """Every-30-min TP reassessment: extend on strong momentum, tighten on
        fading momentum. Never moves TP against a winning position."""
        direction = trade.get("direction")
        entry = float(trade.get("entry_price") or 0.0)
        tp = float(trade.get("tp") or 0.0)
        if not entry or not tp or direction not in ("buy", "sell"):
            return None
        if len(m15_df) < 40:
            return None
        close = m15_df["close"].astype(float)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
        macd_expanding = abs(hist.iloc[-1]) > abs(hist.iloc[-3])
        vol = m15_df["volume"].astype(float)
        vol_rising = float(vol.tail(3).mean()) > float(vol.tail(12).mean()) * 0.9
        strong = macd_expanding and vol_rising
        pip = pip_size(trade.get("pair", "EURUSD"))

        # never reduce profit below current price for a winning trade
        winning = (direction == "buy" and current_price > entry) or \
                  (direction == "sell" and current_price < entry)

        if strong:
            # extend TP one more ATR in the trade direction
            atr_value = float(atr(m15_df).iloc[-1])
            new_tp = tp + atr_value if direction == "buy" else tp - atr_value
            improved = (new_tp > tp) if direction == "buy" else (new_tp < tp)
            if improved:
                return {"tp": round(new_tp, 6), "action": "extended",
                        "reason": "momentum strong (MACD expanding, volume rising)"}
        elif not strong and winning:
            # tighten TP to lock profit, but never beyond current price
            locked = current_price + pip * 2 if direction == "buy" \
                else current_price - pip * 2
            improving = (locked < tp) if direction == "buy" else (locked > tp)
            if improving:
                return {"tp": round(locked, 6), "action": "tightened",
                        "reason": "momentum fading - locking profit"}
        return None
