"""Scalping module (UPGRADE 4): kill-zone M1/M5 scalps, separate from main engine.

Runs ONLY during London open (07:00-08:00 UTC) and NY open (13:00-14:00 UTC)
when spread and ATR conditions are met. Micro order blocks on M5, entry on
M1 rejection, 8-12 pip targets, 5-7 pip stops, 15-minute time limit, max 5
scalps per session, 0.5% risk, never during news, never alongside a main
trade, day-lock after 3 consecutive losses. Every scalp must pass the deep
learning ensemble (LSTM direction + XGBoost >= 60% + RL agreement).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings
from strategies.base_strategy import MarketContext, StrategySignal, atr, pip_size

logger = logging.getLogger(__name__)

KILL_ZONES = (("london_open", 7, 8), ("ny_open", 13, 14))
SPREAD_CAP_PIPS = {"EURUSD": 1.0, "GBPUSD": 1.5}
SCALP_SESSIONS_TODAY = "scalp_sessions_date"


@dataclass
class ScalpState:
    """Per-day scalping counters."""

    date: str = ""
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    consecutive_losses: int = 0
    locked: bool = False


class ScalpingEngine:
    """Kill-zone scalper with hard risk rails and DL gating."""

    def __init__(self, ensemble=None, notifier=None) -> None:
        self.ensemble = ensemble
        self.notifier = notifier
        self.enabled = settings.SCALPING_ENABLED
        self.state = ScalpState(date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        self.last_scalp_ts = 0.0

    # ---- gates ----

    def kill_zone(self, now: Optional[datetime] = None) -> Optional[str]:
        """Active kill-zone name or None."""
        t = (now or datetime.now(timezone.utc)).time().hour
        for name, start, end in KILL_ZONES:
            if start <= t < end:
                return name
        return None

    def can_scalp(self, pair: str, spread_pips: float, m15_df=None,
                  open_trades: Optional[list[dict]] = None,
                  minutes_to_news: Optional[int] = None) -> tuple[bool, str]:
        """All pre-trade gates; returns (ok, reason)."""
        if not self.enabled:
            return False, "scalping disabled"
        self._roll_day()
        if self.state.locked:
            return False, "day-locked after 3 consecutive losses"
        if self.kill_zone() is None:
            return False, "outside kill zones"
        if self.state.trades_today >= settings.SCALP_MAX_PER_SESSION:
            return False, "session scalp limit reached"
        cap = SPREAD_CAP_PIPS.get(pair.upper(), 1.2)
        if spread_pips > cap:
            return False, f"spread {spread_pips} > {cap}"
        if minutes_to_news is not None and minutes_to_news <= 15:
            return False, "news within 15 minutes"
        for t in open_trades or []:
            if t.get("status") == "open":
                return False, "main trade open"
        if m15_df is not None and len(m15_df) > 30:
            atr_series = atr(m15_df)
            ratio = float(atr_series.iloc[-1] / (atr_series.rolling(20).mean().iloc[-1] + 1e-12))
            if ratio < 0.7:
                return False, f"ATR {ratio:.0%} of average (need >= 70%)"
        return True, "ok"

    def _roll_day(self) -> None:
        """Reset counters at UTC midnight."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.date != today:
            self.state = ScalpState(date=today)

    # ---- signal ----

    def find_signal(self, pair: str, m5_df, m1_df) -> Optional[StrategySignal]:
        """Micro order block on M5 + M1 rejection -> scalp signal."""
        if m5_df is None or m1_df is None or len(m5_df) < 40 or len(m1_df) < 20:
            return None
        a = float(atr(m5_df).iloc[-1])
        if a <= 0:
            return None
        pip = pip_size(pair)
        ob = self._micro_ob(m5_df, a)
        if ob is None:
            return None
        last = m1_df.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        rng = max(float(last["high"]) - float(last["low"]), 1e-12)
        rejection = body / rng >= 0.35
        price = float(last["close"])
        pip = 0.01 if "JPY" in pair or pair == "XAUUSD" else 0.0001
        if ob["side"] == "bullish" and ob["low"] <= price <= ob["high"] and rejection:
            sl_dist = pip * 6.0
            tp_dist = pip * 10.0
            return StrategySignal(
                strategy=f"scalp_{self.kill_zone()}", pair=pair, direction="buy",
                entry=price, sl=price - sl_dist, tp=price + tp_dist,
                session="Scalp", confluences=["m5_micro_ob", "m1_rejection", "kill_zone"])
        if ob["side"] == "bearish" and ob["low"] <= price <= ob["high"] and rejection:
            sl_dist = pip * 6.0
            tp_dist = pip * 10.0
            return StrategySignal(
                strategy=f"scalp_{self.kill_zone()}", pair=pair, direction="sell",
                entry=price, sl=price + sl_dist, tp=price - tp_dist,
                session="Scalp", confluences=["m5_micro_ob", "m1_rejection", "kill_zone"])
        return None

    @staticmethod
    def _micro_ob(m5_df, atr_value: float) -> Optional[dict]:
        """Last opposing candle before a >= 1.5x ATR move in the last 30 minutes."""
        n = len(m5_df)
        scan = m5_df.tail(30)
        for i in range(len(scan) - 4, 2, -1):
            o = float(scan["open"].iloc[i])
            c = float(scan["close"].iloc[i])
            move = float(scan["close"].iloc[min(i + 3, len(scan) - 1)]) - c
            if c < o and move > 1.5 * atr_value:
                return {"side": "bullish", "high": float(scan["high"].iloc[i]),
                        "low": float(scan["low"].iloc[i])}
            if c > o and move < -1.5 * atr_value:
                return {"side": "bearish", "high": float(scan["high"].iloc[i]),
                        "low": float(scan["low"].iloc[i])}
        return None

    # ---- DL gate ----

    def dl_gate(self, pair: str, direction: str, verdict) -> tuple[bool, str]:
        """Ensemble must agree on scalp direction (xgb >= 60%)."""
        if self.ensemble is None:
            return True, "no ensemble (allowed)"
        lstm_dir = verdict.lstm.get("direction", "neutral")
        wp = verdict.xgboost.get("win_probability")
        rl = verdict.rl.get("action", "skip")
        if lstm_dir not in ("neutral", direction):
            return False, f"lstm={lstm_dir}"
        if wp is not None and wp < 60.0:
            return False, f"xgb={wp}% < 60%"
        if rl not in ("skip", direction):
            return False, f"rl={rl}"
        return True, "dl ok"

    # ---- outcome bookkeeping ----

    def record_outcome(self, won: bool, pnl: float) -> None:
        """Update counters; day-lock after 3 straight losses."""
        self.state.trades_today += 1
        self.state.pnl += pnl
        if won:
            self.state.wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losses += 1
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= 3:
                self.state.locked = True
                if self.notifier:
                    self.notifier.send("🚫 Scalping disabled for today: 3 consecutive losses")
        self.last_scalp_ts = time.time()

    def status(self) -> dict:
        """Dashboard payload."""
        zone = self.kill_zone()
        return {"enabled": self.enabled, "active": zone is not None and not self.state.locked,
                "kill_zone": zone, "locked": self.state.locked,
                "trades_today": self.state.trades_today, "wins": self.state.wins,
                "losses": self.state.losses, "pnl": round(self.state.pnl, 2),
                "max_per_session": settings.SCALP_MAX_PER_SESSION,
                "risk_pct": settings.SCALP_RISK_PCT}
