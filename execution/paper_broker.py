"""Internal paper broker (UPGRADE: brokerless demo trading).

Same interface as OandaConnector/MT5Connector so the execution engine and
trade manager work unchanged. Fills are simulated from live data-engine
prices: mid ± half-spread + adverse slippage. Balance is virtual (persisted
to system_state so it survives restarts); every trade the bot opens through
this broker is a real decision cycle with simulated money.

This exists because OANDA rejects Kenyan residents and MT5 needs Windows:
the demo gate (60 trades at 52% WR) can now be satisfied entirely internally
before ever funding a real broker account.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from typing import Optional

from config import settings
from core import db
from core.logging_utils import get_logger
from execution.pip_math import pip_size as _pip
from execution.pip_math import pip_value_usd as _pip_value
from execution.pip_math import spread_pips_of as _spread_pips

logger = logging.getLogger(__name__)

__all__ = ["PaperBroker"]

UNIT_SIZE = {"XAUUSD": 100.0, "NAS100": 1.0, "US30": 1.0}  # units per lot


class PaperBroker:
    """Virtual broker with DB-persisted balance and positions."""

    name = "paper"

    def __init__(self, data_engine=None, start_balance: Optional[float] = None) -> None:
        self.data = data_engine
        self.start_balance = start_balance or settings.PAPER_START_BALANCE
        self.enabled = settings.PAPER_BROKER_ENABLED
        self._lock = threading.RLock()
        self._balance: float = self.start_balance
        self._positions: dict[int, dict] = {}
        self._ticket_seq: int = 1000
        self._load()

    # ---- persistence ----

    def _load(self) -> None:
        """Restore balance/positions/ticket counter from system_state."""
        try:
            bal = db.get_state("paper_balance", "")
            if bal:
                self._balance = float(bal)
            pos = db.get_state("paper_positions", "")
            if pos:
                self._positions = {int(k): v for k, v in json.loads(pos).items()}
            seq = db.get_state("paper_ticket_seq", "")
            if seq:
                self._ticket_seq = int(seq)
        except Exception as exc:
            logger.warning("paper broker state restore failed: %s", exc)

    def _save(self) -> None:
        """Persist state (caller holds the lock)."""
        try:
            db.set_state("paper_balance", f"{self._balance:.2f}")
            db.set_state("paper_positions", json.dumps(self._positions))
            db.set_state("paper_ticket_seq", str(self._ticket_seq))
        except Exception as exc:
            logger.warning("paper broker state save failed: %s", exc)

    # ---- pricing ----

    def _mid(self, pair: str) -> Optional[float]:
        """Latest mid price from the data engine."""
        if self.data is None:
            return None
        try:
            candle = self.data.get_candles(pair, 1, 3)
            return float(candle.df["close"].iloc[-1])
        except Exception:
            try:
                candle = self.data.get_candles(pair, 15, 3)
                return float(candle.df["close"].iloc[-1])
            except Exception:
                return None

    def tick(self, pair: str) -> dict:
        """Bid/ask around mid with the pair's modeled spread."""
        pair = pair.upper()
        mid = self._mid(pair)
        if mid is None:
            raise RuntimeError(f"paper broker: no price for {pair}")
        spread = _spread_pips(pair) * _pip(pair)
        return {"bid": round(mid - spread / 2, 6),
                "ask": round(mid + spread / 2, 6),
                "spread_pips": _spread_pips(pair)}

    # ---- interface (mirrors OandaConnector) ----

    def available(self) -> bool:
        """Enabled flag."""
        return self.enabled

    def connect(self) -> bool:
        """Always ready."""
        return self.enabled

    def healthy(self) -> bool:
        """Always healthy."""
        return self.enabled

    def account(self) -> dict:
        """Mark-to-market account snapshot."""
        with self._lock:
            equity = self._balance + sum(self._unrealized(t) for t in self._positions.values())
            margin_used = sum(t["lots"] * 250.0 for t in self._positions.values())
            return {"login": "PAPER", "balance": round(self._balance, 2),
                    "equity": round(equity, 2), "margin_used": round(margin_used, 2),
                    "margin_free": round(equity - margin_used, 2),
                    "margin_level": round((equity / margin_used) * 100.0, 1)
                    if margin_used > 0 else 1000.0,
                    "currency": "USD"}

    def market_order(self, pair: str, direction: str, lots: float,
                     sl: float, tp: float, comment: str = "bot") -> int:
        """Simulated market fill: price ± spread + random adverse slippage."""
        pair = pair.upper()
        # A stop on the wrong side of entry (buy with SL above, sell with SL
        # below) would be "hit" immediately and in profit — GBPJPY #5
        # booked +$73 of fake PnL this way. Refuse it at the door.
        mid = self._mid(pair)
        if sl and mid is not None:
            wrong_side = ((direction == "buy" and sl >= mid)
                          or (direction == "sell" and sl <= mid))
            if wrong_side:
                raise ValueError(
                    f"paper broker: invalid {direction} SL {sl} vs mid {mid} for {pair}")
        tick = self.tick(pair)
        slippage = random.uniform(0.0, 0.8) * _pip(pair)
        if direction == "buy":
            fill = tick["ask"] + slippage
        else:
            fill = tick["bid"] - slippage
        with self._lock:
            self._ticket_seq += 1
            ticket = self._ticket_seq
            self._positions[ticket] = {
                "ticket": ticket, "pair": pair, "direction": direction,
                "lots": round(float(lots), 2), "entry": round(fill, 6),
                "sl": float(sl), "tp": float(tp), "comment": comment[:26],
                "opened": db._utcnow().isoformat(),
            }
            self._save()
        logger.info("paper fill #%d %s %s %.2f lots @ %.5f (sl %s tp %s) [%s]",
                    ticket, pair, direction, lots, fill, sl, tp, comment)
        return ticket

    def close_matching(self, pair: str, direction: str) -> Optional[int]:
        """Close the OLDEST open position matching pair+direction; return its
        ticket (or None if no match).

        Ticket-agnostic reconciliation: the trades table has no ticket
        column, so callers cannot reliably map a trade row to its paper
        ticket (the old id-fallback closed nonexistent tickets and P&L was
        never settled into the balance).
        """
        pair = pair.upper()
        with self._lock:
            for ticket in sorted(self._positions):
                p = self._positions[ticket]
                if p["pair"] == pair and p["direction"] == direction:
                    self.close_position(ticket)
                    return ticket
        return None

    def close_position(self, ticket: int, lots: Optional[float] = None) -> bool:
        """Close full or partial; realizes PnL into the virtual balance."""
        with self._lock:
            pos = self._positions.get(int(ticket))
            if pos is None:
                logger.warning("paper close: unknown ticket %s", ticket)
                return False
            pair = pos["pair"]
            close_lots = round(min(float(lots or pos["lots"]), pos["lots"]), 2)
            try:
                tick = self.tick(pair)
                exit_price = tick["bid"] if pos["direction"] == "buy" else tick["ask"]
            except Exception as exc:
                logger.warning("paper close failed (no price for %s): %s", pair, exc)
                return False
            pips = ((exit_price - pos["entry"]) if pos["direction"] == "buy"
                    else (pos["entry"] - exit_price)) / _pip(pair)
            pnl = pips * _pip_value(pair) * close_lots
            self._balance += pnl
            pos["lots"] = round(pos["lots"] - close_lots, 2)
            if pos["lots"] < 0.01:
                del self._positions[int(ticket)]
            self._save()
        logger.info("paper close #%d: %+.1f pips $%+.2f (balance %.2f)",
                    ticket, pips, pnl, self._balance)
        return True

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        """Update SL/TP on a virtual position."""
        with self._lock:
            pos = self._positions.get(int(ticket))
            if pos is None:
                return False
            pos["sl"] = float(sl)
            pos["tp"] = float(tp)
            self._save()
        return True

    def positions(self) -> list[dict]:
        """Open virtual positions (OANDA-shaped dicts)."""
        with self._lock:
            out = []
            for t in self._positions.values():
                out.append({
                    "ticket": t["ticket"], "pair": t["pair"],
                    "direction": t["direction"], "lots": t["lots"],
                    "entry": t["entry"], "sl": t["sl"], "tp": t["tp"],
                    "profit": round(self._unrealized(t), 2), "swap": 0.0,
                    "opened": t["opened"],
                })
            return out

    # ---- internals ----

    def _unrealized(self, pos: dict) -> float:
        """Unrealized PnL for one position at current prices."""
        try:
            tick = self.tick(pos["pair"])
            price = tick["bid"] if pos["direction"] == "buy" else tick["ask"]
            pips = ((price - pos["entry"]) if pos["direction"] == "buy"
                    else (pos["entry"] - price)) / _pip(pos["pair"])
            return pips * _pip_value(pos["pair"]) * pos["lots"]
        except Exception:
            return 0.0

    def reset(self, balance: Optional[float] = None) -> None:
        """Wipe the virtual account back to start balance (admin action)."""
        with self._lock:
            self._balance = balance or self.start_balance
            self._positions = {}
            self._save()
        logger.info("paper account reset to %.2f", self._balance)

    def status(self) -> dict:
        """Dashboard payload."""
        acct = self.account()
        return {"enabled": self.enabled, "balance": acct["balance"],
                "equity": acct["equity"], "open_positions": len(self._positions),
                "start_balance": self.start_balance}
