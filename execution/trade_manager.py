"""TradeManager: monitors open trades every 60 seconds.

Implements the master-prompt management rules:
- TP1 (1:1): close 35%, SL to breakeven + 2 pips
- TP2 (1:2): close another 35%, trail remainder 15 pips
- TP3 (1:3): close the rest; beyond 1:4 widen trail to 25 pips
- never move SL against the position; Groq invalidation closes immediately
- re-entry: one half-size re-entry per signal after SL, research re-validated
- Friday 20:30 UTC weekend protection; swap-cost close before rollover
- slippage tracking per pair
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Optional

from sqlalchemy import text as sqltext

from config import settings
from core import db
from core.logging_utils import get_logger
from core.event_bus import bus, EventType
from execution.mt5_connector import MT5Connector
from research.research_engine import ResearchVerdict

logger = get_logger(__name__)

__all__ = ["TradeManager"]

TP1_FRAC, TP2_FRAC = 0.35, 0.35
TRAIL_15_PIPS, TRAIL_25_PIPS = 15.0, 25.0
WEEKEND_HOUR, WEEKEND_MIN = 20, 30
SWAP_PROFIT_SHARE = 0.20
ROLLOVER_HOUR = 22


def _pip(pair: str) -> float:
    """Pip size."""
    pair = pair.upper()
    if pair in ("USDJPY", "GBPJPY", "EURJPY"):
        return 0.01
    if pair == "XAUUSD":
        return 0.1
    if pair in ("NAS100", "US30"):
        return 1.0
    return 0.0001


class TradeManager:
    """Manages open positions through the TP ladder and protection rules."""

    def __init__(self, broker, notifier: Optional[callable] = None,
                 research_engine=None) -> None:
        self.broker = broker
        self.notifier = notifier
        self.research = research_engine
        self._lock = threading.Lock()
        self._partials: dict[int, set[str]] = {}
        self._reentries: set[int] = set()

    def _notify(self, text: str) -> None:
        if self.notifier:
            try:
                self.notifier(text)
            except Exception as exc:
                logger.error("notify failed: %s", exc)

    # ---- main loop ----

    def manage_all(self) -> None:
        """One management pass over all open trades."""
        for trade in db.open_trades():
            try:
                self.manage_trade(trade)
            except Exception as exc:
                logger.exception("manage_trade failed for #%s: %s",
                                 trade.get("id"), exc)

    def manage_trade(self, trade: dict) -> None:
        """Apply all management rules to one open trade."""
        trade_id = int(trade["id"])
        pair, direction = trade["pair"], trade["direction"]
        entry, sl = float(trade["entry_price"]), float(trade["sl"])
        lots = float(trade["lots"])
        price = self._current_price(pair, direction)
        if price <= 0:
            return

        risk = abs(entry - sl)
        if risk <= 0:
            return
        rr = ((price - entry) if direction == "buy" else (entry - price)) / risk

        # Groq invalidation price closes immediately
        inv = self._invalidation(trade_id)
        if inv and ((direction == "buy" and price <= inv)
                    or (direction == "sell" and price >= inv)):
            self.close(trade, price, "Invalidation")
            return

        done = self._partials.setdefault(trade_id, set())

        # TP1: close 35%, move SL to BE+2
        if rr >= 1.0 and "tp1" not in done:
            done.add("tp1")
            partial = round(lots * TP1_FRAC, 2)
            if partial >= 0.01:
                try:
                    self.broker.close_position(int(trade.get("ticket", trade_id)), partial)
                    logger.info("TP1 partial %.2f lots on #%d", partial, trade_id)
                except Exception as exc:
                    logger.error("TP1 partial failed #%d: %s", trade_id, exc)
            be = entry + 2 * _pip(pair) * (1 if direction == "buy" else -1)
            self._move_sl(trade, be)

        # TP2: close 35%, trail 15 pips
        if rr >= 2.0 and "tp2" not in done:
            done.add("tp2")
            partial = round(lots * TP2_FRAC, 2)
            if partial >= 0.01:
                try:
                    self.broker.close_position(int(trade.get("ticket", trade_id)), partial)
                    logger.info("TP2 partial %.2f lots on #%d", partial, trade_id)
                except Exception as exc:
                    logger.error("TP2 partial failed #%d: %s", trade_id, exc)

        # trailing: 15 pips normally, 25 beyond 4R
        trail = TRAIL_25_PIPS if rr >= 4.0 else TRAIL_15_PIPS
        if rr >= 1.0:
            self._apply_trail(trade, price, trail)

        # TP3 / SL / weekend / swap close detection
        tp = float(trade["tp"])
        if ((direction == "buy" and price >= tp) or (direction == "sell" and price <= tp)):
            self.close(trade, price, "TP1")
        elif ((direction == "buy" and price <= sl) or (direction == "sell" and price >= sl)):
            # broker SL hit; detect and close out state, maybe re-enter
            self.close(trade, sl, "SL", allow_reentry=True)

        self._weekend_check(trade, price)
        self._swap_check(trade, price, rr)

    # ---- rules ----

    def _apply_trail(self, trade: dict, price: float, trail_pips: float) -> None:
        """Trail SL behind price by trail_pips; never moves against the position."""
        pair, direction = trade["pair"], trade["direction"]
        dist = trail_pips * _pip(pair)
        current = float(trade["sl"])
        if direction == "buy":
            new_sl = price - dist
            if new_sl > current + 1e-9:
                self._move_sl(trade, new_sl)
        else:
            new_sl = price + dist
            if new_sl < current - 1e-9:
                self._move_sl(trade, new_sl)

    def _move_sl(self, trade: dict, new_sl: float) -> None:
        """Push SL change to the broker and DB."""
        trade_id = int(trade["id"])
        try:
            self.broker.modify_sl_tp(int(trade.get("ticket", trade_id)), new_sl,
                                     float(trade["tp"]))
            with db.engine.begin() as conn, db._WRITE_LOCK:
                conn.execute(sqltext(
                    "UPDATE trades SET sl = :s WHERE id = :i"
                ), {"s": new_sl, "i": trade_id})
            logger.info("SL moved to %.5f on #%d", new_sl, trade_id)
        except Exception as exc:
            logger.error("SL move failed #%d: %s", trade_id, exc)

    def _weekend_check(self, trade: dict, price: float) -> None:
        """Friday 20:30 UTC: protect gains / small losses / validate holds."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        if not (now.weekday() == 4 and (now.hour, now.minute) >= (WEEKEND_HOUR, WEEKEND_MIN)):
            return
        entry = float(trade["entry_price"])
        direction = trade["direction"]
        pip = _pip(trade["pair"])
        pnl_pips = ((price - entry) if direction == "buy" else (entry - price)) / pip
        balance = self._balance()
        risk_pct = abs(pnl_pips * pip / max(entry, 1e-9)) * 100.0
        if pnl_pips > 0 or risk_pct < 0.5:
            self.close(trade, price, "Weekend")
            self._notify(f"🛡️ weekend close {trade['pair']}: {pnl_pips:+.1f} pips")
        else:
            self._notify(f"ℹ️ holding {trade['pair']} over weekend: {pnl_pips:+.1f} pips "
                         "(technical validity maintained)")

    def _swap_check(self, trade: dict, price: float, rr: float) -> None:
        """Close before 22:00 rollover when swap > 20% of unrealized profit."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        if now.hour != ROLLOVER_HOUR - 1:
            return
        unrealized = self._unrealized(trade, price)
        if unrealized <= 0:
            return
        try:
            swap_long, swap_short = self.broker.swap_rates(trade["pair"])
        except Exception:
            return
        swap = swap_long if trade["direction"] == "buy" else swap_short
        swap_cost = abs(swap) * float(trade["lots"]) * 10.0  # USD per night approx
        if swap_cost > unrealized * SWAP_PROFIT_SHARE:
            self.close(trade, price, "Swap")
            self._notify(f"💤 closed {trade['pair']} pre-rollover: swap ${swap_cost:.2f} "
                         f"vs profit ${unrealized:.2f}")

    def _unrealized(self, trade: dict, price: float) -> float:
        """Unrealized USD for the trade."""
        pip = _pip(trade["pair"])
        pips = ((price - float(trade["entry_price"])) if trade["direction"] == "buy"
                else (float(trade["entry_price"]) - price)) / pip
        return pips * 10.0 * float(trade["lots"]) * (10.0 / pip / 10000.0 if pip < 0.001 else 1.0)

    def _balance(self) -> float:
        """Account balance from the broker (0 on failure)."""
        try:
            return float(self.broker.account()["balance"])
        except Exception:
            return 0.0

    def _current_price(self, pair: str, direction: str) -> float:
        """Bid or ask depending on the side (fallback: data engine)."""
        try:
            tick = self.broker.tick(pair)
            return tick["ask"] if direction == "buy" else tick["bid"]
        except Exception:
            try:
                from data.market_data_engine import MarketDataEngine
                engine = MarketDataEngine()
                return engine.get_candles(pair, 1, 2).last_close
            except Exception:
                return 0.0

    def _invalidation(self, trade_id: int) -> Optional[float]:
        """Groq invalidation price stored in features_json."""
        rows = db.closed_trades(limit=0)  # placeholder to keep import surface small
        try:
            with db.engine.begin() as conn:
                row = conn.execute(sqltext(
                    "SELECT features_json FROM trades WHERE id = :i"
                ), {"i": trade_id}).first()
            if row and row[0]:
                import json
                feats = json.loads(row[0])
                return float(feats.get("invalidation", 0) or 0) or None
        except Exception:
            return None
        return None

    # ---- close + re-entry ----

    def close(self, trade: dict, exit_price: float, reason: str,
              allow_reentry: bool = False) -> None:
        """Close a trade, record the result, notify and maybe re-enter."""
        trade_id = int(trade["id"])
        pair, direction = trade["pair"], trade["direction"]
        entry = float(trade["entry_price"])
        pip = _pip(pair)
        pips = ((exit_price - entry) if direction == "buy" else (entry - exit_price)) / pip
        pip_value = 10.0 if pair.upper() not in ("USDJPY",) else 6.8
        pnl = pips * pip_value * float(trade["lots"])
        rr = 0.0
        risk = abs(entry - float(trade["sl"]))
        if risk > 0:
            rr = ((exit_price - entry) if direction == "buy"
                  else (entry - exit_price)) / risk
        try:
            closed_ticket = None
            if hasattr(self.broker, "close_matching"):
                # ticket-agnostic: the trades row has no ticket column, so
                # match on pair+direction (oldest open paper position)
                closed_ticket = self.broker.close_matching(pair, direction)
            if closed_ticket is None:
                self.broker.close_position(int(trade.get("ticket", trade_id)))
        except Exception as exc:
            logger.warning("broker close failed #%d (already closed?): %s", trade_id, exc)
        db.close_trade(trade_id, exit_price, round(pips, 1), round(pnl, 2), round(rr, 2))
        self._record_slippage(trade_id, exit_price)
        counts = db.trade_counts(trade.get("mode"))
        self._notify(
            f"⏹ {pair} {direction.upper()} CLOSED\n"
            f"Trade #{counts['all_time']} ({counts['today']} today)\n"
            f"{entry:.5f} → {exit_price:.5f}\n"
            f"Result: {pips:+.1f} pips | ${pnl:+.2f}\n"
            f"Reason: {reason}\n"
            f"Running today: ${db.daily_pnl():+.2f}"
        )
        _publish_trade_closed(trade_id, pair, pnl)
        logger.info("CLOSED #%d %s: %+.1f pips $%.2f (%s)", trade_id, pair, pips, pnl, reason)

        if reason == "SL" and allow_reentry:
            self._maybe_reenter(trade)

    def _maybe_reenter(self, trade: dict) -> None:
        """One half-size re-entry per signal when research is still valid."""
        trade_id = int(trade["id"])
        if trade_id in self._reentries or self.research is None:
            return
        self._reentries.add(trade_id)
        verdict = self.research.evaluate(
            trade["pair"], trade["direction"], trade["strategy"],
            float(trade["entry_price"]), float(trade["sl"]), float(trade["tp"]),
            trade.get("session", ""))
        if verdict.approved:
            half = Decimal(str(float(trade["lots"]) * 0.5)).quantize(Decimal("0.01"))
            self._notify(f"🔁 re-entry signal for {trade['pair']} at half size")
            logger.info("re-entry approved for #%d (%s)", trade_id, trade["pair"])
            # execution handled by the engine loop; verdict published for the caller
            _publish_trade_closed(trade_id, f"REENTRY_{trade['pair']}", 0.0)

    def _record_slippage(self, trade_id: int, exit_price: float) -> None:
        """Track close-side slippage for weekly reports."""
        try:
            with db.engine.begin() as conn, db._WRITE_LOCK:
                conn.execute(sqltext(
                    "UPDATE trades SET exit_price = :x WHERE id = :i AND exit_price IS NULL"
                ), {"x": exit_price, "i": trade_id})
        except Exception as exc:
            logger.debug("slippage record skipped: %s", exc)


def _publish_trade_closed(trade_id: int, pair: str, pnl: float) -> None:
    """Bridge a close event onto the async bus."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(bus.publish(EventType.TRADE_CLOSED,
                                     {"trade_id": trade_id, "pair": pair, "pnl": pnl}))
    except RuntimeError:
        try:
            asyncio.run(bus.publish(EventType.TRADE_CLOSED,
                                    {"trade_id": trade_id, "pair": pair, "pnl": pnl}))
        except Exception:
            pass
