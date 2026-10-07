"""ExecutionEngine: turns approved research verdicts into broker orders.

Features:
- broker chain: MT5 primary -> OANDA fallback -> shadow (DB only)
- duplicate-signal protection via signal hashes (5-minute window)
- smart execution: limit orders 2 pips inside the zone (news-spike fade uses
  market orders because the signal is time-critical)
- shadow mirroring of every fill into shadow_trades
- event-bus publishing of TRADE_OPENED
"""

from __future__ import annotations

import threading
import time
from decimal import Decimal
from typing import Optional

from config import settings
from core import db
from core.event_bus import bus, EventType
from core.logging_utils import get_logger
from execution.mt5_connector import BrokerError, MT5Connector
from execution.oanda_connector import OandaConnector
from research.research_engine import ResearchVerdict

logger = get_logger(__name__)

__all__ = ["ExecutionEngine"]

MARKET_ORDER_STRATEGIES = {"news_spike_fade", "forced_test_trade"}


class ExecutionEngine:
    """Places and tracks orders with failover and safety checks."""

    def __init__(self, mt5: Optional[MT5Connector] = None,
                 oanda: Optional[OandaConnector] = None,
                 paper: Optional[Any] = None,
                 ctrader: Optional[Any] = None,
                 mode: Optional[str] = None,
                 notifier: Optional[callable] = None) -> None:
        self.mt5 = mt5 or MT5Connector()
        self.oanda = oanda or OandaConnector()
        if ctrader is None:
            try:
                from execution.ctrader_connector import CtraderConnector

                ctrader = CtraderConnector()
            except Exception as exc:
                logger.debug("ctrader connector unavailable: %s", exc)
                ctrader = None
        self.ctrader = ctrader
        if paper is None:
            try:
                from execution.paper_broker import PaperBroker

                paper = PaperBroker()
            except Exception as exc:  # pragma: no cover
                logger.warning("paper broker unavailable: %s", exc)
                paper = None
        self.paper = paper
        self.mode = mode or ("demo" if settings.DEMO_MODE else "live")
        self.notifier = notifier
        self._lock = threading.Lock()

    # ---- helpers ----

    def _notify(self, text: str) -> None:
        if self.notifier:
            try:
                self.notifier(text)
            except Exception as exc:
                logger.error("notify failed: %s", exc)

    def active_broker(self):
        """First healthy broker in the chain, else None (shadow mode).

        Chain: MT5 -> OANDA -> cTrader -> paper (internal simulation).
        """
        if self.mt5.available() and self.mt5.healthy():
            return self.mt5
        if self.oanda.available() and self.oanda.healthy():
            return self.oanda
        if self.ctrader is not None and self.ctrader.available() \
                and self.ctrader.healthy():
            return self.ctrader
        if self.paper is not None and self.paper.available() and self.paper.healthy():
            return self.paper
        return None

    # ---- main ----

    def execute(self, verdict: ResearchVerdict, lots: Decimal,
                balance: float, confluence: int = 8) -> Optional[int]:
        """Execute an approved verdict. Returns the trade row id."""
        if not verdict.approved:
            logger.warning("execute called on unapproved verdict: %s", verdict.reason)
            return None
        s_hash = _hash_signal(verdict)
        if db.duplicate_signal_recently(s_hash, minutes=5):
            logger.info("duplicate signal suppressed: %s", s_hash)
            return None

        broker = self.active_broker()
        if broker is None:
            logger.warning("REJECT [no-broker] %s %s (%s): no healthy broker "
                           "in chain (MT5/OANDA/cTrader/paper all unavailable)",
                           verdict.pair, verdict.direction, verdict.strategy)
            return None

        strategy = verdict.strategy
        lots_f = float(lots)
        use_market = strategy in MARKET_ORDER_STRATEGIES
        intended = verdict.entry

        try:
            with self._lock:
                if use_market:
                    fill = self._market_fill(broker, verdict, lots_f)
                else:
                    fill = self._limit_fill(broker, verdict, lots_f)
            if fill.get("deferred"):
                # limit zone not reached yet: NO trade row (a deferred fill is
                # not a position — recording one creates phantom open trades
                # that manage/monitor logic would try to track forever).
                # PERSIST the pending setup so a restart (OOM kills happen
                # ~daily on the 512MB plan; NAS100 #428 lost a confluence-9
                # approval this way) does not silently drop it.
                logger.info("limit fill deferred for %s at %.5f; no trade row",
                            verdict.pair, verdict.entry)
                db.audit("execution", "fill_deferred",
                         f"{verdict.pair} {verdict.direction} {strategy} limit "
                         f"{verdict.entry:.5f} not reached; no trade taken",
                         source="execution_engine")
                self._save_pending_fill(verdict, lots_f)
                self._notify(
                    f"⏳ {verdict.pair} {verdict.direction.upper()} limit deferred\n"
                    f"Strategy: {strategy} | Zone: {verdict.entry:.5f}\n"
                    f"Price did not return to the zone — armed for "
                    f"{settings.PENDING_LIMIT_TTL_MIN:.0f} min."
                )
                return None
        except BrokerError as exc:
            logger.error("MT5 execution failed: %s", exc)
            try:
                fill = self._market_fill(self.oanda, verdict, lots_f) \
                    if self.oanda.available() else None
                if fill is None:
                    self._notify(f"⚠️ execution failed for {verdict.pair}: {exc}")
                    return None
            except Exception as exc2:
                logger.error("OANDA fallback failed: %s", exc2)
                self._notify(f"⚠️ both brokers failed for {verdict.pair}: {exc2}")
                return None
        except Exception as exc:
            logger.exception("execution error: %s", exc)
            self._notify(f"⚠️ execution error {verdict.pair}: {exc}")
            return None

        slippage_pips = abs(fill["price"] - intended) / _pip(verdict.pair)
        trade_id = db.record_trade(
            pair=verdict.pair, direction=verdict.direction, lots=lots_f,
            entry=fill["price"], sl=verdict.sl, tp=verdict.tp1,
            strategy=strategy, session=fill.get("session", ""),
            confluence=confluence, conviction=verdict.conviction,
            reasoning=str(verdict.groq_response.get("reasoning", ""))[:4000],
            signal_hash=s_hash, mode=self.mode, tp2=verdict.tp2, tp3=verdict.tp3,
            features={"consensus_size": verdict.consensus_size,
                      "regime": verdict.regime, "invalidation": verdict.invalidation},
        )
        db.audit("trade", "opened",
                 f"{verdict.pair} {verdict.direction} {lots_f} lots @ {fill['price']:.5f} "
                 f"slippage {slippage_pips:.1f} pips", source="execution_engine")
        # permanent guard tripwire (runs on Render, no laptop watcher needed):
        # lots above the cap should be impossible (lots_for_risk clamps), so
        # this firing means a code path bypassed the risk gate entirely.
        sl_dist_pips = abs(verdict.entry - verdict.sl) / _pip(verdict.pair)
        if lots_f > settings.MAX_ABS_LOTS + 1e-9:
            db.audit("risk", "lot_guard_violation",
                     f"#{trade_id} {verdict.pair} {verdict.direction} {strategy}: "
                     f"{lots_f} lots > MAX_ABS_LOTS {settings.MAX_ABS_LOTS}",
                     source="execution_engine")
            logger.warning("lot guard violation #%d: %.2f lots > cap %s",
                           trade_id, lots_f, settings.MAX_ABS_LOTS)
            self._notify(f"⚠️ SIZING GUARD VIOLATION #{trade_id} {verdict.pair} "
                         f"{verdict.direction.upper()}: {lots_f} lots > cap "
                         f"{settings.MAX_ABS_LOTS}")
        self._mirror_shadow(trade_id, verdict, fill["price"], intended)
        counts = db.trade_counts(self.mode)
        self._notify(
            f"▶ {verdict.pair} {verdict.direction.upper()}\n"
            f"Trade #{counts['all_time']} ({counts['today']} today)\n"
            f"Entry: {fill['price']:.5f} | SL: {verdict.sl:.5f} | TP1: {verdict.tp1:.5f} | "
            f"TP2: {verdict.tp2:.5f}\n"
            f"Sizing: {lots_f:.2f} lots (cap {settings.MAX_ABS_LOTS}) | "
            f"SL dist {sl_dist_pips:.1f} pips\n"
            f"Conviction: {verdict.conviction}% | Strategy: {strategy}\n"
            f"Confluence: {confluence}/10 | COT: {verdict.macro.cot_bias if verdict.macro else 'n/a'}\n"
            f"Session: {fill.get('session', '')} | Regime: {verdict.regime}"
        )
        # publish on the event bus (sync wrapper for async context bridging)
        _publish(EventType.TRADE_OPENED, {"trade_id": trade_id, "pair": verdict.pair,
                                          "direction": verdict.direction,
                                          "lots": lots_f, "mode": self.mode})
        logger.info("EXECUTED %s %s %.2f lots @ %.5f (slippage %.1f pips) id=%d",
                    verdict.pair, verdict.direction, lots_f, fill["price"],
                    slippage_pips, trade_id)
        return trade_id

    # ---- pending limit fills (survive restarts) ----

    def _save_pending_fill(self, verdict: ResearchVerdict, lots: float) -> None:
        """Persist a deferred limit fill under system_state.

        Keyed by signal hash so repeated deferrals of the same setup refresh
        rather than duplicate. _list_pending fills scan every entry.
        """
        import json as _json
        from datetime import datetime, timezone
        try:
            key = f"pending_fill_{_hash_signal(verdict)}"
            payload = {
                "pair": verdict.pair, "direction": verdict.direction,
                "strategy": verdict.strategy, "entry": verdict.entry,
                "sl": verdict.sl, "tp1": verdict.tp1, "tp2": verdict.tp2,
                "tp3": verdict.tp3, "lots": lots,
                "confluence": verdict.confluence, "conviction": verdict.conviction,
                "consensus_size": verdict.consensus_size,
                "size_multipliers": verdict.size_multipliers,
                "regime": verdict.regime,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            db.set_state(key, _json.dumps(payload))
        except Exception as exc:
            logger.warning("pending fill save failed: %s", exc)

    def service_pending_fills(self) -> int:
        """Re-arm deferred fills after a restart.

        Called from the trading cycle. For every stored pending fill inside
        the TTL: if price is now at/beyond the zone, execute it immediately
        (market path) — the research approval already happened. Expired
        entries are dropped with an audit row. Returns fills executed.
        """
        import json as _json
        from datetime import datetime, timezone
        executed = 0
        try:
            rows = db.all_state()
        except Exception as exc:
            logger.warning("pending fill scan failed: %s", exc)
            return 0
        now = datetime.now(timezone.utc)
        for row in rows:
            key = row.get("key", "")
            if not key.startswith("pending_fill_"):
                continue
            try:
                p = _json.loads(row.get("value", "{}"))
                created = datetime.fromisoformat(p["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_min = (now - created).total_seconds() / 60.0
            except Exception:
                db.del_state(key)
                continue
            if age_min > settings.PENDING_LIMIT_TTL_MIN:
                db.audit("execution", "pending_fill_expired",
                         f"{p.get('pair')} {p.get('direction')} {p.get('strategy')} "
                         f"zone {p.get('entry')} expired after {age_min:.0f} min",
                         source="execution_engine")
                db.del_state(key)
                continue
            try:
                price_now = self._current_price(p["pair"])
            except Exception:
                continue
            if not price_now:
                continue
            at_zone = (p["direction"] == "buy" and price_now <= p["entry"] + 2 * _pip(p["pair"])) \
                or (p["direction"] == "sell" and price_now >= p["entry"] - 2 * _pip(p["pair"]))
            if not at_zone:
                continue  # still waiting; stays armed
            verdict = ResearchVerdict(
                pair=p["pair"], direction=p["direction"], strategy=p["strategy"],
                approved=True, reason="pending fill re-armed after restart",
                entry=p["entry"], sl=p["sl"], tp1=p["tp1"], tp2=p["tp2"],
                tp3=p["tp3"], confluence=int(p.get("confluence", 8)),
                conviction=int(p.get("conviction", 0)),
                consensus_size=float(p.get("consensus_size", 1.0)),
                regime=p.get("regime", "unknown"))
            from decimal import Decimal as _Dec
            broker = self.active_broker()
            if broker is None:
                continue
            try:
                with self._lock:
                    fill = self._market_fill(broker, verdict, float(p["lots"]))
                slippage_pips = abs(fill["price"] - verdict.entry) / _pip(verdict.pair)
                trade_id = db.record_trade(
                    pair=verdict.pair, direction=verdict.direction, lots=float(p["lots"]),
                    entry=fill["price"], sl=verdict.sl, tp=verdict.tp1,
                    strategy=verdict.strategy, session=fill.get("session", ""),
                    confluence=verdict.confluence, conviction=verdict.conviction,
                    reasoning="re-armed pending fill (approved pre-restart)",
                    signal_hash=key.replace("pending_fill_", ""), mode=self.mode,
                    tp2=verdict.tp2, tp3=verdict.tp3,
                    features={"consensus_size": verdict.consensus_size,
                              "regime": verdict.regime,
                              "invalidation": 0.0})
                db.audit("trade", "opened",
                         f"{verdict.pair} {verdict.direction} {float(p['lots'])} lots @ "
                         f"{fill['price']:.5f} (re-armed pending fill, slippage "
                         f"{slippage_pips:.1f} pips)", source="execution_engine")
                db.del_state(key)  # consumed
                executed += 1
                self._notify(f"⏫ {verdict.pair} {verdict.direction.upper()} pending "
                             f"fill executed at {fill['price']:.5f} "
                             f"(zone {verdict.entry:.5f}, armed pre-restart)")
            except Exception as exc:
                logger.warning("pending fill execution failed for %s: %s", p.get("pair"), exc)
        return executed

    # ---- fill styles ----

    def _market_fill(self, broker, verdict: ResearchVerdict, lots: float) -> dict:
        """Market order; returns fill info."""
        ticket = broker.market_order(verdict.pair, verdict.direction, lots,
                                     verdict.sl, verdict.tp1, verdict.strategy)
        return {"ticket": ticket, "price": verdict.entry,
                "session": verdict.htf and "" or ""}

    def _limit_fill(self, broker, verdict: ResearchVerdict, lots: float) -> dict:
        """Limit order 2 pips inside the zone; falls back to market when the
        zone is already at/beyond price (immediate fill region)."""
        pip = _pip(verdict.pair)
        offset = 2 * pip
        if verdict.direction == "buy":
            limit_price = min(verdict.entry, verdict.entry - offset)
        else:
            limit_price = max(verdict.entry, verdict.entry + offset)
        logger.info("limit fill for %s at %.5f (entry %.5f)", verdict.pair,
                    limit_price, verdict.entry)
        # OANDA/MT5 unified adapters expose market_order only in this build;
        # the limit is emulated with a market order at the limit price when the
        # distance is within one spread, otherwise we wait for price to return.
        broker_account = broker.account() if hasattr(broker, "account") else {}
        price_now = self._current_price(verdict.pair)
        near = abs(price_now - limit_price) <= 2 * pip
        if near:
            ticket = broker.market_order(verdict.pair, verdict.direction, lots,
                                         verdict.sl, verdict.tp1, verdict.strategy)
            return {"ticket": ticket, "price": price_now}
        logger.info("price %.5f not at zone %.5f; deferring fill", price_now, limit_price)
        return {"ticket": 0, "price": verdict.entry, "deferred": True}

    def _current_price(self, pair: str) -> float:
        """Latest price from the active broker or the data engine."""
        broker = self.active_broker()
        if broker is not None and hasattr(broker, "tick"):
            try:
                tick = broker.tick(pair)
                return (tick["ask"] + tick["bid"]) / 2.0
            except Exception:
                pass
        try:
            return self.mt5.candles(pair, 1, 2)[-1]["close"]
        except Exception:
            return 0.0

    def _mirror_shadow(self, trade_id: int, verdict: ResearchVerdict,
                       live_fill: float, intended: float) -> None:
        """Record the parallel shadow trade for slippage analysis."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        try:
            with engine.begin() as conn, _WRITE_LOCK:
                conn.execute(sqltext(
                    "INSERT INTO shadow_trades (created_at, live_trade_id, pair, direction, "
                    "intended_entry, live_fill, shadow_fill, shadow_status) "
                    "VALUES (:t, :lt, :p, :d, :ie, :lf, :sf, 'open')"
                ), {"t": db._utcnow(), "lt": trade_id, "p": verdict.pair,
                    "d": verdict.direction, "ie": intended, "lf": live_fill,
                    "sf": intended})  # demo fill assumed at intended price
        except Exception as exc:
            logger.warning("shadow mirror failed: %s", exc)


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


def _hash_signal(verdict: ResearchVerdict) -> str:
    """Stable signal hash for duplicate protection."""
    import hashlib
    blob = (f"{verdict.pair}|{verdict.direction}|{verdict.strategy}|"
            f"{round(verdict.entry, 5)}").encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def _publish(etype: EventType, payload: dict) -> None:
    """Publish to the async bus from sync code without blocking."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(bus.publish(etype, payload))
    except RuntimeError:
        try:
            asyncio.run(bus.publish(etype, payload))
        except Exception as exc:  # pragma: no cover
            logger.debug("event publish skipped: %s", exc)
