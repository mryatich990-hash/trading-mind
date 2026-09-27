"""Upgrade registry: wires UPGRADES 1-9 into the running system.

One object owns every new module, starts their background loops, exposes the
pre-trade gate (ensemble vote + tick breaker) used by main._process_signal,
matches closed trades to pending DL predictions for accuracy tracking, and
produces the /api/upgrades dashboard payload. All modules degrade gracefully
when their optional dependencies (tensorflow, xgboost, sb3, transformers,
redis, pytrends, twitter token) are missing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

_registry: Optional["UpgradeRegistry"] = None


def get_registry() -> Optional["UpgradeRegistry"]:
    """Process-wide registry (None until init_upgrade_registry ran)."""
    return _registry


class UpgradeRegistry:
    """Owns all upgrade modules + their background loops."""

    def __init__(self, data_engine=None, notifier=None, pairs: Optional[list[str]] = None,
                 event_bus=None) -> None:
        self.data = data_engine
        self.notifier = notifier
        self.pairs = [p.upper() for p in (pairs or settings.TRADING_PAIRS[:5])]
        self.bus = event_bus

        # UPGRADE 1 - deep learning ensemble
        from deep_learning.ensemble_voter import EnsembleVoter

        self.ensemble = EnsembleVoter(pairs=self.pairs)
        self._pending: deque = deque(maxlen=200)

        # UPGRADE 4/5 - scalping + stat arb
        from scalping.scalping_engine import ScalpingEngine
        from arbitrage.stat_arb_engine import StatArbEngine

        self.scalper = ScalpingEngine(ensemble=self.ensemble, notifier=notifier)
        self.statarb = StatArbEngine(data_engine, notifier=notifier)

        # UPGRADE 3 - tick engine + microstructure + breaker
        from tick.tick_engine import TickDataEngine
        from tick.microstructure import MicrostructureAnalyzer
        from tick.tick_circuit_breaker import TickCircuitBreaker

        self.tick = TickDataEngine(data_engine)
        self.micro = MicrostructureAnalyzer(self.tick)
        self.tick_breaker = TickCircuitBreaker()

        # UPGRADE 2 - NLP
        from nlp.finbert_engine import FinBertEngine
        from nlp.central_bank_parser import CentralBankParser
        from nlp.twitter_sentiment import TwitterSentiment
        from nlp.google_trends import GoogleTrends

        self.finbert = FinBertEngine()
        self.central_banks = CentralBankParser(notifier=notifier)
        self.twitter = TwitterSentiment(self.finbert)
        self.trends = GoogleTrends()

        # UPGRADE 6 - analytics
        from analytics.advanced_analytics import AdvancedAnalytics
        from analytics.mae_mfe_analyzer import MAEMFEAnalyzer
        from analytics.trade_quality_scorer import TradeQualityScorer

        self.analytics = AdvancedAnalytics()
        self.maemfe = MAEMFEAnalyzer()
        self.quality = TradeQualityScorer()

        # UPGRADE 7 - infrastructure
        from infrastructure.redis_cache import get_cache
        from infrastructure.hot_reload import HotReloader
        from infrastructure.strategy_versioning import StrategyVersioning
        from infrastructure.performance_profiler import get_profiler
        from infrastructure.dependency_manager import DependencyManager

        self.cache = get_cache()
        self.reloader = HotReloader(notifier=notifier)
        self.versions = StrategyVersioning()
        self.profiler = get_profiler(notifier=notifier)
        self.deps = DependencyManager(notifier=notifier)

        # UPGRADE 8 - smart SL
        from risk.smart_sl_engine import SmartSLEngine

        self.smart_sl = SmartSLEngine()

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._ensure_dl_table()

    # ---- setup ----

    @staticmethod
    def _ensure_dl_table() -> None:
        """Create the dl_predictions store."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine, _WRITE_LOCK, pg_compatible

            with engine.begin() as conn, _WRITE_LOCK:
                conn.execute(sqltext(pg_compatible(
                    "CREATE TABLE IF NOT EXISTS dl_predictions ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                    "pair VARCHAR(16), direction VARCHAR(8), lstm_dir VARCHAR(12), "
                    "lstm_conf DOUBLE, xgb_prob DOUBLE, rl_action VARCHAR(8), "
                    "agreement VARCHAR(16), won INTEGER)")))
        except Exception as exc:
            logger.warning("dl_predictions table unavailable: %s", exc)

    def start(self) -> None:
        """Start every background loop (idempotent)."""
        if self._threads:
            return
        self.tick.start(self.pairs)
        self.reloader.start()
        self.deps.start()
        for name, target, interval in (
                ("nlp-loop", self._nlp_loop, 60),
                ("tick-analytics", self._tick_loop, 15),
                ("outcome-poller", self._outcome_loop, 60),
                ("statarb-loop", self._statarb_loop, 120)):
            thread = threading.Thread(target=self._loop_wrapper,
                                      args=(target, interval), name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info("upgrade registry started (%d modules, %d pairs)",
                    9, len(self.pairs))

    def stop(self) -> None:
        """Signal all loops to exit."""
        self._stop.set()
        self.tick.stop()
        self.reloader.stop()
        self.deps.stop()

    def _loop_wrapper(self, fn, interval: int) -> None:
        while not self._stop.is_set():
            try:
                fn()
            except Exception as exc:
                logger.warning("upgrade loop %s failed: %s", getattr(fn, "__name__", "?"), exc)
            for _ in range(interval * 10):
                if self._stop.is_set():
                    return
                time.sleep(0.1)

    # ---- background loops ----

    def _nlp_loop(self) -> None:
        """Central banks (10 min), twitter (30 min), trends (1h)."""
        now = time.time()
        if not hasattr(self, "_cb_ts") or now - self._cb_ts >= 600:
            self.central_banks.poll()
            self._cb_ts = now
        if self.twitter.active:
            self.twitter.maybe_poll()
        if not hasattr(self, "_trends_ts") or now - self._trends_ts >= 3600:
            self.trends.poll()
            self._trends_ts = now
        self.profiler.flush_hourly()

    def _tick_loop(self) -> None:
        """Microstructure detection + breaker baselines."""
        for pair in self.pairs:
            stats = self.tick.get_stats(pair)
            if stats:
                self.tick_breaker.update_baseline(pair, stats.get("velocity", 0.0))
            events = self.micro.detect(pair)
            for ev in events:
                if self.notifier and ev["type"] == "stop_hunt":
                    self.notifier.send(f"🎯 {ev['type']}: {pair} {ev['side']} "
                                       f"({ev['detail']})")

    def _outcome_loop(self) -> None:
        """Match newly closed trades to pending DL predictions."""
        closed = self._recent_closed_trades(minutes=20)
        for trade in closed:
            pair = trade.get("pair", "")
            direction = trade.get("direction", "")
            for pending in list(self._pending):
                if pending["pair"] == pair and pending["direction"] == direction:
                    self._pending.remove(pending)
                    won = float(trade.get("pnl_usd") or 0.0) > 0
                    self.ensemble.record_outcome(pair, direction,
                                                 pending["verdict"], won)
                    break

    @staticmethod
    def _recent_closed_trades(minutes: int = 20) -> list[dict]:
        """Closed trades in the last N minutes."""
        try:
            import datetime as dt

            from sqlalchemy import text as sqltext

            from core.db import engine

            cutoff = (dt.datetime.now(dt.timezone.utc)
                      - dt.timedelta(minutes=minutes)).isoformat()
            with engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT pair, direction, pnl_usd FROM trades "
                    "WHERE status = 'closed' AND closed_at >= :c"),
                    {"c": cutoff}).mappings().all()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _statarb_loop(self) -> None:
        """Manage open stat-arb positions + scan for new opportunities."""
        for reason in self.statarb.manage():
            if self.notifier:
                self.notifier.send(f"⚖️ StatArb closed: {reason}")
        if self.position_count() == 0:
            for opp in self.statarb.scan()[:1]:
                if self.notifier:
                    self.notifier.send(f"⚖️ StatArb opportunity: {opp['detail']}")

    def position_count(self) -> int:
        """Open trades from the DB (for scalping/statarb interlocks)."""
        try:
            from core import db

            return len(db.open_trades())
        except Exception:
            return 0

    # ---- pre-trade gate (called from main._process_signal) ----

    def pre_trade_gate(self, signal) -> Any:
        """Ensemble vote + tick breaker for a candidate signal."""
        verdict = self.ensemble.vote(
            signal.pair, signal.direction,
            m15_df=None, ctx_dict=None, features=None)
        halted = self.tick_breaker.is_halted(signal.pair)
        if halted:
            verdict.proceed = False
            verdict.reason = f"tick breaker halted {signal.pair}"
        return verdict

    def record_pending(self, signal, verdict) -> None:
        """Track a live verdict until its trade closes (outcome poller)."""
        self._pending.append({"pair": signal.pair, "direction": signal.direction,
                              "verdict": verdict, "ts": time.time()})

    # ---- dashboard ----

    def status(self) -> dict:
        """Full /api/upgrades payload (all seven panels)."""
        try:
            payload = {
                "deep_learning": self.ensemble.status(),
                "nlp": {
                    "finbert": self.finbert.summary(),
                    "central_banks": {"scores": self.central_banks.scores(),
                                      "latest": self.central_banks.latest(5)},
                    "twitter": self.twitter.status(),
                    "trends": self.trends.status(),
                },
                "tick": {"pairs": self.tick.all_stats(),
                         "events": self.micro.recent_events(limit=10),
                         "today": self.micro.today_summary(),
                         "breaker": self.tick_breaker.status()},
                "scalping": self.scalper.status(),
                "stat_arb": self.statarb.status(),
                "analytics": {"ratios": self.analytics.status()["windows"],
                              "mae_mfe": self.maemfe.recommendation(),
                              "quality": self.quality.status()},
                "system_health": {
                    "cache": self.cache.status(),
                    "profiler": self.profiler.status(),
                    "hot_reload": self.reloader.status(),
                    "versions": self.versions.all_current(),
                    "dependencies": self.deps.status(),
                },
            }
            return payload
        except Exception as exc:
            logger.exception("upgrades status failed")
            return {"error": str(exc)[:200]}


def init_upgrade_registry(data_engine=None, notifier=None, pairs: Optional[list[str]] = None,
                          event_bus=None) -> UpgradeRegistry:
    """Create + start the process-wide registry."""
    global _registry
    if _registry is None:
        _registry = UpgradeRegistry(data_engine=data_engine, notifier=notifier,
                                    pairs=pairs, event_bus=event_bus)
        _registry.start()
    return _registry
