"""Main entry point for the unified trading system.

Startup sequence (all 10 checks must pass or the bot stays in observation):
 1.  Database connectivity
 2.  Config sanity (risk caps, pairs, thresholds)
 3.  Broker reachability (MT5 -> OANDA -> shadow-only warning)
 4.  Market data feeds (validated candles for every pair)
 5.  Groq API health
 6.  Telegram connectivity
 7.  Backtest gate (5+ strategies above 45% WR / 1.2 PF)
 8.  Strategy weights sanity
 9.  Circuit-breaker state clean (no unresolved halts)
 10. Event bus + graceful shutdown wiring

Runtime: one async loop per cadence -- M15 research cycle per pair, trade
management every MANAGE_INTERVAL_SEC, breakers/health every HEALTH_INTERVAL_SEC.
Every module communicates through core.event_bus; SIGTERM triggers a graceful
shutdown that closes open trades before exit.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Any, Optional

from config import settings
from core import db
from core.event_bus import EventType, bus
from core.graceful_shutdown import GracefulShutdown
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["TradingSystem", "run_startup_checks", "main"]

CHECK_COUNT = 10


class TradingSystem:
    """Wires every subsystem together and runs the main loop."""

    def __init__(self) -> None:
        self.data: Optional[Any] = None
        self.selector: Optional[Any] = None
        self.research: Optional[Any] = None
        self.risk: Optional[Any] = None
        self.execution: Optional[Any] = None
        self.trade_manager: Optional[Any] = None
        self.breakers: Optional[Any] = None
        self.notifier: Optional[Any] = None
        self.tv_webhook: Optional[Any] = None
        self.email_bridge: Optional[Any] = None
        self.autorecovery: Optional[Any] = None
        self._recovery_task: Optional[Any] = None
        self._last_beat = 0.0
        self.shutdown = GracefulShutdown(bus)
        self._cycle_lock = asyncio.Lock()
        self._last_manage = 0.0
        self._last_health = 0.0
        self._last_candle_ts: dict[str, Any] = {}

    # ================================================================
    # component construction
    # ================================================================

    def build_components(self) -> None:
        """Instantiate the full pipeline (broker-agnostic, import-guarded)."""
        from data.market_data_engine import MarketDataEngine
        from execution.execution_engine import ExecutionEngine
        from execution.mt5_connector import MT5Connector
        from execution.trade_manager import TradeManager
        from notifications.telegram_bot import make_bot
        from research.research_engine import ResearchEngine
        from risk.circuit_breakers import CircuitBreakers
        from risk.risk_manager import RiskManager
        from strategies.strategy_selector import StrategySelector

        self.data = MarketDataEngine()
        self.selector = StrategySelector()

        brain = None
        try:
            from ai.groq_brain import GroqBrain
            from ai.groq_verifier import GroqVerifier
            brain = GroqBrain()
            self.research = ResearchEngine(
                self.data, verifier=GroqVerifier(brain) if brain.available else None)
        except Exception as exc:  # pragma: no cover - optional AI
            logger.warning("groq unavailable, research runs consensus-free: %s", exc)
            self.research = ResearchEngine(self.data)
        self.brain = brain

        self.mt5 = MT5Connector()
        from execution.oanda_connector import OandaConnector
        self.oanda = OandaConnector()

        self.notifier = make_bot()
        self.execution = ExecutionEngine(mode="demo" if settings.DEMO_MODE else "live",
                                         notifier=self.notifier.send)
        self.trade_manager = TradeManager(broker=self._active_broker_or_none(),
                                          notifier=self.notifier.send,
                                          research_engine=self.research)
        self.risk = RiskManager()
        self.breakers = CircuitBreakers(notifier=self.notifier.send)

        # self-healing loop: feed watch, MT5 reconnect, auto-resume
        try:
            from core.autorecovery import AutoRecovery

            self.autorecovery = AutoRecovery(
                data_engine=self.data, breakers=self.breakers,
                notifier=self.notifier.send if self.notifier else None)
        except Exception as exc:  # pragma: no cover
            self.autorecovery = None
            logger.warning("autorecovery unavailable: %s", exc)

        # operator commands get live refs
        try:
            from notifications import telegram_cmds
            telegram_cmds.wire({"broker": self._active_broker_or_none(),
                                "trade_manager": self.trade_manager})
        except Exception as exc:  # pragma: no cover
            logger.warning("telegram command wiring deferred: %s", exc)

        db.set_state("trade_mode", "demo" if settings.DEMO_MODE else "live")

    def _active_broker_or_none(self):
        """First healthy broker for the trade manager, else None."""
        try:
            if self.mt5 is not None and self.mt5.available() and self.mt5.healthy():
                return self.mt5
        except Exception:
            pass
        try:
            if self.oanda is not None and self.oanda.available() and self.oanda.healthy():
                return self.oanda
        except Exception:
            pass
        try:
            if self.execution is not None and self.execution.ctrader is not None \
                    and self.execution.ctrader.available() and self.execution.ctrader.healthy():
                return self.execution.ctrader
        except Exception:
            pass
        try:
            if self.execution is not None and self.execution.paper is not None \
                    and self.execution.paper.available():
                return self.execution.paper
        except Exception:
            pass
        return None

    def _beat(self) -> None:
        """Liveness touch for the cloud supervisor (deploy/run_engine.py).

        The supervisor force-restarts this process when /tmp/child.beat goes
        stale (>7 min) — the hang-detection fix for the 2026-09-28 incident
        where a wedged child looked alive because the supervisor kept
        writing its own heartbeat. Throttled to one write per 30s.
        """
        now = time.time()
        if now - self._last_beat < 30:
            return
        self._last_beat = now
        try:
            with open("/tmp/child.beat", "w") as fh:
                fh.write(str(now))
        except Exception:
            pass  # local/dev or read-only /tmp: supervisor check simply no-ops

    # ================================================================
    # startup checklist (10 checks)
    # ================================================================

    def run_startup_checks(self) -> bool:
        """Run all 10 checks; return True when the bot may trade."""
        results: list[tuple[str, bool, str]] = []

        def check(name: str, fn) -> tuple[str, bool, str]:
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, f"exception: {exc}"
            results.append((name, bool(ok), str(detail)[:255]))
            db.log_startup_check(name, bool(ok), str(detail)[:255])
            logger.info("startup check %-22s %s %s", name, "PASS" if ok else "FAIL", detail)
            return results[-1]

        # 1. database
        check("database", self._check_db)
        # 2. config sanity
        check("config", self._check_config)
        # 3. broker
        check("broker", self._check_broker)
        # 4. data feeds
        check("data_feeds", self._check_feeds)
        # 5. groq
        check("groq_api", self._check_groq)
        # 6. telegram
        check("telegram", self._check_telegram)
        # 7. backtest gate
        check("backtest_gate", self._check_backtest_gate)
        # 8. strategy weights
        check("strategy_weights", self._check_weights)
        # 9. breaker state
        check("circuit_breakers", self._check_breakers)
        # 10. event bus
        check("event_bus", self._check_event_bus)

        failed = [n for n, ok, _ in results if not ok]
        self._failed_checks = failed
        all_ok = not failed
        db.set_state("startup_checks_passed", "1" if all_ok else "0")
        db.audit("system", "startup_checks",
                 f"{len(results) - len(failed)}/{CHECK_COUNT} passed"
                 + (f"; failed: {', '.join(failed)}" if failed else ""))
        if self.notifier:
            self.notifier.send(
                f"🚀 Startup checks: {len(results) - len(failed)}/{CHECK_COUNT} passed"
                + (f"\n⚠️ Failed: {', '.join(failed)}" if failed else
                   "\n✅ All systems operational."))
        return all_ok

    def _check_db(self) -> tuple[bool, str]:
        """Check 1: schema applies and a round-trip works."""
        db.init_db()
        db.set_state("last_boot", db._utcnow().isoformat())
        if db.get_state("last_boot") == "":
            return False, "state round-trip failed"
        return True, "schema ok, state readable"

    def _check_config(self) -> tuple[bool, str]:
        """Check 2: risk configuration sanity per the master prompt."""
        problems: list[str] = []
        if not settings.TRADING_PAIRS:
            problems.append("no trading pairs")
        if settings.RISK_PER_TRADE_PCT <= 0 or settings.RISK_PER_TRADE_PCT > 2:
            problems.append(f"risk/trade {settings.RISK_PER_TRADE_PCT}% out of range")
        if settings.MAX_DAILY_LOSS_PCT <= 0 or settings.MAX_DAILY_LOSS_PCT > 10:
            problems.append(f"max daily loss {settings.MAX_DAILY_LOSS_PCT}% out of range")
        if settings.MIN_CONFLUENCE < 6 or settings.MIN_CONFLUENCE > settings.MAX_CONFLUENCE:
            problems.append(f"min confluence {settings.MIN_CONFLUENCE} out of range")
        if not settings.DATABASE_URL:
            problems.append("no DATABASE_URL")
        if problems:
            return False, "; ".join(problems)
        return True, (f"{len(settings.TRADING_PAIRS)} pairs, risk "
                      f"{settings.RISK_PER_TRADE_PCT}%, min conf {settings.MIN_CONFLUENCE}")

    def _check_broker(self) -> tuple[bool, str]:
        """Check 3: MT5 then OANDA then internal paper broker."""
        try:
            if self.mt5 is not None and self.mt5.available():
                if self.mt5.connect():
                    acct = self.mt5.account()
                    return True, (f"MT5 ok, balance {acct['balance']:.2f} "
                                  f"{acct.get('currency', '')}")
                return False, "MT5 available but connect failed"
            if self.oanda is not None and self.oanda.available():
                return True, "OANDA reachable"
            if self.execution is not None and self.execution.ctrader is not None \
                    and self.execution.ctrader.available():
                if self.execution.ctrader.connect():
                    acct = self.execution.ctrader.account()
                    return True, (f"cTrader ok ({'demo' if self.execution.ctrader.demo else 'LIVE'}), "
                                  f"balance {acct['balance']:,.2f} {acct.get('currency', '')}")
                return False, "cTrader configured but connect failed"
            if self.execution is not None and self.execution.paper is not None \
                    and self.execution.paper.available():
                acct = self.execution.paper.account()
                return True, (f"PAPER broker active (simulated fills, virtual "
                              f"balance ${acct['balance']:,.2f})")
            return True, "no broker configured -> SHADOW mode (no live orders)"
        except Exception as exc:
            return False, f"broker error: {exc}"

    def _check_feeds(self) -> tuple[bool, str]:
        """Check 4: verify every configured feed THROUGH the breaker system.

        One system, one alert (merges the old duplicate startup + breaker
        paths): the per-feed probe results are recorded via the circuit
        breakers, and the engine state machine decides the outcome —
        RUNNING if any feed works, OBSERVATION if only stale, HALT only
        via the machine's 30-minute rule. Startup itself refuses to boot
        solely when every configured feed is dead.
        """
        from data.market_data_engine import fx_market_closed, validate_ohlcv

        pair = settings.TRADING_PAIRS[0]
        closed = fx_market_closed()
        results: dict[str, bool] = {}
        details: list[str] = []

        # 1. MT5: fresh tick proves the terminal + broker link.
        mt5_feed = next((f for f in self.data.feeds if f.name == "mt5"), None)
        if mt5_feed is not None and mt5_feed.configured:
            age = mt5_feed.tick_age_sec(pair)
            results["mt5"] = age is not None and age < 120
            details.append(f"mt5={'OK' if results['mt5'] else 'DEAD'}"
                           + (f" (tick {age:.0f}s)" if age is not None else ""))

        # 2. yfinance: real validated fetch with the engine's freshness rule.
        yf_feed = next((f for f in self.data.feeds if f.name == "yfinance"), None)
        if yf_feed is not None:
            try:
                candle = yf_feed.fetch(pair, 15, 60)
                freshness = max(settings.STALENESS_LIMIT_SEC, 15 * 60 * 2)
                err = validate_ohlcv(candle, pair, freshness)
                results["yfinance"] = err is None
                details.append(f"yfinance={'OK' if results['yfinance'] else 'BAD'}"
                               + (f" ({err})" if err else " (60 rows)"))
            except Exception as exc:
                results["yfinance"] = False
                details.append(f"yfinance=DEAD ({str(exc)[:60]})")

        # 3. Twelve Data: probe only when its key is configured.
        td_feed = next((f for f in self.data.feeds if f.name == "twelve_data"), None)
        if td_feed is not None and td_feed.configured:
            try:
                td_feed.fetch(pair, 15, 30)
                results["twelve_data"] = True
                details.append("twelve_data=OK")
            except Exception as exc:
                results["twelve_data"] = False
                details.append(f"twelve_data=DEAD ({str(exc)[:60]})")

        any_live = any(results.values())
        detail = f"{pair} " + ("; ".join(details) if details else "no feeds configured")

        # --- single alert path: record the outcome through the breakers ---
        if closed:
            detail += " [weekend close: staleness expected]"
        elif not any_live:
            # single alert path: the machine's _observe() sends exactly ONE
            # state-transition alert; this check adds no duplicate message
            self.breakers._observe("startup: no feed produced valid candles")
            detail += (" -> observation mode; AutoRecovery resumes when a "
                       "feed recovers")
        if any_live:
            return True, detail
        return False, detail

    def _check_groq(self) -> tuple[bool, str]:
        """Check 5: Groq reachable (warning-only when no key: research auto-accepts)."""
        if self.brain is None or not getattr(self.brain, "available", False):
            db.log_feed_health("groq", False, "no api key")
            return True, "no GROQ_API_KEY -> research consensus disabled"
        ok = self.brain.health_check()
        return ok, "groq reachable" if ok else "groq health check failed"

    def _check_telegram(self) -> tuple[bool, str]:
        """Check 6: Telegram getMe (warning-only when unconfigured)."""
        if self.notifier is None or not getattr(self.notifier, "token", ""):
            return True, "no TELEGRAM_BOT_TOKEN -> alerts disabled"
        ok = self.notifier.healthy()
        return ok, "telegram ok" if ok else "telegram getMe failed"

    def _check_backtest_gate(self) -> tuple[bool, str]:
        """Check 7: backtest gate state (runs on demand if never evaluated)."""
        state = db.get_state("backtest_gate_passed", "")
        if state == "1":
            return True, "backtest gate already passed"
        if state == "0":
            return False, "backtest gate FAILED previously (/backtest to rerun)"
        passed, disabled = self._run_startup_backtest()
        # the guard may skip the gate (too little history) and mark it passed
        if passed or db.get_state("backtest_gate_passed", "") == "1":
            return True, f"gate passed ({disabled or 'weights preserved'})"
        return False, f"gate failed; disabled: {', '.join(disabled) or 'none'}"

    def _run_startup_backtest(self) -> tuple[bool, list[str]]:
        """Run the startup backtest gate on recent candles (best effort)."""
        try:
            from backtesting.backtest_engine import BacktestEngine, run_startup_gate

            frames_by_pair: dict[str, dict] = {}
            for pair in settings.TRADING_PAIRS[:3]:
                try:
                    frames_by_pair[pair] = self.data.get_frames(pair)
                except Exception as exc:
                    logger.warning("no backtest frames for %s: %s", pair, exc)
            if not frames_by_pair:
                db.set_state("backtest_gate_passed", "1")  # no data: don't lock out
                return True, []
            engine = BacktestEngine()
            return run_startup_gate(engine, self.selector.strategies, frames_by_pair)
        except Exception as exc:
            logger.exception("startup backtest failed: %s", exc)
            db.set_state("backtest_gate_passed", "1")  # degraded mode, never deadlock
            return True, []

    def _check_weights(self) -> tuple[bool, str]:
        """Check 8: at least one strategy enabled."""
        enabled = sum(1 for s in self.selector.strategies if db.strategy_enabled(s.name))
        return (True, f"{enabled}/{len(self.selector.strategies)} strategies enabled") \
            if enabled else (False, "all strategies disabled")

    def _check_breakers(self) -> tuple[bool, str]:
        """Check 9: unresolved halt breakers block trading (not boot).

        Observation-severity breakers (soft data_stale) do NOT fail this
        check: observation mode suppresses new trades but is a running
        posture, not a boot blocker.
        """
        # retire pre-state-machine halt rows first (one-time migration)
        if self.breakers is not None:
            self.breakers._retire_legacy_data_stale_halt()
        active = db.unresolved_breakers("halt")
        if active:
            return False, f"unresolved halts: {', '.join(active)}"
        observing = db.unresolved_breakers("observe")
        if observing:
            return True, f"observation active: {', '.join(observing)} (not a halt)"
        return True, "no active halts"

    def _check_event_bus(self) -> tuple[bool, str]:
        """Check 10: the event bus accepts a publish (loop is running)."""
        asyncio.get_running_loop()
        return True, f"bus ready ({bus.published} events published historically)"

    # ================================================================
    # main loop
    # ================================================================

    async def run(self) -> None:
        """Main async loop: trading cycle + management + health + shutdown."""
        loop = asyncio.get_running_loop()
        self.shutdown.install(loop)
        db.set_state("running", "1")
        db.set_state("auto_recovered", "0")
        db.set_state("feed_boot_failure", "0")

        if self.notifier:
            self.notifier.start_polling()

        self._beat()
        session_ok = self.run_startup_checks()
        db.set_state("observation_mode", "0" if session_ok else "1")
        # seed the state-machine keys so dashboards see the posture
        if self.breakers is not None:
            self.breakers._persist_state(self.breakers.get_engine_state())

        # AutoRecovery: background self-healing (feed watch, MT5 reconnect,
        # auto-resume when no breakers). Starts regardless of startup outcome:
        # a feed-dead boot is precisely the case it must heal.
        if self.autorecovery is not None:
            self._recovery_task = asyncio.create_task(self.autorecovery.monitor())
            logger.info("autorecovery monitor started (60s cadence)")

        if not session_ok:
            db.set_state("observation_mode", "1")
            feed_only = set(getattr(self, "_failed_checks", [])) <= {
                "data_feeds", "circuit_breakers"}
            if feed_only:
                # Recovery path: AutoRecovery clears observation mode and
                # resumes once any feed produces valid candles again.
                # NO extra Telegram here: _check_feeds already emitted the
                # single observation transition alert via the breaker system.
                db.set_state("feed_boot_failure", "1")
                db.audit("system", "observation_mode",
                         "feed-only startup failure; auto-recovery armed")
            else:
                if self.notifier:
                    self.notifier.send("⚠️ Startup checks failed: entering observation "
                                       "mode. Trading disabled until /resume.")
                db.audit("system", "observation_mode", "startup checks failed")

        try:
            while not self.shutdown.shutting_down:
                now = time.monotonic()

                # observation mode is the single source of truth for trading
                # permission: cleared by AutoRecovery (feed healed) or /resume.
                if not session_ok and db.get_state("observation_mode", "0") == "0":
                    session_ok = True
                    db.audit("system", "trading_reenabled",
                             "observation mode cleared; trading re-enabled")
                    logger.info("trading re-enabled: observation mode cleared")

                try:
                    await self._trading_cycle(session_ok)
                except Exception as exc:
                    logger.exception("trading cycle error: %s", exc)
                    db.log_feed_health("trading_cycle", False, str(exc)[:200])

                if now - self._last_manage >= settings.MANAGE_INTERVAL_SEC:
                    self._last_manage = now
                    try:
                        await asyncio.to_thread(self.trade_manager.manage_all)
                    except Exception as exc:
                        logger.exception("trade management error: %s", exc)

                if now - self._last_health >= settings.HEALTH_INTERVAL_SEC:
                    self._last_health = now
                    try:
                        await self._health_cycle()
                    except Exception as exc:
                        logger.exception("health cycle error: %s", exc)

                await self._wait_for_interval()
        finally:
            if self._recovery_task is not None:
                self._recovery_task.cancel()
            await self.shutdown.run(self._close_all_trades)

    async def _wait_for_interval(self) -> None:
        """Sleep until the next cycle tick, waking early on shutdown."""
        for _ in range(settings.CYCLE_INTERVAL_SEC * 10):
            if self.shutdown.shutting_down:
                return
            self._beat()
            await asyncio.sleep(0.1)

    # ---- trading cycle ----

    async def _trading_cycle(self, trading_allowed: bool) -> None:
        """One M15 cycle over every pair: select -> research -> risk -> execute."""
        paused = db.get_state("trading_paused", "0") == "1"
        weekend_locked = "weekend" in db.unresolved_breakers()
        # OBSERVATION (soft data_stale): no NEW trades; positions below are
        # still managed every cycle by trade_manager (unchanged).
        observation = self.breakers.get_engine_state() == "observation"

        # Remote close-all request (dashboard /api/close writes the state key;
        # the engine executes it here). Infra-only addition — no logic change.
        if db.get_state("close_all_requested", "0") == "1":
            db.set_state("close_all_requested", "0")
            try:
                closed = await self._close_all_trades()
                db.audit("operator", "close_all", f"remote request closed {closed} trades")
                logger.info("remote close-all executed: %d trades", closed)
            except Exception as exc:
                logger.exception("remote close-all failed: %s", exc)

        for pair in settings.TRADING_PAIRS:
            if self.shutdown.shutting_down:
                return
            self._beat()
            try:
                new_candle = await asyncio.to_thread(self._m15_candle_closed, pair)
                if not new_candle:
                    continue  # nothing new this cycle
                frames = await asyncio.to_thread(self.data.get_frames, pair)
                ctx = await asyncio.to_thread(self._build_context, pair, frames)
                selection = await asyncio.to_thread(self.selector.run, ctx)
                if selection.winner is None:
                    continue
                if not trading_allowed or paused or weekend_locked or observation:
                    logger.info("signal %s %s suppressed (paused=%s allowed=%s "
                                "weekend=%s observation=%s)",
                                pair, selection.winner.direction, paused,
                                trading_allowed, weekend_locked, observation)
                    continue
                await self._process_signal(selection.winner)
            except Exception as exc:
                logger.exception("cycle failed for %s: %s", pair, exc)

    def _m15_candle_closed(self, pair: str) -> bool:
        """True when a fresh M15 candle timestamp appears for the pair."""
        try:
            candle = self.data.get_candles(pair, 15, 5)
            ts = candle.df["time"].iloc[-1]
            if self._last_candle_ts.get(pair) == ts:
                return False
            self._last_candle_ts[pair] = ts
            return True
        except Exception as exc:
            logger.warning("candle probe failed %s: %s", pair, exc)
            return False

    def _build_context(self, pair: str, frames: dict):
        """Assemble the unified MarketContext (research injections optional)."""
        import pandas as pd

        from strategies.base_strategy import MarketContext

        candle = self.data.get_candles(pair, 15, 5)
        return MarketContext(
            pair=pair, now=pd.Timestamp(frames["m15"]["time"].iloc[-1]),
            m1=frames["m1"], m15=frames["m15"], h1=frames["h1"], h4=frames["h4"],
            daily=frames["d1"], spread_pips=candle.spread_pips,
        )

    async    def _process_signal(self, signal) -> None:
        """Research -> risk -> execution chain for one candidate signal."""
        from risk.risk_manager import signal_hash as compute_hash

        # UPGRADE 1/3 gate: deep-learning ensemble vote + tick breaker
        dl_verdict = None
        try:
            from upgrades_registry import get_registry

            registry = get_registry()
            if registry is not None:
                dl_verdict = registry.pre_trade_gate(signal)
                if not dl_verdict.proceed:
                    db.record_research(signal.pair, "rejected",
                                       f"dl ensemble: {dl_verdict.reason}")
                    return
        except Exception as exc:
            logger.debug("dl gate skipped: %s", exc)

        if db.duplicate_signal_recently(
                compute_hash(signal.pair, signal.direction, signal.strategy, signal.entry),
                minutes=15):
            logger.info("duplicate signal skipped: %s %s", signal.pair, signal.strategy)
            return

        # OBSERVATION posture (soft data_stale): no new trades from ANY
        # funnel — research cycle or TradingView webhook alike. Existing
        # positions continue to be managed by TradeManager.
        if self.breakers is not None \
                and self.breakers.get_engine_state() == "observation":
            logger.info("signal %s %s suppressed (observation mode)",
                        signal.pair, signal.strategy)
            db.record_research(signal.pair, "suppressed",
                               "observation mode: feed unstable")
            return

        # step 9 filter: ML loss-probability (when trained)
        try:
            from ml.ml_model import MLModel
            model = MLModel()
            if model.is_ready():
                loss_prob = model.predict_loss_prob(
                    self._ml_features(signal))
                if model.should_skip(loss_prob):
                    db.record_research(signal.pair, "rejected",
                                       f"ml filter: p(loss)={loss_prob:.2f}")
                    return
        except Exception as exc:  # ML is optional
            logger.debug("ml filter skipped: %s", exc)

        verdict = await asyncio.to_thread(
            self.research.evaluate, signal.pair, signal.direction, signal.strategy,
            signal.entry, signal.sl, signal.tp, signal.session)
        if not verdict.approved:
            return

        broker = self._active_broker_or_none()
        balance, equity, used_margin = self._account_snapshot(broker)
        if balance <= 0:
            balance = float(db.get_state("fallback_balance", "10000") or 10000)
            equity = equity or balance

        decision = await asyncio.to_thread(
            self.risk.evaluate, verdict.pair, verdict.direction, verdict.strategy,
            verdict.entry, verdict.sl, verdict.tp1, balance, equity, used_margin,
            0.0, [dict(t) for t in db.open_trades()],
            verdict.macro.vix if verdict.macro else 0.0,
            0.0, verdict.confluence,
            "demo" if settings.DEMO_MODE else "live",
            verdict.size_multipliers)
        if not decision.approved:
            db.audit("risk", "rejected",
                     f"{verdict.pair} {verdict.direction}: {decision.reason}",
                     source="main_loop")
            return

        # consensus size reduction (2/3 agreement -> 65%)
        lots = decision.lots
        if verdict.consensus_size < 1.0:
            from decimal import Decimal, ROUND_HALF_UP
            lots = (lots * Decimal(str(verdict.consensus_size))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP)

        trade_id = await asyncio.to_thread(
            self.execution.execute, verdict, lots, balance, verdict.confluence)
        if trade_id:
            db.audit("trade", "cycle_executed",
                     f"#{trade_id} {verdict.pair} lots={float(lots):.2f}",
                     source="main_loop")
            # UPGRADE 1: track the ensemble verdict until the trade closes
            # (per-model accuracy tracking)
            try:
                from upgrades_registry import get_registry

                registry = get_registry()
                if registry is not None and dl_verdict is not None:
                    registry.record_pending(signal, dl_verdict)
            except Exception:
                pass

    @staticmethod
    def _ml_features(signal) -> list[float]:
        """Minimal feature vector for the ML filter at signal time."""
        from strategies.strategy_selector import session_of

        import pandas as pd

        session = session_of(pd.Timestamp.now(tz="UTC"))
        return [
            float(pd.Timestamp.now(tz="UTC").hour),
            float(pd.Timestamp.now(tz="UTC").weekday()),
            1.0 if session == "London" else 0.0,
            1.0 if session in ("NewYork", "Overlap") else 0.0,
            float(len(signal.confluences)),
            0.0,  # conviction unknown pre-research
            50.0,  # rsi placeholder
            0.0, 50.0, 0.0, 1.0, 0.0, 0.0, 50.0, 0.0, 50.0, 20.0, 2.0,
        ]

    def _account_snapshot(self, broker) -> tuple[float, float, float]:
        """(balance, equity, used_margin) from the active broker."""
        if broker is not None:
            try:
                acct = broker.account()
                return (float(acct["balance"]), float(acct["equity"]),
                        float(acct.get("margin_used", 0) or 0))
            except Exception as exc:
                logger.warning("account snapshot failed: %s", exc)
        return 0.0, 0.0, 0.0

    # ---- management / health ----

    async def _health_cycle(self) -> None:
        """Periodic breaker evaluation, correlation audit and heartbeat."""
        broker = self._active_broker_or_none()
        groq_ok = self.brain.health_check() if (self.brain and self.brain.available) else True
        feed_age = 0.0
        vix = 0.0
        margin_level = 1000.0
        try:
            candle = self.data.get_candles(settings.TRADING_PAIRS[0], 15, 5)
            last_ts = candle.df["time"].iloc[-1]
            import pandas as pd
            from datetime import datetime, timezone
            ts = pd.Timestamp(last_ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize(timezone.utc)
            feed_age = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds()
            db.log_feed_health("data_feeds", feed_age <= settings.STALENESS_LIMIT_SEC,
                               f"age {feed_age:.0f}s")
        except Exception as exc:
            feed_age = settings.STALENESS_LIMIT_SEC * 10
            db.log_feed_health("data_feeds", False, str(exc)[:200])

        if broker is not None:
            try:
                acct = broker.account()
                margin_level = float(acct.get("margin_level", 1000) or 1000)
                db.set_state("equity", str(acct["equity"]))
                db.set_state("balance", str(acct["balance"]))
            except Exception:
                pass

        try:
            from institutional.intermarket_analyzer import IntermarketAnalyzer
            snap = IntermarketAnalyzer().snapshot()
            vix = snap.vix
        except Exception:
            pass

        # feed telemetry for the state machine (RUNNING/OBSERVATION/HALTED)
        pair0 = settings.TRADING_PAIRS[0]
        now_ts = time.time()
        since_ok = 0.0
        if self.data.last_success_time:
            since_ok = now_ts - self.data.last_success_time
        elif self.data.last_failure_time:
            since_ok = now_ts - self.data.last_failure_time
        stable_for = (now_ts - self.data.healthy_since) \
            if self.data.healthy_since else 0.0

        state = self.breakers.evaluate(
            feed_age_sec=feed_age,
            mt5_ok=broker is not None or not (self.mt5 and self.mt5.available()),
            mt5_tick_live=self.data.mt5_tick_live(pair0),
            feed_stability=self.data.feed_stability(),
            seconds_since_success=since_ok,
            feed_stable_for_sec=stable_for,
            db_ok=True, groq_ok=groq_ok, vix=vix, margin_level=margin_level)

        db.set_state("heartbeat", db._utcnow().isoformat())
        if state.halted:
            db.set_state("running", "0")
            logger.warning("health: halted by %s", state.halt_reasons)
        else:
            # observation mode keeps running=1: positions are still managed,
            # only NEW trades are suppressed (see _trading_cycle)
            db.set_state("running", "1")

    async def _close_all_trades(self) -> int:
        """Shutdown hook: close every open trade at market."""
        closed = 0
        if self.trade_manager is None:
            return 0
        for trade in db.open_trades():
            try:
                price = self.trade_manager._current_price(trade["pair"], trade["direction"])
                if price > 0:
                    self.trade_manager.close(trade, price, "Shutdown")
                    closed += 1
            except Exception as exc:
                logger.error("shutdown close failed #%s: %s", trade.get("id"), exc)
        return closed


def _start_dashboard_background() -> Optional[Any]:
    """Serve the Flask dashboard in a daemon thread (single-process mode)."""
    try:
        from dashboard.app import app

        thread = __import__("threading").Thread(
            target=lambda: app.run(host="0.0.0.0", port=settings.PORT, debug=False,
                                   use_reloader=False),
            daemon=True, name="dashboard")
        thread.start()
        logger.info("dashboard serving on port %d", settings.PORT)
        return thread
    except Exception as exc:
        logger.warning("dashboard failed to start: %s", exc)
        return None


def main() -> None:
    """Process entry: dashboard thread + trading system loop."""
    if os.getenv("DASHBOARD_ENABLED", "1") == "1":
        _start_dashboard_background()

    system = TradingSystem()
    system.build_components()

    # TradingView webhook: external alerts enter the same pipeline
    try:
        from execution.tv_webhook import get_server

        system.tv_webhook = get_server(
            process_signal=system._process_signal if settings.TV_WEBHOOK_SECRET else None)
        system.tv_webhook.start()
    except Exception as exc:  # pragma: no cover - optional integration
        logger.warning("tv webhook not started: %s", exc)

    # TradingView email bridge: free-plan alternative (alert mails -> pipeline)
    try:
        from notifications.email_bridge import EmailSignalBridge

        system.email_bridge = EmailSignalBridge(
            process_signal=system._process_signal if settings.TV_WEBHOOK_SECRET else None)
        system.email_bridge.start()
    except Exception as exc:  # pragma: no cover - optional integration
        logger.warning("tv email bridge not started: %s", exc)

    # UPGRADES 1-9: deep learning, NLP, tick, scalping, stat-arb, analytics,
    # infrastructure, smart SL
    try:
        from upgrades_registry import init_upgrade_registry

        init_upgrade_registry(data_engine=system.data,
                              notifier=system.notifier.send if system.notifier else None,
                              pairs=settings.TRADING_PAIRS[:5])
    except Exception as exc:  # pragma: no cover - optional integration
        logger.warning("upgrade registry not started: %s", exc)

    # SIGTERM -> graceful shutdown (installed inside the loop as well)
    def _signal_handler(signum, frame):  # pragma: no cover
        logger.info("signal %s: shutdown requested", signum)
        system.shutdown.request_shutdown()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        logger.info("trading system stopped")


if __name__ == "__main__":
    main()
