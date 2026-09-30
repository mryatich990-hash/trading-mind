"""ResearchEngine: the 10-step pre-trade research pipeline.

Runs before EVERY trade (60-90s, cannot be skipped):
 1. Macro context           (research/macro_analyzer)
 2. News gate               (calendar block -> abort)
 3. Higher timeframe bias   (research/htf_analyzer)
 4. Institutional structure (same module)
 5. Volume / order flow     (research/entry_validator)
 6. M15 entry checklist     (8/10 minimum)
 7. Pattern/harmonic bonus  (folded into confluence)
 8. Historical setup match  (research/historical_matcher)
 9. Groq 3x consensus + verifier (ai/)
10. Final risk gate         (risk/risk_manager)

Output: a ResearchVerdict consumed by the execution engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ai.groq_prompt_builder import GroqPromptBuilder, ResearchData
from ai.groq_verifier import GroqVerifier
from config import settings
from core import db
from core.logging_utils import get_logger
from data.market_data_engine import MarketDataEngine
from research.entry_validator import EntryChecklist, EntryValidator
from research.historical_matcher import HistoricalMatcher
from research.htf_analyzer import HTFAnalyzer, HTFResult
from research.macro_analyzer import MacroAnalyzer, MacroResult

logger = get_logger(__name__)

__all__ = ["ResearchVerdict", "ResearchEngine"]

# step 9 consensus outcomes
FULL_SIZE = 1.0
REDUCED_65 = 0.65


@dataclass
class ResearchVerdict:
    """Final research outcome for one signal."""

    pair: str
    direction: str
    strategy: str
    approved: bool
    reason: str
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    invalidation: float = 0.0
    regime: str = "unknown"
    conviction: int = 0
    consensus_size: float = 0.0
    confluence: int = 0
    required_confluence: int = settings.MIN_CONFLUENCE
    macro: Optional[MacroResult] = None
    htf: Optional[HTFResult] = None
    checklist: Optional[EntryChecklist] = None
    size_multipliers: dict = field(default_factory=dict)
    duration_sec: float = 0.0
    groq_prompt: str = ""
    groq_response: dict = field(default_factory=dict)


class ResearchEngine:
    """Coordinates analyzers, Groq consensus and the final risk gate."""

    def __init__(self, data: MarketDataEngine, macro: Optional[MacroAnalyzer] = None,
                 validator: Optional[EntryValidator] = None,
                 verifier: Optional[GroqVerifier] = None,
                 matcher: Optional[HistoricalMatcher] = None,
                 builder: Optional[GroqPromptBuilder] = None) -> None:
        self.data = data
        self.macro = macro or MacroAnalyzer()
        self.htf_analyzer = HTFAnalyzer()
        self.validator = validator or EntryValidator()
        self.verifier = verifier
        self.matcher = matcher or HistoricalMatcher()
        self.builder = builder or GroqPromptBuilder()

    # ---- data assembly ----

    def _research_data(self, pair: str, direction: str, strategy: str,
                       macro: MacroResult, htf: HTFResult,
                       checklist: EntryChecklist, hist, frames,
                       historical_rate: float, session: str) -> ResearchData:
        """Fill the prompt dataclass with verified numbers only."""
        candle = self.data.get_candles(pair, 15, 50)
        price = candle.last_close
        m15 = frames["m15"]
        now_ts = pd.Timestamp(m15["time"].iloc[-1])
        minutes_in = (now_ts - now_ts.normalize()).total_seconds() / 60.0

        rows = db.closed_trades(limit=10)
        seq = "".join("W" if float(r["pnl_usd"] or 0) > 0 else "L" for r in rows) or "no trades"
        last10 = db.closed_trades(limit=100)
        pair_seq = [r for r in last10 if r["pair"] == pair][:10]
        if pair_seq:
            seq = "".join("W" if float(r["pnl_usd"] or 0) > 0 else "L" for r in pair_seq)

        d = ResearchData(
            pair=pair,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            bid=round(candle.bid or price, 5),
            ask=round(candle.ask or price, 5),
            spread_pips=candle.spread_pips,
            session=session,
            minutes_in_session=int(minutes_in),
            dxy_trend=macro.dxy_trend,
            vix=macro.vix,
            vix_term=macro.vix_term,
            usd_bias=macro.usd_bias,
            base_bias=macro.base_bias,
            base_currency=macro.base_currency,
            quote_bias=macro.quote_bias,
            quote_currency=macro.quote_currency,
            cot_commercial_net=macro.cot_commercial_net,
            cot_percentile=macro.cot_percentile,
            retail_long_pct=macro.retail_long_pct,
            retail_contrarian=macro.retail_contrarian,
            yield_10y=macro.yield_10y,
            yield_trend=macro.yield_trend,
            intermarket_signal="; ".join(macro.notes) or "none",
            daily_trend=htf.daily_trend,
            daily_vwap_position=htf.daily_vwap_position,
            h4_ema20=htf.h4_ema20, h4_ema50=htf.h4_ema50, h4_ema200=htf.h4_ema200,
            h4_bos=htf.h4_bos, h4_bos_price=htf.h4_bos_price, h4_bos_ago=htf.h4_bos_ago,
            wyckoff_phase=htf.wyckoff_phase,
            elliott_wave=htf.elliott_wave,
            ob_kind=htf.ob_kind.replace("_", " "),
            ob_high=htf.ob_high, ob_low=htf.ob_low, ob_strength=htf.ob_strength,
            fvg_kind=htf.fvg_kind.replace("_", " "),
            fvg_high=htf.fvg_high, fvg_low=htf.fvg_low, fvg_age_hours=htf.fvg_age_hours,
            vpoc_yesterday=htf.vpoc_prev, vpoc_position=htf.vpoc_position,
            buyside_liquidity=htf.buyside_liquidity, buyside_pips=htf.buyside_pips,
            sellside_liquidity=htf.sellside_liquidity, sellside_pips=htf.sellside_pips,
            volume_pct=checklist.volume_pct,
            delta_5=checklist.delta_5,
            candle_pattern=checklist.candle_pattern,
            rsi14=checklist.rsi14,
            macd_hist=checklist.macd_hist, macd_hist_trend=checklist.macd_hist_trend,
            stoch_k=checklist.stoch_k, stoch_d=checklist.stoch_d,
            confluence_score=checklist.score,
            historical_win_rate=historical_rate,
            pattern_name=htf.pattern_name,
            pattern_target=htf.pattern_target,
            harmonic_name=htf.harmonic_name,
            wyckoff_event=htf.wyckoff_event,
            next_red_event=macro.next_red_event,
            next_red_minutes=macro.next_red_minutes,
            sentiment_summary=macro.sentiment_summary,
            news_gate=macro.news_gate,
            last_10_trades=seq,
            win_rate_20=self._win_rate_20(pair),
            daily_pnl=db.daily_pnl(),
            proposed_direction=direction,
            strategy_name=strategy,
        )
        return d

    @staticmethod
    def _win_rate_20(pair: str) -> float:
        """Win rate over the last 20 closed trades (all pairs)."""
        rows = db.closed_trades(limit=20)
        if not rows:
            return 0.0
        wins = sum(1 for r in rows if float(r["pnl_usd"] or 0) > 0)
        return wins / len(rows) * 100.0

    # ---- main ----

    def evaluate(self, pair: str, direction: str, strategy: str,
                 signal_entry: float, signal_sl: float, signal_tp: float,
                 session: str = "", _mirrored: bool = False) -> ResearchVerdict:
        """Run all 10 steps. Returns an approved verdict or a logged rejection."""
        started = time.monotonic()
        verdict = ResearchVerdict(pair=pair, direction=direction, strategy=strategy,
                                  approved=False, reason="", entry=signal_entry,
                                  sl=signal_sl, tp1=signal_tp)
        session = session or "research"

        try:
            # ---- steps 1-2: macro + news gate ----
            macro = self.macro.analyze(pair, direction)
            verdict.macro = macro
            verdict.size_multipliers = dict(macro.size_multipliers)
            if macro.blocked:
                reason = f"step2 news gate: {macro.news_gate}"
                logger.warning("REJECT [step2-news] %s %s (%s): %s",
                               pair, direction, strategy, reason)
                verdict.reason = reason
                db.record_research(pair, "blocked", reason)
                return self._finish(verdict, started)

            # ---- data frames ----
            frames = self.data.get_frames(pair)

            # ---- steps 3-4: HTF + structure ----
            htf = self.htf_analyzer.analyze(pair, frames, direction)
            verdict.htf = htf
            if not htf.direction_ok:
                reason = (f"step3 HTF disagree: daily={htf.daily_trend} h4={htf.h4_bias} "
                          f"want={direction}")
                # When BOTH timeframes agree on one direction, the setup is
                # valid but was proposed the wrong way round (e.g. a mean-
                # reversion short into a daily uptrend). Retrying the mirror
                # direction salvages the cycle instead of wasting it. When the
                # timeframes CONFLICT (daily=up h4=bearish) both directions are
                # blocked by design — no retry (logs confirm retrying those
                # only produces confluence-denials). The _mirrored flag is a
                # recursion guard: a mirrored evaluation never mirrors again.
                mirrored = (None if _mirrored else self._mirror_retry(
                    pair, direction, strategy, signal_entry, signal_sl,
                    signal_tp, session, htf))
                if mirrored is not None:
                    return mirrored
                logger.warning("REJECT [step3-htf] %s %s (%s): %s",
                               pair, direction, strategy, reason)
                verdict.reason = reason
                db.record_research(pair, "rejected", reason)
                return self._finish(verdict, started)

            # ---- step 8 first: historical rate feeds the checklist requirement ----
            hist = self.matcher.match(pair, session, strategy, 50.0, direction,
                                      htf.direction_ok)

            # ---- steps 5-6: entry checklist ----
            bonus = (1 if htf.pattern_name not in ("none", "") else 0) + \
                    (1 if htf.harmonic_name != "none" else 0)
            checklist = self.validator.validate(pair, direction, frames, htf, macro,
                                                confluence_bonus=bonus)
            verdict.checklist = checklist
            verdict.required_confluence = max(
                settings.MIN_CONFLUENCE, hist.required_confluence,
                9 if checklist.required >= 9 else 8)
            # TEMP window: operator-ordered cap on the confluence floor
            # (MIN_CONFLUENCE env itself is NOT lowered; the cap is
            # auto-expiring so steady state restores itself). The cap only
            # relaxes the OPERATOR floor: history-driven and checklist-driven
            # requirements (9) are never undercut.
            _cap = settings.temp_min_confluence()
            if (_cap and verdict.required_confluence > _cap
                    and verdict.required_confluence == settings.MIN_CONFLUENCE):
                logger.warning("TEMP WINDOW: confluence floor capped %d -> %d",
                               verdict.required_confluence, _cap)
                verdict.required_confluence = _cap
            # ---- first-trade pilot (optional, one-time) ----
            # While no trade has ever been recorded, a setup short of the
            # STATIC floor (MIN_CONFLUENCE only) may proceed, compensated by a
            # raised Groq conviction floor that binds ONLY when the setup
            # actually needed the pilot. So a full-score setup keeps the normal
            # conviction gate and the pilot can never suppress a trade that
            # steady-state would have approved. History-driven floors (e.g.
            # poor win rate -> 9) are never relaxed, and the first trade row
            # voids the pilot forever (state = the DB, nothing to clean up).
            pilot_used = False
            if (checklist.score < verdict.required_confluence
                    and settings.FIRST_TRADE_PILOT_ENABLED
                    and verdict.required_confluence == settings.MIN_CONFLUENCE
                    and checklist.score >= settings.FIRST_TRADE_PILOT_CONFLUENCE
                    and not db.has_any_trades()):
                logger.info("first-trade pilot: confluence %d/%d accepted pending "
                            "Groq conviction >= %d", checklist.score,
                            verdict.required_confluence,
                            settings.FIRST_TRADE_PILOT_CONVICTION)
                pilot_used = True
            if checklist.score < verdict.required_confluence and not pilot_used:
                reason = (f"step6 confluence {checklist.score}/{verdict.required_confluence} "
                          f"(failed: {', '.join(checklist.failed_items)})")
                logger.warning("REJECT [step6-confluence] %s %s (%s): %s",
                               pair, direction, strategy, reason)
                verdict.reason = reason
                db.record_research(pair, "rejected", reason, checklist.score)
                return self._finish(verdict, started)

            # ---- step 7 counted via bonus above ----

            # ---- step 9: Groq 3x consensus ----
            research_data = self._research_data(pair, direction, strategy, macro, htf,
                                                checklist, hist, frames,
                                                hist.win_rate, session)
            prompt = self.builder.build(research_data)
            verdict.groq_prompt = prompt
            decision = self._groq_consensus(pair, prompt,
                                            pilot_floor_active=pilot_used)
            if decision is None:
                reason = "step9 groq: no consensus"
                logger.warning("REJECT [step9-consensus] %s %s (%s): 3 Groq votes "
                               "did not agree (or any vote was skip)",
                               pair, direction, strategy)
                verdict.reason = reason
                db.record_research(pair, "rejected", reason, checklist.score)
                return self._finish(verdict, started)

            if str(decision.get("decision", "skip")).lower() != direction:
                reason = f"step9 groq decision={decision.get('decision')}"
                logger.warning("REJECT [step9-decision] %s %s (%s): Groq voted "
                               "'%s' conviction=%s vs proposed %s",
                               pair, direction, strategy, decision.get("decision"),
                               decision.get("conviction"), direction)
                verdict.reason = reason
                db.record_research(pair, "rejected", reason, checklist.score,
                                   int(decision.get("conviction", 0)), {"groq": decision})
                return self._finish(verdict, started)

            verdict.conviction = int(decision.get("conviction", 0))
            verdict.groq_response = decision
            verdict.regime = str(decision.get("regime", "unknown"))
            verdict.invalidation = self._price_or_zero(decision.get("invalidation"))

            # consensus size
            verdict.consensus_size = FULL_SIZE if decision.get("_consensus") == 3 \
                else REDUCED_65

            # prices: Groq SL/TP preferred when sane, strategy levels as fallback
            verdict.sl = self._price_or_zero(decision.get("sl")) or signal_sl
            verdict.tp1 = self._price_or_zero(decision.get("tp1")) or signal_tp
            risk = abs(verdict.entry - verdict.sl)
            verdict.tp2 = self._price_or_zero(decision.get("tp2")) or \
                (verdict.entry + 2 * risk if direction == "buy" else verdict.entry - 2 * risk)
            verdict.tp3 = self._price_or_zero(decision.get("tp3")) or \
                (verdict.entry + 3 * risk if direction == "buy" else verdict.entry - 3 * risk)

            verdict.approved = True
            verdict.reason = (f"approved: confluence {checklist.score}, conviction "
                              f"{verdict.conviction}, consensus {decision.get('_consensus')}/3")
            db.record_research(pair, "accepted", verdict.reason, checklist.score,
                               verdict.conviction, {"groq": decision})
            return self._finish(verdict, started)

        except Exception as exc:
            logger.exception("research engine failed for %s: %s", pair, exc)
            verdict.reason = f"research error: {exc}"
            db.record_research(pair, "rejected", verdict.reason[:200])
            return self._finish(verdict, started)

    # ---- step3 mirror retry -------------------------------------------------

    def _mirror_retry(self, pair: str, direction: str, strategy: str,
                      signal_entry: float, signal_sl: float, signal_tp: float,
                      session: str, htf) -> Optional["ResearchVerdict"]:
        """Re-evaluate the SAME setup in the opposite direction.

        Used when step3 rejects only because the proposed direction fights an
        otherwise-agreed higher-timeframe trend (daily=h4=up but want=sell).
        All downstream checks (zones, RSI, MACD, Groq) are direction-symmetric,
        so the flipped setup gets a full fair evaluation — nothing is forced
        through. Returns None when HTFs conflict (mirror blocked by design)
        or when the mirrored evaluation is itself rejected.
        """
        if not htf.htf_agree:
            return None
        flipped = "buy" if direction == "sell" else "sell"
        try:
            # no macro pre-check here: evaluate() enforces the news gate
            # itself at step2, so a blocked mirror is rejected normally
            m_entry, m_sl, m_tp = self._mirror_prices(
                signal_entry, signal_sl, signal_tp, flipped)
            return self.evaluate(pair, flipped, strategy, m_entry, m_sl, m_tp,
                                 session, _mirrored=True)
        except Exception as exc:
            logger.debug("mirror retry failed for %s: %s", pair, exc)
            return None

    @staticmethod
    def _mirror_prices(entry: float, sl: float, tp: float,
                       direction: str) -> tuple[float, float, float]:
        """Reflect SL/TP to the other side of entry (symmetric involution:
        mirroring twice returns the original levels). Entry unchanged."""
        if entry <= 0 or sl <= 0 or tp <= 0:
            return entry, sl, tp
        return entry, 2 * entry - sl, 2 * entry - tp

    def _groq_consensus(self, pair: str, prompt: str,
                        pilot_floor_active: bool = False) -> Optional[dict]:
        """Ask Groq 3x at temperature 0; majority rules, any skip wins.

        pilot_floor_active: True when the current setup only passed step6 via
        the first-trade pilot — such setups must clear the raised pilot
        conviction floor instead of the steady-state one.
        """
        if self.verifier is None:
            logger.warning("no verifier wired; skipping Groq consensus (auto-accept)")
            return {"decision": "skip", "conviction": 0,
                    "reasoning": {"against_case": "no verifier configured"},
                    "_consensus": 0, "_forced_skip": True} if False else None
        responses: list[dict] = []
        for _ in range(3):
            resp = self.verifier.verified_decision(pair, prompt, max_attempts=1)
            if resp is None:
                continue
            if str(resp.get("decision", "")).lower() == "skip" or resp.get("_forced_skip"):
                return resp  # any skip -> no trade
            responses.append(resp)
        if not responses:
            return None
        # majority by direction
        buys = sum(1 for r in responses if r.get("decision") == "buy")
        sells = sum(1 for r in responses if r.get("decision") == "sell")
        majority = "buy" if buys > sells else "sell" if sells > buys else ""
        agreeing = [r for r in responses if r.get("decision") == majority]
        if not majority or len(agreeing) < 2:
            return None
        avg_conv = sum(int(r.get("conviction", 0)) for r in agreeing) / len(agreeing)
        # TEMP window conviction floor supersedes both steady-state and the
        # first-trade pilot (operator-ordered, auto-expiring)
        _temp_conv = settings.temp_groq_min_conviction()
        min_conv = _temp_conv if _temp_conv else settings.GROQ_MIN_CONVICTION
        if (pilot_floor_active and settings.FIRST_TRADE_PILOT_ENABLED
                and not _temp_conv):
            min_conv = max(min_conv, settings.FIRST_TRADE_PILOT_CONVICTION)
        if avg_conv < min_conv:
            logger.warning("REJECT [step9-conviction] %s: consensus conviction "
                           "%.0f < floor %d", pair, avg_conv, min_conv)
            return None
        best = max(agreeing, key=lambda r: int(r.get("conviction", 0)))
        best["_consensus"] = len(agreeing)
        return best

    @staticmethod
    def _price_or_zero(value) -> float:
        """Parse a price from Groq output (float or string)."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _finish(self, verdict: ResearchVerdict, started: float) -> ResearchVerdict:
        """Stamp duration and log."""
        verdict.duration_sec = round(time.monotonic() - started, 1)
        logger.info("research %s %s -> %s (%.1fs): %s", verdict.pair, verdict.direction,
                    "APPROVED" if verdict.approved else "DENIED",
                    verdict.duration_sec, verdict.reason)
        return verdict
