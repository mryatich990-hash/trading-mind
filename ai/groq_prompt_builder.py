"""GroqPromptBuilder: assembles the complete research data package.

Contains ONLY verified numbers from the research engine — the exact template
from the master prompt, with every claim traceable to a line of data.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["ResearchData", "GroqPromptBuilder"]


@dataclass
class ResearchData:
    """Everything step 1-8 of the research engine produced."""

    pair: str
    timestamp: str
    bid: float
    ask: float
    spread_pips: float
    session: str
    minutes_in_session: int
    # macro
    dxy_trend: str = "neutral"
    vix: float = 0.0
    vix_term: str = "normal"
    usd_bias: float = 0.0
    base_bias: float = 0.0
    base_currency: str = ""
    quote_bias: float = 0.0
    quote_currency: str = ""
    cot_commercial_net: float = 0.0
    cot_percentile: float = 50.0
    retail_long_pct: float = 50.0
    retail_contrarian: str = "neutral"
    yield_10y: float = 0.0
    yield_trend: str = "flat"
    intermarket_signal: str = ""
    # htf
    daily_trend: str = "range"
    daily_vwap_position: str = "at"
    h4_ema20: float = 0.0
    h4_ema50: float = 0.0
    h4_ema200: float = 0.0
    h4_bos: str = "none"
    h4_bos_price: float = 0.0
    h4_bos_ago: int = 0
    wyckoff_phase: str = "ranging"
    elliott_wave: str = "no count"
    # structure
    ob_kind: str = "none"
    ob_high: float = 0.0
    ob_low: float = 0.0
    ob_strength: float = 0.0
    fvg_kind: str = "none"
    fvg_high: float = 0.0
    fvg_low: float = 0.0
    fvg_age_hours: float = 0.0
    vpoc_yesterday: float = 0.0
    vpoc_position: str = "at"
    buyside_liquidity: float = 0.0
    buyside_pips: float = 0.0
    sellside_liquidity: float = 0.0
    sellside_pips: float = 0.0
    volume_pct: float = 100.0
    delta_5: float = 0.0
    # m15 entry
    candle_pattern: str = "none"
    rsi14: float = 50.0
    macd_hist: float = 0.0
    macd_hist_trend: str = "flat"
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    confluence_score: int = 0
    historical_win_rate: float = 0.0
    # patterns
    pattern_name: str = "none"
    pattern_target: float = 0.0
    harmonic_name: str = "none"
    wyckoff_event: str = ""
    # news
    next_red_event: str = "none"
    next_red_minutes: int = 0
    sentiment_summary: str = "neutral"
    news_gate: str = "clear"
    # performance
    last_10_trades: str = "no trades"
    win_rate_20: float = 0.0
    daily_pnl: float = 0.0
    # strategy proposal
    proposed_direction: str = "none"
    strategy_name: str = ""


class GroqPromptBuilder:
    """Renders the research prompt exactly per the master template."""

    def build(self, d: ResearchData) -> str:
        """Render the structured prompt. Deterministic, numbers only."""
        lines = [
            f"TIMESTAMP: {d.timestamp}",
            f"PAIR: {d.pair} | PRICE: {d.bid:.5f}/{d.ask:.5f} | SPREAD: {d.spread_pips} pips",
            f"SESSION: {d.session} | MINUTES IN: {d.minutes_in_session}",
            "",
            "MACRO INTELLIGENCE:",
            f"DXY trend: {d.dxy_trend} | VIX: {d.vix:.1f} | Term structure: {d.vix_term}",
            f"USD bias: {d.usd_bias:+.0f}/100 | {d.base_currency} bias: {d.base_bias:+.0f}/100",
            f"COT commercials net: {d.cot_commercial_net:.0f} | Percentile: {d.cot_percentile:.0f}%",
            f"Retail sentiment: {d.retail_long_pct:.0f}% long (contrarian: {d.retail_contrarian})",
            f"Bond yield 10yr: {d.yield_10y:.2f}% | Trend: {d.yield_trend}",
            f"Intermarket signal: {d.intermarket_signal}",
            "",
            "HIGHER TIMEFRAME:",
            f"Daily trend: {d.daily_trend} | Daily VWAP: {d.daily_vwap_position}",
            f"H4 EMA20: {d.h4_ema20:.5f} H4 EMA50: {d.h4_ema50:.5f} H4 EMA200: {d.h4_ema200:.5f}",
            f"H4 last BOS: {d.h4_bos} at {d.h4_bos_price:.5f} ({d.h4_bos_ago} candles ago)",
            f"Wyckoff phase: {d.wyckoff_phase} {d.wyckoff_event}".strip(),
            f"Elliott Wave: {d.elliott_wave}",
            "",
            "INSTITUTIONAL STRUCTURE:",
            f"Nearest OB: {d.ob_kind} {d.ob_high:.5f}-{d.ob_low:.5f} | Strength: {d.ob_strength:.1f}",
            f"Nearest FVG: {d.fvg_kind} {d.fvg_high:.5f}-{d.fvg_low:.5f} | Age: {d.fvg_age_hours}h",
            f"VPOC yesterday: {d.vpoc_yesterday:.5f} | Price vs VPOC: {d.vpoc_position}",
            f"Nearest buy-side liquidity: {d.buyside_liquidity:.5f} ({d.buyside_pips} pips)",
            f"Nearest sell-side liquidity: {d.sellside_liquidity:.5f} ({d.sellside_pips} pips)",
            f"Volume at price: {d.volume_pct:.0f}% of average",
            f"Delta volume last 5 candles: {d.delta_5:.0f}",
            "",
            "M15 ENTRY:",
            f"Candle pattern: {d.candle_pattern}",
            f"RSI14: {d.rsi14:.1f}",
            f"MACD histogram: {d.macd_hist:.5f} ({d.macd_hist_trend})",
            f"Stochastic K: {d.stoch_k:.0f} D: {d.stoch_d:.0f}",
            f"Confluence score: {d.confluence_score}/10",
            f"Historical win rate this setup: {d.historical_win_rate:.0f}%",
            "",
            "HARMONIC/CLASSICAL:",
            f"Pattern detected: {d.pattern_name} target {d.pattern_target:.5f}"
            if d.pattern_name != "none" else "Pattern detected: none",
            f"Harmonic detected: {d.harmonic_name}" if d.harmonic_name != "none"
            else "Harmonic detected: none",
            "",
            "NEWS:",
            f"Next red event: {d.next_red_event} in {d.next_red_minutes} min",
            f"Sentiment last 2h: {d.sentiment_summary}",
            f"News gate: {d.news_gate}",
            "",
            f"LAST 10 TRADES THIS PAIR: {d.last_10_trades}",
            f"OVERALL WIN RATE LAST 20: {d.win_rate_20:.0f}%",
            f"DAILY PnL: {d.daily_pnl:.2f}",
            f"PROPOSED DIRECTION: {d.proposed_direction} (strategy: {d.strategy_name})",
            "",
            "Return ONLY valid JSON:",
            '{"decision": "buy|sell|skip", "conviction": 0-100, "sl": price, "tp1": price, '
            '"tp2": price, "tp3": price, "reasoning": {"macro_case": "cite number", '
            '"structure_case": "cite number", "entry_case": "cite number", '
            '"institutional_case": "cite number", "against_case": "strongest reason NOT to '
            'take this"}, "invalidation": "exact price", "regime": "trending|ranging|volatile"}',
        ]
        prompt = "\n".join(lines)
        logger.info("research prompt built for %s (%d chars)", d.pair, len(prompt))
        return prompt
