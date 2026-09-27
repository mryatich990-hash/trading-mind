"""Strategy library: all 15 strategies in one place.

The ten session/SMC strategies live in this package (migrated from the
prompt-3 engine); the five institutional strategies (vwap_reversion,
vpoc_magnet, wyckoff_springthrust, harmonic_prz, cot_extreme) are also part
of the unified package. The selector imports from here.
"""

from strategies.asian_range_fade import AsianRangeFadeStrategy
from strategies.cot_extreme import COTExtremeStrategy
from strategies.ema_trend_rider import EMATrendRiderStrategy
from strategies.fvg_fill import FVGFillStrategy
from strategies.harmonic_prz import HarmonicPRZStrategy
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.london_breakout import LondonBreakoutStrategy
from strategies.macd_momentum import MACDMomentumStrategy
from strategies.news_spike_fade import NewsSpikeFadeStrategy
from strategies.ny_reversal import NYReversalStrategy
from strategies.order_block_sniper import OrderBlockSniperStrategy
from strategies.rsi_divergence import RSIDivergenceStrategy
from strategies.vpoc_magnet import VPOCMagnetStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.wyckoff_springthrust import WyckoffSpringThrustStrategy

__all__ = [
    "AsianRangeFadeStrategy", "COTExtremeStrategy", "EMATrendRiderStrategy",
    "FVGFillStrategy", "HarmonicPRZStrategy", "LiquiditySweepStrategy",
    "LondonBreakoutStrategy", "MACDMomentumStrategy", "NewsSpikeFadeStrategy",
    "NYReversalStrategy", "OrderBlockSniperStrategy", "RSIDivergenceStrategy",
    "VPOCMagnetStrategy", "VWAPReversionStrategy", "WyckoffSpringThrustStrategy",
]


def all_strategies() -> list:
    """Instantiate all 15 strategies in selector priority order."""
    return [
        LondonBreakoutStrategy(), NYReversalStrategy(), OrderBlockSniperStrategy(),
        FVGFillStrategy(), EMATrendRiderStrategy(), LiquiditySweepStrategy(),
        RSIDivergenceStrategy(), MACDMomentumStrategy(), NewsSpikeFadeStrategy(),
        AsianRangeFadeStrategy(), VWAPReversionStrategy(), VPOCMagnetStrategy(),
        WyckoffSpringThrustStrategy(), HarmonicPRZStrategy(), COTExtremeStrategy(),
    ]
