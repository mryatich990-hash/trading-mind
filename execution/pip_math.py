"""Shared FX math: pip sizes and USD pip values, single source of truth.

PaperBroker (realized P&L), RiskManager (sizing) and TradeManager (recorded
P&L / Telegram alerts) must agree, otherwise recorded results drift from the
broker balance (EURJPY #4: DB said +$575, broker credited +$352).
"""

from __future__ import annotations

__all__ = ["pip_size", "pip_value_usd"]

_PIP = {"JPY": 0.01, "XAUUSD": 0.1, "NAS100": 1.0, "US30": 1.0}
_PIP_VALUE = {"JPY": 6.8}  # USD per pip per 1.0 lot; others 10.0


def pip_size(pair: str) -> float:
    """Price increment of one pip for a pair."""
    pair = pair.upper()
    for key, value in _PIP.items():
        if key in pair:
            return value
    return 0.0001


def pip_value_usd(pair: str) -> float:
    """USD per pip per 1.0 standard lot (JPY crosses ~6.8, else 10.0)."""
    pair = pair.upper()
    for key, value in _PIP_VALUE.items():
        if key in pair:
            return value
    return 10.0
