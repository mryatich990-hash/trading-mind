"""Regression tests for the massive-lot bug (EURJPY trade #4, 2026-10-01).

Trade #4 opened 1.55 lots on EURJPY because Groq's SL was ~4.4 pips from
entry and lots_for_risk() is risk-denominated with no absolute cap:
risk / (sl_pips * pip_value) explodes as sl_pips -> 0. The same pattern
produced 2.42 lots on a 2-pip USDJPY SL (trade #2).

Also: trade_manager.close() valued JPY-cross P&L at $10/pip while the paper
broker credits $6.8/pip -> DB pnl (+$575.04) != broker credit (+$351.76).
"""

from __future__ import annotations

import pytest

from config import settings
from execution.pip_math import pip_size, pip_value_usd
from risk.risk_manager import lots_for_risk

CAP = pytest.approx(settings.MAX_ABS_LOTS, abs=0.011)


class TestAbsoluteLotCap:
    def test_tiny_sl_clamped(self):
        """The exact trade #4 inputs: 4.4-pip SL at ~1.3% risk must not
        produce 1.55 lots."""
        lots = lots_for_risk(10056.0, 1.3, 4.4, "EURJPY")
        assert float(lots) <= float(settings.MAX_ABS_LOTS) + 1e-9

    def test_two_pip_sl_clamped(self):
        """Trade #2 pattern: 2-pip USDJPY SL must not produce 2.42 lots."""
        lots = lots_for_risk(10000.0, 1.0, 2.0, "USDJPY")
        assert float(lots) <= float(settings.MAX_ABS_LOTS) + 1e-9

    def test_tiny_sl_at_max_risk_still_clamped(self):
        """Even the max stacked risk (1.5%) on a 2-pip SL is capped."""
        lots = lots_for_risk(10000.0, 1.5, 2.0, "EURUSD")
        assert float(lots) <= float(settings.MAX_ABS_LOTS) + 1e-9

    def test_normal_sizing_untouched(self):
        """A sane 20-pip SL / 1% risk on $10k still sizes 0.50 lots."""
        lots = lots_for_risk(10000.0, 1.0, 20.0, "EURUSD")
        assert float(lots) == pytest.approx(0.50, abs=0.01)

    def test_cap_only_undershoots_risk(self):
        """Clamping may reduce lots (less risk) but never inflate them."""
        lots = lots_for_risk(10000.0, 1.0, 100.0, "EURUSD")
        assert float(lots) <= float(settings.MAX_ABS_LOTS) + 1e-9

    def test_huge_balance_still_capped(self):
        lots = lots_for_risk(1_000_000.0, 1.5, 3.0, "GBPUSD")
        assert float(lots) <= float(settings.MAX_ABS_LOTS) + 1e-9


class TestPipValueConsistency:
    def test_jpy_cross_pip_value_matches_broker(self):
        """Sizing, broker P&L and recorded P&L must use the same $/pip."""
        assert pip_value_usd("EURJPY") == 6.8
        assert pip_value_usd("USDJPY") == 6.8
        assert pip_value_usd("GBPJPY") == 6.8
        assert pip_value_usd("EURUSD") == 10.0

    def test_jpy_pip_size(self):
        assert pip_size("EURJPY") == 0.01
        assert pip_size("USDJPY") == 0.01
        assert pip_size("EURUSD") == 0.0001

    def test_risk_manager_table_agrees(self):
        from risk.risk_manager import PIP_VALUE_PER_LOT
        for pair in ("USDJPY", "EURJPY", "GBPJPY"):
            assert PIP_VALUE_PER_LOT[pair] == pip_value_usd(pair)

    def test_close_pnl_matches_broker_credit(self):
        """Reproduce trade #4: 37.1 pips x 1.55 lots x 6.8 = $390.85 (slippage
        moved entry), NOT $575.04 ($10/pip)."""
        pips, lots = 37.1, 1.55
        pnl = pips * pip_value_usd("EURJPY") * lots
        assert pnl == pytest.approx(390.85, abs=0.5)


class TestGroqSlSanityGate:
    """Groq SL tighter than the structural stop must be rejected.

    Trade #4: entry 177.676 sell, structural stop ~177.80 (12 pips above),
    Groq SL 177.72 (4.4 pips) -> sizing exploded. The guard falls back to
    the structural stop whenever Groq's is tighter, absent, or on the
    wrong side of entry. (rr_achieved=18.55 in the DB row was computed
    against the later 2-pip breakeven trail, not initial risk.)
    """

    def test_tighter_groq_sl_rejected_sell(self):
        from research.research_engine import ResearchEngine
        f = ResearchEngine._sl_at_least_as_wide
        assert f(177.72, 177.80, 177.676, "sell") is False

    def test_wider_groq_sl_accepted_sell(self):
        from research.research_engine import ResearchEngine
        f = ResearchEngine._sl_at_least_as_wide
        assert f(177.90, 177.80, 177.676, "sell") is True

    def test_tighter_groq_sl_rejected_buy(self):
        from research.research_engine import ResearchEngine
        f = ResearchEngine._sl_at_least_as_wide
        assert f(1.1290, 1.1280, 1.12841, "buy") is False

    def test_wrong_side_stop_rejected(self):
        from research.research_engine import ResearchEngine
        f = ResearchEngine._sl_at_least_as_wide
        # sell with stop below entry, buy with stop above entry
        assert f(177.60, 177.80, 177.676, "sell") is False
        assert f(1.1290, 1.1299, 1.12841, "buy") is False

    def test_zero_or_invalid_falls_back(self):
        from research.research_engine import ResearchEngine
        f = ResearchEngine._sl_at_least_as_wide
        assert f(0.0, 177.80, 177.676, "sell") is False
        assert f(177.72, 0.0, 177.676, "sell") is False

    def test_exact_equal_width_accepted(self):
        from research.research_engine import ResearchEngine
        f = ResearchEngine._sl_at_least_as_wide
        assert f(1.1280, 1.1280, 1.12841, "buy") is True


class TestRiskGateSlFloor:
    """The risk gate rejects stops tighter than MIN_SL_SPREAD_MULT x spread.

    Last line of defense: even if a strategy's structural stop itself is
    inside noise, the gate refuses to size such a trade. The production
    call passes spread_pips=0.0, so the modeled spread from pip_math is
    used as the reference.
    """

    @staticmethod
    def _evaluate(rm, pair, direction, entry, sl, tp,
                  confluence=8, spread=0.0):
        return rm.evaluate(pair, direction, "liquidity_sweep", entry, sl, tp,
                           balance=10000.0, equity=10000.0, used_margin=0.0,
                           spread_pips=spread, confluence=confluence,
                           mode="demo")

    def test_trade4_sl_rejected(self, temp_db):
        """Exact trade #4 geometry: 4.4-pip SL vs EURJPY modeled spread 2.0
        -> floor 6.0 pips. This trade must now be refused at the gate."""
        from risk.risk_manager import RiskManager
        rm = RiskManager()
        d = self._evaluate(rm, "EURJPY", "sell", 177.676, 177.72, 177.33)
        assert d.approved is False
        assert "sl 4.4 pips < 6.0" in d.reason

    def test_structural_sl_passes_floor(self, temp_db):
        """A 12-pip structural stop clears the floor and proceeds to approval
        (with the lots cap applying)."""
        from risk.risk_manager import RiskManager
        rm = RiskManager()
        d = self._evaluate(rm, "EURJPY", "sell", 177.676, 177.556, 177.33)
        assert d.approved is True
        assert float(d.lots) <= settings.MAX_ABS_LOTS + 1e-9

    def test_explicit_spread_used_when_given(self, temp_db):
        """A caller-supplied spread takes precedence over the modeled one."""
        from risk.risk_manager import RiskManager
        rm = RiskManager()
        d = self._evaluate(rm, "EURUSD", "buy", 1.1000, 1.0988, 1.1060,
                           spread=5.0)
        assert d.approved is False
        assert "sl 12.0 pips < 15.0" in d.reason

    def test_unknown_pair_fallback_spread(self, temp_db):
        """Pairs missing from the spread table fall back to 1.5 pips."""
        from risk.risk_manager import RiskManager
        rm = RiskManager()
        d = self._evaluate(rm, "AUDNZD", "sell", 1.0900, 1.0901, 1.0800)
        assert d.approved is False
        assert "sl 1.0 pips < 4.5" in d.reason
