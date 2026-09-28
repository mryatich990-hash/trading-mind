"""Funnel replay: does the live entry gate (HTF + checklist) select edge?

Replays every strategy signal from history through the SAME gates the live
bot uses (step-3 HTF agreement, step-6 checklist with the volume-exclusion
fix), then resolves each setup with the strategy's own SL/TP exactly like
backtest.py does (fill next open, walk until SL/TP, max 24h).

Outputs per-setup records to /tmp/funnel_backtest.json for analysis:
  gate-passed (score >= required) vs near-miss (required-1) vs weak.
"""
import json
import logging
import sys

sys.path.insert(0, "/home/yatich/trading-bot")
logging.disable(logging.INFO)

import pandas as pd
import yfinance as yf

from backtesting.backtest_engine import _walk_trade
from backtesting.data_loader import _normalize, load_frames
from core.db import pg_compatible  # noqa: F401  (import side-effects ok)
from research.entry_validator import EntryValidator
from research.htf_analyzer import HTFAnalyzer
from research.macro_analyzer import MacroResult
from strategies.library import all_strategies
from strategies.base_strategy import MarketContext

PAIRS_DEFAULT = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "GBPJPY", "EURJPY"]
DAYS = 55
RESULTS = "/home/yatich/trading-bot/scripts/funnel_backtest_results.json"

_YF_MAP = {"XAUUSD": "GC=F"}   # =X fallback covers EURUSD/GBP*/USDJPY/GBPJPY/EURJPY

def _real_htf(pair: str) -> dict[str, pd.DataFrame]:
    """Fetch genuine multi-month h1/h4/d1 history like the live engine uses
    (resampling 55d of m15 gives <60 daily bars -> daily trend never forms)."""
    sym = _YF_MAP.get(pair, f"{pair}=X")
    d1 = _normalize(yf.download(sym, period="1y", interval="1d", progress=False))
    h1 = _normalize(yf.download(sym, period="60d", interval="1h", progress=False))
    h4 = h1.set_index("time").resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"}).dropna().reset_index()
    return {"h1": h1, "h4": h4, "d1": d1}

def _load_previous(pairs: list[str]) -> list[dict]:
    """Existing results, minus any records for the pairs being re-run."""
    try:
        with open(RESULTS) as fh:
            old = json.load(fh)
        return [r for r in old if r.get("pair") not in pairs]
    except Exception:
        return []

def main() -> None:
    pairs = [p.strip().upper() for p in (sys.argv[1].split(",") if len(sys.argv) > 1
                                         else PAIRS_DEFAULT)]
    print(f"pairs: {pairs}", flush=True)
    strategies = all_strategies()
    htf_an = HTFAnalyzer()
    validator = EntryValidator()
    macro = MacroResult(news_gate="clear", cot_bias="neutral")  # proxy: history unavailable
    records = []

    previous = _load_previous(pairs)
    for pair in pairs:
        # incremental save: a crash/timeout must not lose completed pairs
        with open(RESULTS, "w") as fh:
            json.dump(previous + records, fh)
        print(f"=== {pair}: loading {DAYS}d ...", flush=True)
        try:
            frames = load_frames(pair, days=DAYS, use_cache=True)
        except Exception as exc:
            print(f"  load failed: {exc}", flush=True)
            continue
        m15 = frames["m15"].reset_index(drop=True)
        real = _real_htf(pair)
        h1 = real["h1"]
        h4 = real["h4"]
        d1 = real["d1"]
        h1_ts = pd.DatetimeIndex(pd.to_datetime(h1["time"], utc=True))
        h4_ts = pd.DatetimeIndex(pd.to_datetime(h4["time"], utc=True))
        d1_ts = pd.DatetimeIndex(pd.to_datetime(d1["time"], utc=True))
        print(f"  {len(m15)} m15 candles; replaying ...", flush=True)

        n_signals = n_htf = n_scored = 0
        for i in range(220, len(m15) - 1):
            ts = pd.Timestamp(m15["time"].iloc[i])
            window = m15.iloc[max(0, i - 400): i + 1].reset_index(drop=True)
            ctx = MarketContext(
                pair=pair, now=ts, m1=window.tail(60), m15=window,
                h1=h1.iloc[: h1_ts.searchsorted(ts)].reset_index(drop=True),
                h4=h4.iloc[: h4_ts.searchsorted(ts)].reset_index(drop=True),
                daily=d1.iloc[: d1_ts.searchsorted(ts)].reset_index(drop=True))
            fdict = {"m1": ctx.m1, "m15": ctx.m15, "h1": ctx.h1,
                     "h4": ctx.h4, "d1": ctx.daily}

            for strat in strategies:
                try:
                    sig = strat.evaluate(ctx)
                except Exception:
                    continue
                if sig is None:
                    continue
                n_signals += 1

                htf = htf_an.analyze(pair, fdict, sig.direction)
                if not htf.direction_ok:
                    continue  # step-3 gate: HTF disagreement (live bot stops here)
                n_htf += 1

                chk = validator.validate(pair, sig.direction, fdict, htf, macro,
                                         confluence_bonus=0)
                n_scored += 1

                if i + 1 >= len(m15):
                    continue
                fill = float(m15["open"].iloc[i + 1])
                try:
                    sl, tp = float(sig.sl), float(sig.tp)
                except (TypeError, ValueError):
                    continue
                risk_dist = abs(fill - sl)
                if risk_dist <= 0:
                    continue
                outcome, exit_price = _walk_trade(m15, i + 1, sig.direction,
                                                  fill, sl, tp)
                pnl_dist = (exit_price - fill) if sig.direction == "buy" \
                    else (fill - exit_price)
                r_mult = pnl_dist / risk_dist

                records.append({
                    "pair": pair, "ts": str(ts), "strategy": strat.name,
                    "direction": sig.direction, "score": chk.score,
                    "required": chk.required, "failed": list(chk.failed_items),
                    "outcome": outcome, "r": round(r_mult, 3),
                })
        print(f"  signals={n_signals} passed_htf={n_htf} scored={n_scored}", flush=True)

    with open(RESULTS, "w") as fh:
        json.dump(previous + records, fh)
    print(f"DONE: {len(records)} new setups recorded "
          f"({len(previous)} kept from previous runs)", flush=True)


if __name__ == "__main__":
    main()
