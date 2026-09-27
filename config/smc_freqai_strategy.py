"""Freqtrade FreqAI strategy mirroring the live SMC features for backtesting/hyperopt.

Usage (freqtrade must be installed):
    freqtrade download-data -c config/freqtrade_config.json --timerange 20200101- --timeframes 15m 1h 4h 1d
    freqtrade backtesting -c config/freqtrade_config.json --strategy SMCFreqAIStrategy --timerange 20200101-
    freqtrade hyperopt -c config/freqtrade_config.json --strategy SMCFreqAIStrategy --hyperopt-loss SharpeHyperOptLoss --epochs 100
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.freqai.freqai_interface import FreqaiModelResolver  # noqa: F401


class SMCFreqAIStrategy(IStrategy):
    """
    FreqAI-powered SMC strategy. Walk-forward is inherent to FreqAI
    (train_period_days / backtest_period_days sliding window).
    """

    INTERFACE_VERSION = 3
    timeframe = "15m"
    informative_timeframes = ["1h", "4h", "1d"]
    can_short = True

    minimal_roi = {"0": 0.03, "60": 0.015, "240": 0.008}
    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    startup_candle_count = 200

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(p, tf) for p in pairs for tf in self.informative_timeframes]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- M15 base indicators ---
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)

        # --- Informative timeframes (H1 structure, H4/D1 bias) ---
        for tf in self.informative_timeframes:
            informative = self.dp.get_pair_dataframe(metadata["pair"], tf)
            informative["ema50"] = ta.EMA(informative, timeperiod=50)
            informative["ema200"] = ta.EMA(informative, timeperiod=200)
            informative["rsi"] = ta.RSI(informative, timeperiod=14)
            informative["atr"] = ta.ATR(informative, timeperiod=14)

            ffill_map = {
                "date": "date",
                f"ema50_{tf}": informative["ema50"],
                f"ema200_{tf}": informative["ema200"],
                f"rsi_{tf}": informative["rsi"],
                f"atr_{tf}": informative["atr"],
            }
            for col, series in ffill_map.items():
                if col == "date":
                    continue
                dataframe[col] = series.ffill()

        dataframe["h4_bias_bull"] = (
            (dataframe["ema50_4h"] > dataframe["ema200_4h"]).astype(int)
        )

        # --- FreqAI features ---
        for val in [10, 20]:
            dataframe[f"rsi-period-{val}"] = (
                dataframe["rsi"].rolling(val).mean()
            )
            dataframe[f"atr-period-{val}"] = (
                dataframe["atr"].rolling(val).mean()
            )
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["hour"] = dataframe["date"].dt.hour
        dataframe["london_kz"] = dataframe["hour"].isin([9, 10, 11]).astype(int)
        dataframe["ny_kz"] = dataframe["hour"].isin([13, 14, 15]).astype(int)
        dataframe["body"] = (dataframe["close"] - dataframe["open"]).abs()
        dataframe["range"] = dataframe["high"] - dataframe["low"]

        # Label: did price move +1R (risk=2*ATR) within next 16 candles?
        risk = 2 * dataframe["atr"]
        future_max = dataframe["high"].shift(-16).rolling(16, min_periods=1).max()
        future_min = dataframe["low"].shift(-16).rolling(16, min_periods=1).min()
        dataframe["&s-up"] = (
            (future_max - dataframe["close"]) > risk
        ).astype(int) * 2 - 1
        dataframe["&s-down"] = (
            (dataframe["close"] - future_min) > risk
        ).astype(int) * 2 - 1

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        enter_long = (
            (dataframe["h4_bias_bull"] == 1)
            & (dataframe["do_predict"] == 1)
            & (dataframe["&s-up"] > 0)
            & (dataframe["rsi"] < 65)
            & (dataframe["london_kz"] + dataframe["ny_kz"] > 0)
        )
        enter_short = (
            (dataframe["h4_bias_bull"] == 0)
            & (dataframe["do_predict"] == 1)
            & (dataframe["&s-down"] > 0)
            & (dataframe["rsi"] > 35)
            & (dataframe["london_kz"] + dataframe["ny_kz"] > 0)
        )
        dataframe.loc[enter_long, "enter_long"] = 1
        dataframe.loc[enter_short, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["rsi"] > 75, "exit_long"] = 1
        dataframe.loc[dataframe["rsi"] < 25, "exit_short"] = 1
        return dataframe
