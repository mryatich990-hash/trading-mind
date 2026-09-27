"""Deep learning price prediction (UPGRADE 1, model 2): XGBoost win probability.

47 features describe the current market state; the target is whether the
proposed trade won. XGBoost when installed, sklearn's HistGradientBoosting
as functional fallback otherwise. Trains on all closed trades with outcomes;
retrains every DL_RETRAIN_TRADES closed trades. Model file: models/xgb_<pair>.json.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

try:  # guarded heavy dependency
    import xgboost as xgb  # type: ignore
    XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    xgb = None
    XGB_AVAILABLE = False

# the 47 features: name -> extractor key order (stable contract for dashboard/tests)
FEATURE_NAMES = [
    # price action (12)
    "ret_1", "ret_5", "ret_15", "body_ratio", "wick_ratio", "range_atr",
    "close_pos", "gap", "dist_day_high", "dist_day_low", "dist_week_high", "dist_week_low",
    # indicators (10)
    "rsi14", "macd_hist", "macd_signal_dist", "stoch_k", "stoch_d",
    "ema20_dist", "ema50_dist", "ema200_dist", "atr14", "atr_ratio",
    # volatility / volume (6)
    "vol_z", "vol_ratio", "spread_pips", "bb_pos", "keltner_pos", "adx",
    # time (2)
    "hour_sin", "hour_cos",
    # session (5)
    "is_asia", "is_london", "is_ny", "is_overlap", "minutes_in_session",
    # context (12)
    "day_high_low_range", "vix", "dxy_trend_num", "usd_bias", "eur_bias",
    "gbp_bias", "jpy_bias", "xau_bias", "cot_pctile", "retail_long",
    "groq_conviction_avg", "confluence_score",
]
assert len(FEATURE_NAMES) == 47, f"expected 47 features, got {len(FEATURE_NAMES)}"


def build_feature_vector(ctx_dict: dict) -> np.ndarray:
    """Extract the 47-feature vector from a market-state dict (missing -> 0)."""
    get = lambda k, d=0.0: float(ctx_dict.get(k, d) or 0.0)
    values = [get(name) for name in FEATURE_NAMES]
    return np.asarray(values, dtype=np.float32)


class XGBoostModel:
    """Win-probability model over the 47 market-state features."""

    def __init__(self, pair: str = "ALL", models_dir: str = "") -> None:
        self.pair = pair.upper()
        self.models_dir = models_dir or settings.DL_MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        self.path = os.path.join(self.models_dir, f"xgb_{self.pair.lower()}.json")
        self.backend = "xgboost" if XGB_AVAILABLE else "sklearn"
        self.model = None
        self.trained_trades = 0
        self._load()

    # ---- persistence ----

    def save(self) -> None:
        """Persist model + metadata."""
        try:
            if self.model is None:
                return
            if self.backend == "xgboost":
                self.model.save_model(self.path)
            else:
                import pickle

                with open(self.path, "wb") as fh:
                    pickle.dump(self.model, fh)
            with open(self.path + ".meta.json", "w") as fh:
                json.dump({"trained_trades": self.trained_trades,
                           "backend": self.backend}, fh)
        except Exception as exc:
            logger.warning("xgb save failed: %s", exc)

    def _load(self) -> None:
        """Load persisted model when present."""
        try:
            if not os.path.exists(self.path):
                return
            if self.backend == "xgboost":
                booster = xgb.XGBClassifier()
                booster.load_model(self.path)
                self.model = booster
            else:
                import pickle

                with open(self.path, "rb") as fh:
                    self.model = pickle.load(fh)
            meta = self.path + ".meta.json"
            if os.path.exists(meta):
                with open(meta) as fh:
                    self.trained_trades = json.load(fh).get("trained_trades", 0)
        except Exception as exc:
            logger.warning("xgb load failed: %s", exc)

    # ---- training ----

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fit on feature rows (47) and 0/1 outcomes; returns metrics."""
        if len(X) < 30 or len(set(y.tolist())) < 2:
            return {"ok": False, "reason": f"insufficient data ({len(X)} rows)"}
        split = max(1, int(len(X) * 0.85))
        if self.backend == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=250, max_depth=5, learning_rate=0.06,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss")
            self.model.fit(X[:split], y[:split])
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(max_depth=4, max_iter=200)
            self.model.fit(X[:split], y[:split])
        acc = float(self.model.score(X[split:], y[split:])) * 100.0
        self.trained_trades += int(len(X))
        self.save()
        return {"ok": True, "val_accuracy": round(acc, 1), "rows": int(len(X)),
                "backend": self.backend}

    def needs_retrain(self, closed_trades: int) -> bool:
        """True when DL_RETRAIN_TRADES new closed trades arrived."""
        return closed_trades - self.trained_trades >= settings.DL_RETRAIN_TRADES

    # ---- inference ----

    def predict_proba(self, ctx_dict: dict) -> Optional[float]:
        """Win probability 0-100 (None when untrained)."""
        if self.model is None:
            return None
        x = build_feature_vector(ctx_dict).reshape(1, -1)
        try:
            proba = self.model.predict_proba(x)[0]
            return round(float(proba[list(self.model.classes_).index(1)]) * 100.0, 1)
        except Exception as exc:
            logger.warning("xgb predict failed: %s", exc)
            return None

    def feature_importances(self, top: int = 10) -> list[dict]:
        """Top features for the dashboard."""
        if self.model is None:
            return []
        try:
            if self.backend == "xgboost":
                scores = self.model.feature_importances_
            else:
                scores = np.zeros(len(FEATURE_NAMES))  # sklearn fallback exposes none
            order = np.argsort(scores)[::-1][:top]
            return [{"feature": FEATURE_NAMES[i], "score": round(float(scores[i]), 4)}
                    for i in order]
        except Exception:
            return []
