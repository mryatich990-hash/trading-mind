"""Deep learning price prediction (UPGRADE 1, model 1): LSTM over M15 candles.

TensorFlow/Keras is used when installed; otherwise a functional NumPy
fallback (logistic regression over the same sequence features) keeps the
engine operational. Model file: models/lstm_<pair>.keras (TF) or
models/lstm_<pair>.npz (fallback weights). Trains on the last 2000 M15
candles; retrains every DL_RETRAIN_CANDLES new candles.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

try:  # guarded heavy dependency
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
except Exception:  # pragma: no cover - TF not installed
    tf = None
    TF_AVAILABLE = False

SEQ_LEN = 60
HORIZON = 5  # predict direction over the next 5 candles
TRAIN_WINDOW = 2000


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Per-candle feature vector: OHLCV + RSI + MACD + ATR + EMA distances."""
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-12))

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    # ATR 14
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # EMA distances (in ATR units for scale invariance)
    e20 = close.ewm(span=20, adjust=False).mean()
    e50 = close.ewm(span=50, adjust=False).mean()
    d20 = (close - e20) / (atr + 1e-12)
    d50 = (close - e50) / (atr + 1e-12)

    rng = (high - low).replace(0, np.nan)
    feats = pd.DataFrame({
        "ret": close.pct_change().clip(-0.05, 0.05).fillna(0.0),
        "body": ((close - open_) / rng).fillna(0).clip(-1, 1),
        "pos": ((close - low) / rng).fillna(0.5).clip(0, 1),
        "vol_z": ((vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-12)).fillna(0).clip(-3, 3),
        "rsi": (rsi / 50 - 1).fillna(0),
        "macd": (macd_hist / (atr + 1e-12)).fillna(0).clip(-3, 3),
        "atr_ratio": (atr / atr.rolling(20).mean()).fillna(1).clip(0, 3),
        "d20": d20.fillna(0).clip(-5, 5),
        "d50": d50.fillna(0).clip(-5, 5),
    })
    return feats.to_numpy(dtype=np.float32)


def make_sequences(feats: np.ndarray, close: np.ndarray,
                   seq_len: int = SEQ_LEN, horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows -> (X sequences, y up/down over next `horizon` candles)."""
    X, y = [], []
    for i in range(seq_len, len(feats) - horizon + 1):
        X.append(feats[i - seq_len:i])
        future_ret = close[i + horizon - 1] / close[i] - 1.0
        y.append(1.0 if future_ret > 0 else 0.0)
    if not X:
        return np.zeros((0, seq_len, feats.shape[1]), dtype=np.float32), np.zeros((0,))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class LSTMModel:
    """Direction classifier over the last 60 M15 candles (TF or NumPy)."""

    def __init__(self, pair: str, models_dir: str = "") -> None:
        self.pair = pair.upper()
        self.models_dir = models_dir or settings.DL_MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        self.ext = "keras" if TF_AVAILABLE else "npz"
        self.path = os.path.join(self.models_dir, f"lstm_{self.pair.lower()}.{self.ext}")
        self.trained_candles = 0
        self.tf = TF_AVAILABLE
        self._init_backend()

    @property
    def backend(self) -> str:
        """Active compute backend name."""
        return "tensorflow" if self.tf else "numpy"

    def _init_backend(self) -> None:
        if self.tf:
            self.model = self._build_tf()
        else:
            self.w, self.b = None, 0.0
        self._load()

    # ---- TF model ----

    def _build_tf(self):
        """2-layer LSTM (128, 64) + Dense sigmoid head."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(SEQ_LEN, 9)),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    # ---- persistence ----

    def save(self) -> None:
        """Persist model + metadata to the models dir."""
        try:
            if self.tf:
                self.model.save(self.path)
            else:
                np.savez(self.path, w=self.w, b=np.array([self.b]))
            with open(self.path + ".meta.json", "w") as fh:
                json.dump({"pair": self.pair, "trained_candles": self.trained_candles,
                           "backend": "tensorflow" if self.tf else "numpy"}, fh)
        except Exception as exc:
            logger.warning("lstm save failed for %s: %s", self.pair, exc)

    def _load(self) -> None:
        """Load persisted model when present."""
        try:
            if not os.path.exists(self.path):
                return
            if self.tf:
                self.model = tf.keras.models.load_model(self.path)
            else:
                data = np.load(self.path)
                self.w, self.b = data["w"], float(data["b"][0])
            meta = self.path + ".meta.json"
            if os.path.exists(meta):
                with open(meta) as fh:
                    self.trained_candles = json.load(fh).get("trained_candles", 0)
        except Exception as exc:
            logger.warning("lstm load failed for %s: %s", self.pair, exc)

    # ---- training ----

    def train(self, df: pd.DataFrame) -> dict:
        """Train on the last TRAIN_WINDOW candles; returns metrics."""
        feats = build_features(df)
        close = df["close"].to_numpy(dtype=float)
        X, y = make_sequences(feats, close)
        if len(X) < 200:
            return {"ok": False, "reason": f"insufficient sequences ({len(X)})"}
        X, y = X[-TRAIN_WINDOW - SEQ_LEN:], y[-TRAIN_WINDOW:]
        split = int(len(X) * 0.85)
        info: dict[str, Any] = {"ok": True, "backend": "tensorflow" if self.tf else "numpy"}
        if self.tf:
            self.model.fit(X, y, epochs=6, batch_size=64, verbose=0,
                           validation_split=0.1)
        else:
            # flatten last timestep + mean-pool over the window (stable, fast)
            Xf = np.concatenate([X[:, -1, :], X.mean(axis=1)], axis=1)
            self.w = np.zeros(Xf.shape[1], dtype=np.float32)
            self.b = 0.0
            lr = 0.05
            for _ in range(300):
                z = Xf @ self.w + self.b
                p = 1 / (1 + np.exp(-z))
                grad = (Xf.T @ (p - y)) / len(y)
                self.w -= lr * grad
                self.b -= lr * float(np.mean(p - y))
        preds = self.predict_batch(X[split:])
        acc = float(np.mean((preds > 0.5) == (y[split:] > 0.5))) * 100.0
        info["val_accuracy"] = round(acc, 1)
        info["sequences"] = int(len(X))
        self.trained_candles = int(len(df))
        self.save()
        return info

    def needs_retrain(self, candle_count: int) -> bool:
        """True when DL_RETRAIN_CANDLES new candles arrived."""
        return candle_count - self.trained_candles >= settings.DL_RETRAIN_CANDLES

    # ---- inference ----

    def _numpy_predict(self, feats: np.ndarray) -> float:
        Xf = np.concatenate([feats[-1], feats.mean(axis=0)])
        z = float(Xf @ self.w + self.b) if self.w is not None else 0.0
        return 1 / (1 + np.exp(-z))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Probabilities for a batch of sequences."""
        if len(X) == 0:
            return np.zeros((0,))
        if self.tf:
            return self.model.predict(X, verbose=0).ravel()
        return np.asarray([self._numpy_predict(x) for x in X])

    def predict(self, df: pd.DataFrame) -> dict:
        """Direction vote from the latest window: bullish/bearish + confidence."""
        if len(df) < SEQ_LEN + 10:
            return {"direction": "neutral", "confidence": 0.0, "ready": False}
        feats = build_features(df)
        prob = float(self.predict_batch(feats[np.newaxis, -SEQ_LEN:, :])[0])
        direction = "bullish" if prob > 0.5 else "bearish"
        confidence = abs(prob - 0.5) * 200.0  # 0..100
        return {"direction": direction, "confidence": round(confidence, 1),
                "p_up": round(prob, 4), "ready": True}
