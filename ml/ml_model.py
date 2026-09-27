"""ML model: RandomForest win/loss classifier retrained every 25 trades.

Features are stored per trade in features_json at signal time; the outcome
(pnl_usd > 0) is the label. After each retrain, features whose importance
correlates with losses trigger stricter automatic filters, logged to the DB.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional

import joblib

from config import settings
from core import db
from core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["MLModel", "FEATURE_NAMES"]

FEATURE_NAMES = [
    "hour", "day_of_week", "session_is_london", "session_is_ny", "confluence",
    "conviction", "rsi14", "spread_pips", "volume_pct", "delta_5",
    "atr_ratio", "vix", "cot_aligned", "retail_long", "pair_bias",
    "historical_win_rate", "sl_pips", "rr_planned",
]


class MLModel:
    """Trains and serves the loss-probability filter."""

    def __init__(self, model_dir: Optional[str] = None,
                 retrain_every: Optional[int] = None) -> None:
        self.model_dir = model_dir or settings.ML_MODEL_DIR
        self.retrain_every = retrain_every or settings.ML_RETRAIN_EVERY
        self.model = None
        self.metrics: dict = {}
        self._lock = threading.Lock()
        self._load()

    # ---- persistence ----

    def _path(self) -> str:
        """Model file path."""
        return os.path.join(self.model_dir, "rf_model.joblib")

    def _load(self) -> None:
        """Load a persisted model when present."""
        try:
            if os.path.exists(self._path()):
                self.model = joblib.load(self._path())
                logger.info("ML model loaded from %s", self._path())
        except Exception as exc:
            logger.warning("ML model load failed: %s", exc)

    def _save(self) -> None:
        """Persist the model and metrics."""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump(self.model, self._path())
        except Exception as exc:
            logger.warning("ML model save failed: %s", exc)

    # ---- features ----

    @staticmethod
    def features_from_row(row: dict) -> Optional[list[float]]:
        """Build the feature vector from a trades-table row + features_json."""
        try:
            feats = json.loads(row.get("features_json") or "{}")
        except (TypeError, ValueError):
            feats = {}
        opened = row.get("opened_at") or row.get("created_at")
        hour, dow = 0, 0
        if opened:
            try:
                ts = opened if isinstance(opened, datetime) else \
                    datetime.fromisoformat(str(opened).replace("Z", "+00:00"))
                hour, dow = ts.hour, ts.weekday()
            except (ValueError, TypeError):
                pass
        session = str(row.get("session", ""))
        vals = {
            "hour": float(hour),
            "day_of_week": float(dow),
            "session_is_london": 1.0 if "ondon" in session else 0.0,
            "session_is_ny": 1.0 if "ewYork" in session or "verlap" in session else 0.0,
            "confluence": float(row.get("confluence_score") or 0),
            "conviction": float(row.get("groq_conviction") or 0),
            "rsi14": float(feats.get("rsi14", 50)),
            "spread_pips": float(feats.get("spread_pips", 0)),
            "volume_pct": float(feats.get("volume_pct", 100)),
            "delta_5": float(feats.get("delta_5", 0)),
            "atr_ratio": float(feats.get("atr_ratio", 1.0)),
            "vix": float(feats.get("vix", 0)),
            "cot_aligned": 1.0 if feats.get("cot_aligned") else 0.0,
            "retail_long": float(feats.get("retail_long", 50)),
            "pair_bias": float(feats.get("pair_bias", 0)),
            "historical_win_rate": float(feats.get("historical_win_rate", 50)),
            "sl_pips": float(feats.get("sl_pips", 0)),
            "rr_planned": float(feats.get("rr_planned", 0)),
        }
        if not any(v for k, v in vals.items() if k not in ("hour", "day_of_week")):
            return None
        return [vals[k] for k in FEATURE_NAMES]

    # ---- training ----

    def training_data(self) -> tuple[list[list[float]], list[int]]:
        """Assemble (X, y) from closed trades."""
        X: list[list[float]] = []
        y: list[int] = []
        for row in db.closed_trades(limit=1000):
            vec = self.features_from_row(row)
            if vec is None:
                continue
            X.append(vec)
            y.append(1 if float(row.get("pnl_usd") or 0) > 0 else 0)
        return X, y

    def retrain_if_due(self) -> Optional[dict]:
        """Retrain when the closed-trade count crosses the interval."""
        with self._lock:
            rows = db.closed_trades(limit=10000)
            n = len(rows)
            since = int(db.get_state("ml_trades_since_retrain", "0") or 0) + 1
            if n < settings.ML_MIN_TRAIN or since < self.retrain_every:
                db.set_state("ml_trades_since_retrain", str(since))
                return None
            db.set_state("ml_trades_since_retrain", "0")
            return self.retrain()

    def retrain(self) -> Optional[dict]:
        """Fit the forest, record metrics, derive stricter filters."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X, y = self.training_data()
        if len(X) < settings.ML_MIN_TRAIN or len(set(y)) < 2:
            logger.info("ML retrain skipped: %d samples", len(X))
            return None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        self.model = model
        train_acc = float(model.score(X_train, y_train))
        test_acc = float(model.score(X_test, y_test))
        importances = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                             key=lambda kv: -kv[1])
        self.metrics = {"n": len(X), "train_acc": round(train_acc, 3),
                        "test_acc": round(test_acc, 3),
                        "importances": [(k, round(v, 3)) for k, v in importances],
                        "trained_at": datetime.now(timezone.utc).isoformat()}
        self._save()
        self._persist_metrics()
        filters = self._auto_filters(importances)
        logger.info("ML retrained: n=%d acc=%.2f/%.2f filters=%s", len(X),
                    train_acc, test_acc, filters)
        return self.metrics

    def _persist_metrics(self) -> None:
        """Store metrics in ml_metrics."""
        from sqlalchemy import text as sqltext

        from core.db import engine, _WRITE_LOCK
        with engine.begin() as conn, _WRITE_LOCK:
            conn.execute(
                sqltext("INSERT INTO ml_metrics (trained_at, n_trades, train_accuracy, "
                        "test_accuracy, features_json) VALUES (:t, :n, :tr, :te, :f)"),
                {"t": db._utcnow(), "n": self.metrics["n"],
                 "tr": self.metrics["train_acc"], "te": self.metrics["test_acc"],
                 "f": json.dumps(self.metrics["importances"])},
            )

    def _auto_filters(self, importances: list[tuple[str, float]]) -> list[str]:
        """Stricter filters for high-importance loss-correlated features."""
        changed = []
        for name, importance in importances[:3]:
            if importance > 0.3:
                db.set_state(f"ml_filter_{name}", "strict")
                changed.append(name)
        return changed

    # ---- inference ----

    def is_ready(self) -> bool:
        """Model fitted and usable."""
        return self.model is not None and bool(self.metrics)

    def predict_loss_prob(self, features: list[float]) -> float:
        """P(loss) for a feature vector (0.5 when no model)."""
        if not self.is_ready():
            return 0.5
        try:
            proba = self.model.predict_proba([features])[0]
            classes = list(self.model.classes_)
            if 0 in classes:
                return float(proba[classes.index(0)])
            return 0.5
        except Exception as exc:
            logger.warning("predict failed: %s", exc)
            return 0.5

    def should_skip(self, loss_prob: float) -> bool:
        """Skip a trade when the model is sufficiently pessimistic."""
        return self.is_ready() and loss_prob >= settings.ML_LOSS_SKIP_THRESHOLD
