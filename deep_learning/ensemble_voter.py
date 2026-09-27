"""Deep learning ensemble (UPGRADE 1): three-model voting gate.

LSTM gives a direction vote, XGBoost a win probability, the RL agent an
action. All three must agree on direction for a trade to proceed unmodified;
a "skip" vote demands extra confluence from the research engine instead of
hard-blocking. Every prediction/outcome pair is logged to dl_predictions for
per-model accuracy tracking; accuracy below DL_MIN_ACCURACY over the last 50
predictions triggers automatic retraining.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class EnsembleVerdict:
    """Combined verdict for one candidate trade."""

    proceed: bool
    require_extra_confluence: bool
    lstm: dict = field(default_factory=dict)
    xgboost: dict = field(default_factory=dict)
    rl: dict = field(default_factory=dict)
    agreement: str = "unknown"  # all_agree / split / skipped
    reason: str = ""


class EnsembleVoter:
    """Runs the three models and votes on every candidate trade."""

    def __init__(self, pairs: Optional[list[str]] = None) -> None:
        self.enabled = settings.DEEP_LEARNING_ENABLED
        self.lstm: dict[str, Any] = {}
        self.xgb: Optional[Any] = None
        self.rl: Optional[Any] = None
        self.last_retrain: dict[str, str] = {}
        self.pairs = pairs or []
        if self.enabled:
            self._init_models()

    def _init_models(self) -> None:
        """Instantiate models with per-pair LSTM (never raises)."""
        try:
            from deep_learning.lstm_model import LSTMModel

            for pair in self.pairs:
                self.lstm[pair] = LSTMModel(pair)
        except Exception as exc:
            logger.warning("lstm init failed: %s", exc)
        try:
            from deep_learning.xgboost_model import XGBoostModel

            self.xgb = XGBoostModel()
        except Exception as exc:
            logger.warning("xgboost init failed: %s", exc)
        try:
            from deep_learning.rl_agent import RLAgent

            self.rl = RLAgent()
        except Exception as exc:
            logger.warning("rl agent init failed: %s", exc)

    # ---- voting ----

    def vote(self, pair: str, direction: str, m15_df=None,
             ctx_dict: Optional[dict] = None, features: Optional[np.ndarray] = None) -> EnsembleVerdict:
        """Combined verdict for a candidate trade.

        proceed=False only when the ensemble is unambiguous; a skip vote
        downgrades to "extra confluence required" instead.
        """
        verdict = EnsembleVerdict(proceed=True, require_extra_confluence=False)
        if not self.enabled:
            verdict.reason = "deep learning disabled"
            return verdict

        # LSTM direction vote
        lstm_vote: dict = {"direction": "neutral", "confidence": 0.0, "ready": False}
        model = self.lstm.get(pair.upper())
        if model is not None and m15_df is not None:
            try:
                lstm_vote = model.predict(m15_df)
            except Exception as exc:
                logger.warning("lstm predict failed: %s", exc)
        verdict.lstm = lstm_vote

        # XGBoost win probability
        xgb_vote: dict = {"win_probability": None}
        if self.xgb is not None and ctx_dict:
            proba = self.xgb.predict_proba(ctx_dict)
            if proba is not None:
                xgb_vote = {"win_probability": proba}
        verdict.xgboost = xgb_vote

        # RL action
        rl_vote: dict = {"action": "skip", "confidence": 0.0}
        if self.rl is not None:
            try:
                rl_vote = self.rl.act(ctx_dict or {}, features=features)
            except Exception as exc:
                logger.warning("rl act failed: %s", exc)
        verdict.rl = rl_vote

        # agreement logic
        lstm_dir = lstm_vote.get("direction", "neutral")
        lstm_agrees = lstm_dir == "neutral" or lstm_dir == direction
        wp = xgb_vote.get("win_probability")
        xgb_ok = wp is None or wp >= 50.0
        rl_action = rl_vote.get("action", "skip")

        if rl_action == "skip" and not lstm_agrees:
            verdict.proceed = False
            verdict.agreement = "skipped"
            verdict.reason = f"lstm={lstm_dir} rl=skip"
        elif rl_action == "skip":
            verdict.require_extra_confluence = True
            verdict.agreement = "split"
            verdict.reason = "rl=skip -> extra confluence required"
        elif not lstm_agrees:
            verdict.require_extra_confluence = True
            verdict.agreement = "split"
            verdict.reason = f"lstm={lstm_dir} vs {direction}"
        elif wp is not None and wp < 40.0:
            verdict.require_extra_confluence = True
            verdict.agreement = "split"
            verdict.reason = f"xgb={wp}% too low"
        else:
            verdict.agreement = "all_agree"
            verdict.reason = f"lstm={lstm_dir} xgb={wp} rl={rl_action}"
        return verdict

    # ---- outcome logging + accuracy ----

    def record_outcome(self, pair: str, direction: str, verdict: EnsembleVerdict,
                       won: bool) -> None:
        """Persist prediction/outcome for accuracy tracking + RL learning."""
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine, _WRITE_LOCK

            row = {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pair": pair, "direction": direction,
                "lstm_dir": verdict.lstm.get("direction", ""),
                "lstm_conf": verdict.lstm.get("confidence", 0.0),
                "xgb_prob": verdict.xgboost.get("win_probability"),
                "rl_action": verdict.rl.get("action", ""),
                "agreement": verdict.agreement, "won": 1 if won else 0,
            }
            with engine.begin() as conn, _WRITE_LOCK:
                conn.execute(sqltext(
                    "INSERT INTO dl_predictions (created_at, pair, direction, lstm_dir, "
                    "lstm_conf, xgb_prob, rl_action, agreement, won) "
                    "VALUES (:t, :pair, :direction, :lstm_dir, :lstm_conf, :xgb_prob, "
                    ":rl_action, :agreement, :won)"), row)
            if self.rl is not None:
                self.rl.learn_outcome({}, verdict.rl.get("action", "skip"), won)
        except Exception as exc:
            logger.warning("dl outcome logging failed: %s", exc)

    def rolling_accuracy(self, window: int = 50) -> dict:
        """Per-model accuracy over the last `window` decided predictions."""
        out = {"lstm": None, "xgboost": None, "rl": None, "window": window}
        try:
            from sqlalchemy import text as sqltext

            from core.db import engine

            with engine.begin() as conn:
                rows = conn.execute(sqltext(
                    "SELECT lstm_dir, xgb_prob, rl_action, direction, won "
                    "FROM dl_predictions ORDER BY id DESC LIMIT :w"),
                    {"w": window * 3}).mappings().all()
            rows = list(rows)[:window * 3]
            lstm = [r for r in rows if r["lstm_dir"] not in ("", "neutral")]
            out["lstm"] = round(np.mean([
                (r["lstm_dir"] == r["direction"]) == (r["won"] == 1)
                for r in lstm]) * 100, 1) if lstm else None
            xgb = [r for r in rows if r["xgb_prob"] is not None]
            out["xgboost"] = round(np.mean([
                ((r["xgb_prob"] >= 50) == (r["won"] == 1)) for r in xgb]) * 100, 1) if xgb else None
            rl = [r for r in rows if r["rl_action"] in ("buy", "sell")]
            out["rl"] = round(np.mean([
                (r["rl_action"] == r["direction"]) == (r["won"] == 1)
                for r in rl]) * 100, 1) if rl else None
        except Exception as exc:
            logger.debug("accuracy query failed: %s", exc)
        return out

    def models_need_retrain(self) -> list[str]:
        """Model names below the accuracy floor over the last 50 predictions."""
        acc = self.rolling_accuracy(50)
        return [m for m, a in acc.items() if m != "window"
                and a is not None and a < settings.DL_MIN_ACCURACY]

    def status(self) -> dict:
        """Dashboard payload."""
        return {
            "enabled": self.enabled,
            "backends": {
                "lstm": {p: ("tensorflow" if m.tf else "numpy") for p, m in self.lstm.items()},
                "xgboost": getattr(self.xgb, "backend", None) if self.xgb else None,
                "rl": getattr(self.rl, "backend", None) if self.rl else None,
            },
            "accuracy": self.rolling_accuracy(),
            "accuracy_floor": settings.DL_MIN_ACCURACY,
            "needs_retrain": self.models_need_retrain(),
            "last_retrain": self.last_retrain,
        }
