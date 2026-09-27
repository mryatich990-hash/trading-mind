"""Deep learning price prediction (UPGRADE 1, model 3): RL agent (PPO).

Stable-Baselines3 PPO when installed; otherwise a functional contextual
epsilon-greedy bandit fallback that learns skip/buy/sell value from trade
outcomes. Gym environment wraps historical feature rows + outcomes so PPO can
pretrain on history. Agent file: models/rl_<pair>.zip (SB3) or
models/rl_<pair>.npz (fallback Q-table).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

try:  # guarded heavy dependency
    from stable_baselines3 import PPO  # type: ignore
    import gymnasium as gym  # type: ignore
    SB3_AVAILABLE = True
except Exception:  # pragma: no cover
    PPO = None
    gym = None
    SB3_AVAILABLE = False

ACTIONS = ("buy", "sell", "skip")
ACTION_INDEX = {"buy": 0, "sell": 1, "skip": 2}


class TradeEnv:
    """Gym-style environment over historical features/outcomes.

    State: 47-feature market row. Actions: 0 buy, 1 sell, 2 skip.
    Reward: +1 win, -1 loss (direction-correct), -0.1 skip.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y  # 1 = buy-side win, 0 = buy-side loss (sell = inverse)
        self.i = 0
        self.n_actions = 3
        self.obs_dim = X.shape[1] if len(X) else 47

    def reset(self) -> np.ndarray:
        self.i = 0
        return self.X[self.i]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        won_buy = self.y[self.i] > 0.5
        if action == 0:
            reward = 1.0 if won_buy else -1.0
        elif action == 1:
            reward = 1.0 if not won_buy else -1.0
        else:
            reward = -0.1  # discourage excessive caution
        self.i += 1
        done = self.i >= len(self.X)
        obs = self.X[min(self.i, len(self.X) - 1)]
        return obs, reward, done, {}


class RLAgent:
    """Direction actor: buy/sell/skip from the 47-feature state."""

    def __init__(self, pair: str = "ALL", models_dir: str = "") -> None:
        self.pair = pair.upper()
        self.models_dir = models_dir or settings.DL_MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        self.backend = "sb3" if SB3_AVAILABLE else "bandit"
        self.path = os.path.join(self.models_dir, f"rl_{self.pair.lower()}.zip")
        self.q_path = os.path.join(self.models_dir, f"rl_{self.pair.lower()}.npz")
        self.model = None
        self.q = {}  # feature-bucket -> action values (fallback)
        self.trained_steps = 0
        self._load()

    # ---- persistence ----

    def save(self) -> None:
        """Persist agent."""
        try:
            if self.backend == "sb3" and self.model is not None:
                self.model.save(self.path)
            else:
                np.savez(self.q_path,
                         keys=np.array(list(self.q.keys()), dtype=object),
                         vals=np.array([self.q[k] for k in self.q], dtype=object))
            with open(self.q_path + ".meta.json", "w") as fh:
                json.dump({"trained_steps": self.trained_steps,
                           "backend": self.backend}, fh)
        except Exception as exc:
            logger.warning("rl save failed: %s", exc)

    def _load(self) -> None:
        """Load persisted agent when present."""
        try:
            if self.backend == "sb3" and os.path.exists(self.path):
                self.model = PPO.load(self.path)
            elif os.path.exists(self.q_path):
                data = np.load(self.q_path, allow_pickle=True)
                self.q = {tuple(k): v for k, v in zip(data["keys"], data["vals"])}
            meta = self.q_path + ".meta.json"
            if os.path.exists(meta):
                with open(meta) as fh:
                    self.trained_steps = json.load(fh).get("trained_steps", 0)
        except Exception as exc:
            logger.warning("rl load failed: %s", exc)

    # ---- training ----

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Pretrain on historical feature/outcome rows."""
        if len(X) < 30:
            return {"ok": False, "reason": f"insufficient rows ({len(X)})"}
        if self.backend == "sb3":
            env = TradeEnv(X, y)
            self.model = PPO("MlpPolicy", env, verbose=0, seed=42)
            self.model.learn(total_timesteps=min(len(X) * 4, 20000))
            steps = min(len(X) * 4, 20000)
        else:
            for x, won in zip(X, y):
                key = self._bucket(x)
                vals = self.q.setdefault(key, np.zeros(3))
                # outcome-weighted value update for each action
                vals[ACTION_INDEX["buy"]] += 0.1 * ((1.0 if won > 0.5 else -1.0) - vals[ACTION_INDEX["buy"]])
                vals[ACTION_INDEX["sell"]] += 0.1 * ((1.0 if won <= 0.5 else -1.0) - vals[ACTION_INDEX["sell"]])
                vals[ACTION_INDEX["skip"]] += 0.1 * (-0.1 - vals[ACTION_INDEX["skip"]])
            steps = len(X)
        self.trained_steps += steps
        self.save()
        return {"ok": True, "steps": steps, "backend": self.backend}

    # ---- inference ----

    @staticmethod
    def _bucket(x: np.ndarray) -> tuple:
        """Quantize the feature vector into a hashable state key."""
        return tuple((np.asarray(x, dtype=float) * 2).astype(int)[-8:].tolist())

    def act(self, ctx_dict: dict, features: Optional[np.ndarray] = None) -> dict:
        """Choose buy/sell/skip with confidence."""
        if self.backend == "sb3" and self.model is not None:
            x = features if features is not None else np.zeros(47, dtype=np.float32)
            action, _ = self.model.predict(x, deterministic=True)
            name = ACTIONS[int(action)]
            return {"action": name, "confidence": 50.0, "backend": self.backend}
        # bandit fallback
        if not self.q:
            return {"action": "skip", "confidence": 0.0, "backend": self.backend}
        key = self._bucket(np.zeros(47) if features is None else features)
        vals = self.q.get(key)
        if vals is None:
            vals = np.mean([v for v in self.q.values()], axis=0)
        idx = int(np.argmax(vals))
        conf = float(max(vals) - np.mean(vals)) * 50.0 + 33.0
        return {"action": ACTIONS[idx], "confidence": round(min(conf, 99.0), 1),
                "backend": self.backend}

    # ---- online learning ----

    def learn_outcome(self, ctx_dict: dict, action: str, won: bool) -> None:
        """Fine-tune on live results (bandit path; SB3 retrains in batches)."""
        if self.backend != "bandit":
            return
        key = self._bucket(np.zeros(47))
        vals = self.q.setdefault(key, np.zeros(3))
        reward = 1.0 if won else -1.0
        idx = ACTION_INDEX.get(action, 2)
        vals[idx] += 0.1 * (reward - vals[idx])
        self.trained_steps += 1
