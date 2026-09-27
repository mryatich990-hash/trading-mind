"""NLP sentiment (UPGRADE 2): FinBERT over financial text.

ProsusAI/finbert via transformers when installed; otherwise a finance-tuned
lexicon fallback (hawkish/dovish + bull/bear phrases) keeps scores flowing.
Aggregates per-currency sentiment over a rolling window and exposes a summary
for the Groq data package. Scores range -100 (max bearish) to +100 (max bullish).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)

try:  # guarded heavy dependency
    from transformers import pipeline as hf_pipeline  # type: ignore
    HF_AVAILABLE = True
except Exception:  # pragma: no cover
    hf_pipeline = None
    HF_AVAILABLE = False

CURRENCY_KEYWORDS = {
    "USD": ("dollar", "usd", "fed", "fomc", "treasury", "dxy"),
    "EUR": ("euro", "eur", "ecb", "eurozone", "bund"),
    "GBP": ("pound", "gbp", "boe", "sterling", "gilt"),
    "JPY": ("yen", "jpy", "boj", "bank of japan"),
    "XAU": ("gold", "xau", "bullion"),
}

BULL_TERMS = ("rate hike", "tightening", "hawkish", "robust growth", "inflation concern",
              "beat estimates", "record high", "strong demand", "upgrade", "rally")
BEAR_TERMS = ("rate cut", "accommodative", "dovish", "economic concern", "easing",
              "missed estimates", "recession", "downgrade", "selloff", "crisis", "panic")


class FinBertEngine:
    """Headline sentiment with per-currency aggregation."""

    def __init__(self, window_hours: float = 4.0) -> None:
        self.window_hours = window_hours
        self.backend = "finbert"
        self._pipe = None
        if HF_AVAILABLE:
            try:
                self._pipe = hf_pipeline(
                    "text-classification",
                    model="ProsusAI/finbert", top_k=None,
                    token=getattr(__import__("config.settings", fromlist=["HUGGINGFACE_TOKEN"]),
                                  "HUGGINGFACE_TOKEN", None) or None)
            except Exception as exc:
                logger.warning("finbert load failed, lexicon fallback active: %s", exc)
                self._pipe = None
        if self._pipe is None:
            self.backend = "lexicon"
        # currency -> deque[(ts, score -100..100)]
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

    # ---- scoring ----

    def score_text(self, text: str) -> dict:
        """Sentiment for one headline: {label, score(-100..100), confidence}."""
        if not text:
            return {"label": "neutral", "score": 0.0, "confidence": 0.0}
        if self._pipe is not None:
            try:
                result = self._pipe(text[:512])[0]
                best = max(result, key=lambda r: r["score"])
                label = best["label"].lower()
                signed = {"positive": 1, "negative": -1}.get(label, 0)
                return {"label": label, "score": round(signed * best["score"] * 100, 1),
                        "confidence": round(best["score"] * 100, 1)}
            except Exception as exc:
                logger.warning("finbert inference failed: %s", exc)
        # lexicon fallback
        lowered = text.lower()
        bull = sum(lowered.count(t) for t in BULL_TERMS)
        bear = sum(lowered.count(t) for t in BEAR_TERMS)
        if bull > bear:
            label, score = "positive", min(60 + 10 * (bull - bear), 95)
        elif bear > bull:
            label, score = "negative", -min(60 + 10 * (bear - bull), 95)
        else:
            label, score = "neutral", 0.0
        return {"label": label, "score": round(score, 1), "confidence": min(abs(score), 80.0)}

    def currencies_in(self, text: str) -> list[str]:
        """Currency codes mentioned by keyword."""
        lowered = text.lower()
        return [c for c, kws in CURRENCY_KEYWORDS.items()
                if any(kw in lowered for kw in kws)]

    # ---- aggregation ----

    def add_headline(self, text: str, ts: Optional[float] = None) -> dict:
        """Score one headline and fold it into per-currency history."""
        result = self.score_text(text)
        now = ts or time.time()
        for cur in self.currencies_in(text):
            self._history[cur].append((now, result["score"]))
        return result

    def aggregate(self, currency: str) -> dict:
        """Rolling sentiment per currency: mean score + sample count."""
        cutoff = time.time() - self.window_hours * 3600
        scores = [s for t, s in self._history.get(currency.upper(), deque())
                  if t >= cutoff]
        if not scores:
            return {"score": 0.0, "samples": 0}
        return {"score": round(sum(scores) / len(scores), 1), "samples": len(scores)}

    def summary(self) -> dict:
        """All currencies + backend info for the Groq package / dashboard."""
        return {"backend": self.backend,
                "window_hours": self.window_hours,
                "currencies": {c: self.aggregate(c) for c in CURRENCY_KEYWORDS}}
