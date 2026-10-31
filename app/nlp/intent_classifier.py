"""Intent classification: multilingual sentence embeddings + calibrated linear SVM."""

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.nlp.normalizer import TextNormalizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OOD_LABEL = "out_of_domain"
OOD_CONFIDENCE_THRESHOLD = 0.45


@dataclass(frozen=True)
class IntentResult:
    """Prediction for one message."""

    intent: str
    confidence: float
    is_out_of_domain: bool  # intent == 'out_of_domain' OR confidence below threshold
    all_scores: dict[str, float] = field(default_factory=dict)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class IntentClassifier:
    """Loads artifacts produced by scripts/train_intent.py."""

    def __init__(
        self,
        classifier_path: str = "data/models/intent_classifier.pkl",
        labels_path: str = "data/models/intent_labels.pkl",
        config_path: str = "data/models/intent_config.json",
        ood_threshold: float = OOD_CONFIDENCE_THRESHOLD,
    ) -> None:
        paths = [_resolve(p) for p in (classifier_path, labels_path, config_path)]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Intent model files not found: {missing}. Run: python scripts/train_intent.py"
            )
        import joblib
        from sentence_transformers import SentenceTransformer

        self._classifier: Any = joblib.load(paths[0])
        self._encoder: Any = joblib.load(paths[1])
        config = json.loads(paths[2].read_text(encoding="utf-8"))
        self._embedder = SentenceTransformer(config["embedding_model"])
        self._normalizer = TextNormalizer()
        self.ood_threshold = ood_threshold
        self.labels: list[str] = [str(c) for c in self._encoder.classes_]

    @property
    def embedder(self) -> Any:
        """The loaded SentenceTransformer, shared with the OOD detector to avoid a second load."""
        return self._embedder

    def classify(self, text: str, language: str | None = None) -> IntentResult:
        """Classify one message. `language` is accepted for API symmetry; the embedding model is
        multilingual so it does not change the prediction."""
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: list[str]) -> list[IntentResult]:
        """Classify many messages with a single embedding pass."""
        if not texts:
            return []
        cleaned = [self._normalizer.normalize(t) for t in texts]
        embeddings = self._embedder.encode(cleaned, batch_size=32, show_progress_bar=False)
        probas = self._classifier.predict_proba(embeddings)
        results: list[IntentResult] = []
        for row in probas:
            scores = {label: float(p) for label, p in zip(self.labels, row)}
            intent = max(scores, key=scores.__getitem__)
            confidence = scores[intent]
            results.append(
                IntentResult(
                    intent=intent,
                    confidence=confidence,
                    is_out_of_domain=intent == OOD_LABEL or confidence < self.ood_threshold,
                    all_scores=scores,
                )
            )
        return results


@lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    """Process-wide classifier instance (the embedding model takes a few seconds to load)."""
    return IntentClassifier()
