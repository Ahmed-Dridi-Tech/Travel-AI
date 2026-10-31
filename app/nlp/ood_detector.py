"""Second-opinion out-of-domain detector: cosine distance to travel-domain anchor phrases."""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.nlp.normalizer import TextNormalizer

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Travel anchors from the spec, plus smalltalk anchors: greetings must count as in-domain.
TRAVEL_ANCHORS = [
    "book a flight", "hotel reservation", "travel to Tunisia", "weather forecast",
    "currency exchange rate", "visit destination", "voyage en Tunisie", "réserver un hôtel",
    "météo", "nheb nsafer", "vol avion", "فندق",
]
SMALLTALK_ANCHORS = ["bonjour", "hello", "salam", "merci", "thank you", "au revoir", "goodbye", "السلام عليكم"]

COMBINED_CONF, COMBINED_DIST = 0.35, 0.7
LOW_CONF = 0.25
LOW_CONF_MIN_DIST = 0.5  # deviation from the spec: low confidence alone must not reject near-anchor queries
FAR_DIST = 0.75


@dataclass(frozen=True)
class OODResult:
    """Verdict of the embedding-distance check."""

    is_ood: bool
    reason: str  # intent_confidence | embedding_distance | combined | in_domain
    embedding_distance: float  # 1 - max cosine similarity to any anchor


class OODDetector:
    """Flags queries far from the travel domain. Anchors are encoded once at construction."""

    def __init__(self, threshold: float = 0.35, embedder: Any | None = None) -> None:
        self.threshold = threshold
        if embedder is None:
            from app.nlp.intent_classifier import get_intent_classifier

            embedder = get_intent_classifier().embedder
        self._embedder = embedder
        self._normalizer = TextNormalizer()
        self._anchors = self._encode(TRAVEL_ANCHORS + SMALLTALK_ANCHORS)

    def _encode(self, texts: list[str]) -> "np.ndarray":
        import numpy as np  # lazy: the lean deployment has no numpy

        vectors = np.asarray(self._embedder.encode(texts, show_progress_bar=False), dtype=np.float32)
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)

    def distance(self, text: str) -> float:
        """Cosine distance (1 - max similarity) from `text` to the nearest anchor."""
        vector = self._encode([self._normalizer.normalize(text) or " "])[0]
        return float(1.0 - (self._anchors @ vector).max())

    def is_out_of_domain(self, text: str, intent_confidence: float) -> OODResult:
        """Apply the combined / confidence / distance rules; default is to keep the query."""
        dist = self.distance(text)
        if intent_confidence < COMBINED_CONF and dist > COMBINED_DIST:
            return OODResult(True, "combined", dist)
        if intent_confidence < LOW_CONF and dist > LOW_CONF_MIN_DIST:
            return OODResult(True, "intent_confidence", dist)
        if dist > FAR_DIST:
            return OODResult(True, "embedding_distance", dist)
        return OODResult(False, "in_domain", dist)


@lru_cache(maxsize=1)
def get_ood_detector() -> OODDetector:
    """Process-wide detector instance."""
    return OODDetector()
