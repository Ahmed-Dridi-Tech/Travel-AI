"""Single entry point running normalizer -> language -> intent -> OOD -> entities."""

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.nlp.entity_extractor import DateEntity, EntityResult
from app.nlp.normalizer import TextNormalizer

logger = logging.getLogger(__name__)

OOD_LABEL = "out_of_domain"
EXPLICIT_OOD_MIN_CONFIDENCE = 0.5


@dataclass(frozen=True)
class NLPResult:
    """Everything the chat service needs to know about one user message."""

    original_text: str
    normalized_text: str
    language: str
    script: str
    language_confidence: float
    is_tunisian: bool
    intent: str
    intent_confidence: float
    is_out_of_domain: bool
    ood_reason: str
    destinations: list[str] = field(default_factory=list)
    dates: list[DateEntity] = field(default_factory=list)
    travelers: int | None = None
    budget: str | None = None
    processing_time_ms: float = 0.0
    currencies: list[str] = field(default_factory=list)
    origin: str | None = None
    flight_destination: str | None = None

    @property
    def entities(self) -> EntityResult:
        """The extracted entities as an EntityResult (what the tool registry consumes)."""
        return EntityResult(
            destinations=list(self.destinations),
            dates=list(self.dates),
            travelers=self.travelers,
            budget=self.budget,
            currencies=list(self.currencies),
            origin=self.origin,
            flight_destination=self.flight_destination,
        )


class NLPPipeline:
    """Runs all NLP components; never raises. Components are injectable for tests."""

    def __init__(
        self,
        normalizer: Any | None = None,
        language_detector: Any | None = None,
        intent_classifier: Any | None = None,
        ood_detector: Any | None = None,
        entity_extractor: Any | None = None,
    ) -> None:
        started = time.perf_counter()
        from app.nlp.entity_extractor import get_entity_extractor
        from app.nlp.intent_classifier import get_intent_classifier
        from app.nlp.language_detector import get_language_detector
        from app.nlp.ood_detector import get_ood_detector

        self.normalizer = normalizer or TextNormalizer()
        self.language_detector = language_detector or get_language_detector()
        self.intent_classifier = intent_classifier or get_intent_classifier()
        self.ood_detector = ood_detector or get_ood_detector()
        self.entity_extractor = entity_extractor or get_entity_extractor()
        logger.info("NLPPipeline initialised in %.0f ms", (time.perf_counter() - started) * 1000)

    def analyze(self, text: str) -> NLPResult:
        """Analyze one message. On any component failure return a safe out-of-domain result."""
        started = time.perf_counter()
        stage = "normalizer"
        normalized = ""
        try:
            normalized = self.normalizer.normalize(text)
            stage = "language_detector"
            lang = self.language_detector.detect(normalized)
            stage = "intent_classifier"
            intent = self.intent_classifier.classify(normalized, lang.language)
            stage = "ood_detector"
            ood = self.ood_detector.is_out_of_domain(normalized, intent.confidence)
            stage = "entity_extractor"
            entities = self.entity_extractor.extract(normalized, lang.language, intent.intent)
        except Exception:  # noqa: BLE001 - the chat endpoint must survive any NLP failure
            logger.exception("NLP pipeline failed in %s", stage)
            return NLPResult(
                original_text=text, normalized_text=normalized, language="unknown", script="mixed",
                language_confidence=0.0, is_tunisian=False, intent=OOD_LABEL, intent_confidence=0.0,
                is_out_of_domain=True, ood_reason="pipeline_error",
                processing_time_ms=(time.perf_counter() - started) * 1000,
            )

        # Dismissal policy: an explicit, reasonably confident 'out_of_domain' label stands on its
        # own; a low-confidence verdict needs the embedding check to agree. Otherwise keep the query.
        explicit = intent.intent == OOD_LABEL and intent.confidence >= EXPLICIT_OOD_MIN_CONFIDENCE
        if explicit:
            is_ood, reason = True, "intent_label"
        elif intent.is_out_of_domain and ood.is_ood:
            is_ood, reason = True, ood.reason
        else:
            is_ood, reason = False, "in_domain"

        return NLPResult(
            original_text=text,
            normalized_text=normalized,
            language=lang.language,
            script=lang.script,
            language_confidence=lang.confidence,
            is_tunisian=lang.is_tunisian,
            intent=intent.intent,
            intent_confidence=intent.confidence,
            is_out_of_domain=is_ood,
            ood_reason=reason,
            destinations=entities.destinations,
            dates=entities.dates,
            travelers=entities.travelers,
            budget=entities.budget,
            processing_time_ms=(time.perf_counter() - started) * 1000,
            currencies=entities.currencies,
            origin=entities.origin,
            flight_destination=entities.flight_destination,
        )


@lru_cache(maxsize=1)
def get_nlp_pipeline() -> NLPPipeline:
    """Process-wide pipeline (loads fastText, embeddings, spaCy once)."""
    return NLPPipeline()
