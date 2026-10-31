"""NLP pipeline. Heavy dependencies (fasttext, sentence-transformers, spaCy) are imported lazily."""

from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.nlp.entity_extractor import DateEntity, EntityExtractor, EntityResult, get_entity_extractor
from app.nlp.intent_classifier import IntentClassifier, IntentResult, get_intent_classifier
from app.nlp.language_detector import LanguageDetector, LanguageResult, get_language_detector
from app.nlp.normalizer import TextNormalizer
from app.nlp.ood_detector import OODDetector, OODResult, get_ood_detector
from app.nlp.pipeline import NLPPipeline, NLPResult, get_nlp_pipeline


@lru_cache(maxsize=1)
def get_pipeline() -> Any:
    """The NLP pipeline for this process: keyword MockNLPPipeline when USE_MOCK_NLP=true
    (lean deployment, no ML packages), otherwise the real NLPPipeline."""
    if get_settings().use_mock_nlp:
        from app.nlp.mock_pipeline import MockNLPPipeline

        return MockNLPPipeline()
    return get_nlp_pipeline()


__all__ = [
    "DateEntity",
    "EntityExtractor",
    "EntityResult",
    "IntentClassifier",
    "IntentResult",
    "LanguageDetector",
    "LanguageResult",
    "NLPPipeline",
    "NLPResult",
    "OODDetector",
    "OODResult",
    "TextNormalizer",
    "get_entity_extractor",
    "get_intent_classifier",
    "get_language_detector",
    "get_nlp_pipeline",
    "get_pipeline",
    "get_ood_detector",
]
