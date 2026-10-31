"""Keyword-based stand-in for NLPPipeline, for the lean deployment without ML packages.

Selected by USE_MOCK_NLP=true (see app.nlp.get_pipeline). It is a demo fallback, not production
NLP: intents come from a handful of keywords. Entities (places, dates, travelers, budget, flight
route) still come from the real rule-based EntityExtractor, which needs no ML dependencies.
"""

import logging
import re
import time

from app.nlp.entity_extractor import EntityExtractor, _fold
from app.nlp.normalizer import TextNormalizer
from app.nlp.pipeline import NLPResult

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[\w']+")
_ARABIC = re.compile(r"[؀-ۿ]")

# Checked in this order; keywords are accent-folded tokens (whole words).
_INTENT_KEYWORDS: list[tuple[str, frozenset[str]]] = [
    ("weather", frozenset({"meteo", "weather", "jaw", "طقس", "temperature", "forecast"})),
    ("currency", frozenset({"euro", "euros", "dinar", "dinars", "change", "exchange", "sarf", "taux", "rate",
                            "dollar", "dollars", "usd", "eur", "tnd", "دينار", "صرف"})),
    ("flight_inquiry", frozenset({"vol", "vols", "flight", "flights", "tayyara", "billet", "avion", "طيارة"})),
    ("hotel_inquiry", frozenset({"hotel", "hotels", "logement", "chambre", "فندق", "accommodation"})),
    ("travel_faq", frozenset({"visa", "viza", "document", "documents", "passeport", "passport", "فيزا"})),
    ("smalltalk", frozenset({"bonjour", "salam", "merci", "hello", "hi", "thanks", "aaslema", "aslema", "bonsoir"})),
    ("destination", frozenset({"visiter", "visit", "destination", "destinations", "voyage", "trip", "plage",
                               "sejour", "partir", "excursion"})),
]
_TUN_LAT_WORDS = frozenset({"nheb", "chnowa", "wesh", "naaref", "barsha", "famma", "bech", "mta3", "nabi"})
_EN_HINTS = frozenset({"the", "what", "whats", "how", "much", "is", "are", "for", "to", "from", "i", "want",
                       "need", "book", "please", "rate", "exchange", "weather", "flight", "hello", "thanks", "in"})


class MockNLPPipeline:
    """Same interface as NLPPipeline: analyze(text) -> NLPResult. Never raises."""

    def __init__(self) -> None:
        started = time.perf_counter()
        self.normalizer = TextNormalizer()
        self.entity_extractor = EntityExtractor()  # gazetteer-only when spaCy is not installed
        logger.info("MockNLPPipeline initialised in %.0f ms", (time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------------ heuristics
    def _language(self, text: str, tokens: set[str]) -> str:
        if _ARABIC.search(text):
            return "tun_ar"
        if tokens & _TUN_LAT_WORDS or self.normalizer.arabizi_score(text) >= 2:
            return "tun_lat"
        if len(tokens & _EN_HINTS) >= 2:
            return "en"
        return "fr"

    @staticmethod
    def _intent(text: str, tokens: set[str], has_destination: bool) -> tuple[str, float]:
        for intent, keywords in _INTENT_KEYWORDS:
            if tokens & keywords or any(k in text for k in keywords if _ARABIC.search(k)):
                return intent, 0.9
        if has_destination:  # a bare place name: the customer wants to know about it
            return "destination", 0.6
        return "out_of_domain", 0.5

    # ------------------------------------------------------------------ public API
    def analyze(self, text: str) -> NLPResult:
        """Analyze one message with keyword rules."""
        started = time.perf_counter()
        try:
            normalized = self.normalizer.normalize(text)
            folded = _fold(normalized)
            tokens = set(_WORD.findall(folded))
            language = self._language(normalized, tokens)
            probe = self.entity_extractor.extract(normalized, language)
            intent, confidence = self._intent(folded, tokens, bool(probe.destinations))
            entities = self.entity_extractor.extract(normalized, language, intent)  # route parsing needs the intent
        except Exception:  # noqa: BLE001 - same contract as NLPPipeline: never raise
            logger.exception("MockNLPPipeline failed")
            return NLPResult(
                original_text=text, normalized_text="", language="unknown", script="mixed", language_confidence=0.0,
                is_tunisian=False, intent="out_of_domain", intent_confidence=0.0, is_out_of_domain=True,
                ood_reason="pipeline_error", processing_time_ms=(time.perf_counter() - started) * 1000,
            )
        is_ood = intent == "out_of_domain"
        return NLPResult(
            original_text=text,
            normalized_text=normalized,
            language=language,
            script=self.normalizer.detect_script(normalized),
            language_confidence=0.8,
            is_tunisian=language in {"tun_ar", "tun_lat"},
            intent=intent,
            intent_confidence=confidence,
            is_out_of_domain=is_ood,
            ood_reason="mock_no_match" if is_ood else "in_domain",
            destinations=entities.destinations,
            dates=entities.dates,
            travelers=entities.travelers,
            budget=entities.budget,
            processing_time_ms=(time.perf_counter() - started) * 1000,
            currencies=entities.currencies,
            origin=entities.origin,
            flight_destination=entities.flight_destination,
        )

