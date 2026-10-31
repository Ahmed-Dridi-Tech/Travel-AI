"""Language + Tunisian-dialect detection: script rules, keywords for short texts, then fastText."""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.nlp.normalizer import TextNormalizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = "data/models/lid.176.bin"
SUPPORTED_LANGUAGES = ["fr", "en", "ar", "tun_ar", "tun_lat"]

MIN_CONFIDENCE = 0.5  # fastText verdicts below this are not trusted
KEYWORD_CONFIDENCE = 0.75
TUNISIAN_FLOOR = 0.6  # lid.176 has no Tunisian class; heuristic evidence sets its own confidence
ARABIC_SCRIPT_FLOOR = 0.5
LOW_CONFIDENCE_GUESS = 0.3  # last-resort script-based guess
MIN_RELIABLE_CONFIDENCE = 0.35  # callers should treat anything below this as "not really detected"
SHORT_TEXT_WORDS = 4  # fewer words than this -> try keywords before fastText

# Spec markers plus a few extra. "Weak" ones also occur in MSA, so two are required.
_TUN_AR_STRONG = frozenset({
    "باش", "ماش", "فما", "برشا", "برشة", "يزي", "نحب", "كيفاش", "علاش", "شنو", "شنوة", "شنية",
    "متاع", "هكا", "توا", "بكري", "وين", "قداش", "موش", "باهي", "نمشي", "نجم", "ياسر", "عسلامة",
})
_TUN_AR_WEAK = frozenset({"ماكان", "نعرف", "يسر", "نسافر", "نعمل", "غدوة"})
_ARABIC_TOKEN = re.compile(r"[؀-ۿ]+")
_ARABIC_CHAR = re.compile(r"[؀-ۿ]")
_LATIN_CHAR = re.compile(r"[A-Za-zÀ-ɏ]")
_WORD = re.compile(r"[\w']+")
_ARABIC_SCRIPT_LANGS = frozenset({"ar", "arz", "ary", "fa", "ur", "ckb", "ps", "azb", "sd"})
_LABEL_PREFIX = "__label__"

# High-signal words for short texts. Words that are valid in several languages
# ("hotel", "destination", "visa", "taxi") are deliberately absent.
_TUN_LAT_KEYWORDS = frozenset({
    "nheb", "nabi", "chnowa", "chnia", "wesh", "barsha", "barcha", "besh", "bech", "mta3", "jaw", "famma",
    "yezzi", "yezzi", "kifech", "ghoudwa", "ghodwa", "salam", "aaslema", "aslema", "3aslema", "lazem", "behi",
})
_FR_KEYWORDS = frozenset({
    "météo", "meteo", "vol", "vols", "hôtel", "merci", "bonjour", "bonsoir", "convertir", "combien",
    "quel", "quelle", "comment", "voyage", "billet", "séjour", "sejour", "taux",
})
_EN_KEYWORDS = frozenset({
    "weather", "flight", "flights", "book", "what", "where", "when", "price", "trip", "travel", "hello",
    "hi", "thanks", "convert", "exchange", "ticket", "tickets", "cheap",
})
_EN_PHRASES = ("how much", "thank you")
_AR_KEYWORDS = frozenset({"طقس", "رحلة", "فندق", "كيف", "أين", "اين", "متى", "سعر", "مرحبا"})


@dataclass(frozen=True)
class LanguageResult:
    """Outcome of language detection."""

    language: str  # fr | en | ar | tun_ar | tun_lat | unknown (only for empty input)
    script: str  # arabic | latin | mixed
    confidence: float  # < MIN_RELIABLE_CONFIDENCE means a last-resort guess
    raw_fasttext_lang: str  # e.g. '__label__ar'; '' when fastText was not consulted
    is_tunisian: bool


class LanguageDetector:
    """Detects fr / en / ar / tun_ar / tun_lat. Load once (see get_language_detector)."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        path = Path(model_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"fastText model not found at {path}. Run: python scripts/download_models.py"
            )
        import fasttext  # heavy import deferred so the API can boot without the nlp extra

        self._model: Any = fasttext.load_model(str(path))
        self._normalizer = TextNormalizer()

    def _predict(self, text: str) -> tuple[str, float]:
        """Top fastText label and probability.

        Calls the C++ binding directly: FastText.predict() breaks on numpy>=2 (copy=False).
        """
        pairs = self._model.f.predict(text, 1, 0.0, "strict")
        if not pairs:
            return "", 0.0
        prob, label = pairs[0]
        return str(label), min(1.0, max(0.0, float(prob)))

    def _tun_ar_score(self, text: str) -> int:
        tokens = _ARABIC_TOKEN.findall(text)
        return sum(2 if t in _TUN_AR_STRONG else 1 for t in tokens if t in _TUN_AR_STRONG | _TUN_AR_WEAK)

    def _keyword_detect(self, text: str, script: str) -> str | None:
        """Language from high-signal words, or None when there is no hit or the hits conflict."""
        lowered = text.lower()
        tokens = set(_WORD.findall(lowered))
        if script != "arabic" and (tokens & _TUN_LAT_KEYWORDS):
            return "tun_lat"
        hits = {
            "fr": bool(tokens & _FR_KEYWORDS),
            "en": bool(tokens & _EN_KEYWORDS) or any(p in lowered for p in _EN_PHRASES),
            "ar": bool(tokens & _AR_KEYWORDS),
        }
        matched = [lang for lang, hit in hits.items() if hit]
        return matched[0] if len(matched) == 1 else None  # a conflict ("convert 500 euros") defers to fastText

    def _guess_from_script(self, text: str, script: str) -> str:
        if script == "arabic":
            return "ar"
        if script == "mixed" and len(_ARABIC_CHAR.findall(text)) > len(_LATIN_CHAR.findall(text)):
            return "ar"
        return "fr"

    def detect(self, text: str) -> LanguageResult:
        """Detect language and dialect of one message. Only empty input yields 'unknown'."""
        normalized = self._normalizer.normalize(text)
        if not normalized:
            return LanguageResult("unknown", "mixed", 0.0, "", False)

        script = self._normalizer.detect_script(normalized)
        raw_label, ft_conf = self._predict(normalized)
        ft_lang = raw_label.removeprefix(_LABEL_PREFIX)

        # Tier 1: Arabic script is decisive (Tunisian markers -> tun_ar, otherwise MSA)
        if script == "arabic":
            if self._tun_ar_score(normalized) >= 2:
                return LanguageResult("tun_ar", script, round(max(ft_conf, TUNISIAN_FLOOR), 4), raw_label, True)
            conf = ft_conf if ft_lang in _ARABIC_SCRIPT_LANGS else 0.0
            return LanguageResult("ar", script, round(max(conf, ARABIC_SCRIPT_FLOOR), 4), raw_label, False)

        # Tier 2: keywords rescue short texts, where fastText is unreliable
        if len(normalized.split()) < SHORT_TEXT_WORDS:
            hit = self._keyword_detect(normalized, script)
            if hit:
                return LanguageResult(hit, script, KEYWORD_CONFIDENCE, raw_label, hit == "tun_lat")

        # Tier 3: Arabizi evidence, then fastText
        if script == "latin":
            arabizi = self._normalizer.arabizi_score(normalized)
            if arabizi >= 2:  # checked for any fastText label: lid.176 scatters Arabizi over many languages
                base = ft_conf if ft_lang in {"fr", "en", "ar"} else 0.0
                return LanguageResult("tun_lat", script, round(max(base, min(0.95, TUNISIAN_FLOOR + 0.1 * arabizi)), 4), raw_label, True)
            if ft_lang in {"fr", "en"} and ft_conf >= MIN_CONFIDENCE:
                return LanguageResult(ft_lang, script, round(ft_conf, 4), raw_label, False)
        elif ft_lang in {"fr", "en", "ar"} and ft_conf >= MIN_CONFIDENCE:  # mixed script
            return LanguageResult(ft_lang, script, round(ft_conf, 4), raw_label, False)

        # Tier 4: never give up on non-empty text; callers see the low confidence
        return LanguageResult(self._guess_from_script(normalized, script), script, LOW_CONFIDENCE_GUESS, raw_label, False)

    def detect_batch(self, texts: list[str]) -> list[LanguageResult]:
        """Detect each text; the model call is a cheap in-process C++ call per item."""
        return [self.detect(t) for t in texts]

    def get_supported_languages(self) -> list[str]:
        """Language codes this detector can return (besides 'unknown' for empty input)."""
        return list(SUPPORTED_LANGUAGES)


@lru_cache(maxsize=1)
def get_language_detector() -> LanguageDetector:
    """Process-wide detector instance (model load takes ~1 s)."""
    return LanguageDetector()
