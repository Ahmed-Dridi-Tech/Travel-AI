"""Text normalization and script / Arabizi heuristics for multilingual input."""

import re
import unicodedata

_ZERO_WIDTH = re.compile("[​‌‍‎‏‪-‮⁠⁦-⁩﻿]")
_WHITESPACE = re.compile(r"\s+")
_DIACRITICS = re.compile("[ؐ-ًؚ-ٟ]")
_TATWEEL = "ـ"
_ARABIC_CHAR = re.compile("[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN_CHAR = re.compile("[A-Za-zÀ-ÖØ-öø-ɏ]")
_TOKEN = re.compile(r"[a-z0-9']+")

_ARABIC_MAP = str.maketrans(
    {"ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا",
     "ة": "ه", "ى": "ي"}
)

# Unambiguous Arabizi words: practically never valid French/English tokens, so one hit is decisive.
_STRONG_MARKERS = frozenset({
    "wesh", "nheb", "nabi", "besh", "bech", "mta3", "famma", "fama", "naaref", "barsha", "barcha",
    "chnowa", "chnia", "kifech", "ghoudwa", "ghodwa", "qaddesh", "9adech", "kaddech", "mouch",
    "behi", "yaatik", "aslema", "najem", "nemchi", "nsafer", "nesafer", "tayyara", "taw",
})
# Plausible in other languages ("ken" is an English name), so two are needed.
_WEAK_MARKERS = frozenset({"esh", "ken", "jaw", "sarf", "lazem", "bel", "fi", "el", "win"})

# digit used as a letter inside a word: mta3, 7aja, 9adech, n7eb ...
_DIGIT_AS_LETTER = re.compile(
    r"(?<=[a-z])[379](?=[a-z]|\b)|\b[379](?=[a-z]{2,})(?!(?:eme|ème|er|ère)\b)", re.IGNORECASE
)


class TextNormalizer:
    """Cleans text for NLP and exposes script/Arabizi heuristics. Stateless and thread-safe."""

    def normalize(self, text: str) -> str:
        """NFC-normalize, drop zero-width characters, collapse whitespace. Case is preserved."""
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = _ZERO_WIDTH.sub("", text)
        return _WHITESPACE.sub(" ", text).strip()

    def normalize_arabic(self, text: str) -> str:
        """normalize() plus alef/teh-marbuta/alef-maqsura unification; strips tashkeel and tatweel."""
        text = self.normalize(text)
        text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
        return text.translate(_ARABIC_MAP)

    def detect_script(self, text: str) -> str:
        """Return 'arabic', 'latin' or 'mixed' from the share of letters in each script.

        A script needs >30% of letters. When both exceed 30% the text is 'mixed'
        (a deliberate refinement: "مرحبا hello" is genuinely mixed). No letters -> 'mixed'.
        """
        arabic = len(_ARABIC_CHAR.findall(text))
        latin = len(_LATIN_CHAR.findall(text))
        total = arabic + latin
        if total == 0:
            return "mixed"
        arabic_share, latin_share = arabic / total, latin / total
        if arabic_share > 0.3 and latin_share > 0.3:
            return "mixed"
        if arabic_share > 0.3:
            return "arabic"
        if latin_share > 0.3:
            return "latin"
        return "mixed"

    def arabizi_score(self, text: str) -> int:
        """Evidence for Tunisian Arabizi: strong marker = 2, weak marker = 1, digit-as-letter = 2."""
        lowered = self.normalize(text).lower()
        tokens = _TOKEN.findall(lowered)
        score = sum(2 if t in _STRONG_MARKERS else 1 for t in tokens if t in _STRONG_MARKERS | _WEAK_MARKERS)
        if _DIGIT_AS_LETTER.search(lowered):
            score += 2
        return score

    def is_arabizi(self, text: str) -> bool:
        """True for Latin-script text showing Arabizi evidence (score >= 2)."""
        normalized = self.normalize(text)
        return self.detect_script(normalized) == "latin" and self.arabizi_score(normalized) >= 2
