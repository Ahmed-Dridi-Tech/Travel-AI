"""Travel entity extraction: destinations (gazetteer + spaCy), dates, travelers, budget."""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.nlp.normalizer import TextNormalizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AR_PREFIX = "[وفبلك]?"  # attached Arabic particles: و (and) ف/ب (in/with) ل (to) ك (like)
_NORM = TextNormalizer()


def _fold(text: str) -> str:
    """Lowercase, unify Arabic letter variants and strip Latin accents for matching."""
    text = _NORM.normalize_arabic(text).lower()
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


# ---------------------------------------------------------------- months / dates
_MONTH_NAMES: dict[int, list[str]] = {
    1: ["janvier", "january", "jan", "يناير", "جانفي", "كانون الثاني"],
    2: ["fevrier", "february", "feb", "فبراير", "فيفري", "شباط"],
    3: ["mars", "march", "مارس", "اذار"],
    4: ["avril", "april", "ابريل", "افريل", "نيسان"],
    5: ["mai", "may", "مايو", "ماي", "ايار"],
    6: ["juin", "june", "jouan", "jwan", "يونيو", "جوان", "حزيران"],
    7: ["juillet", "july", "juillet", "يوليو", "جويلية", "تموز"],
    8: ["aout", "august", "اغسطس", "اوت", "اب"],
    9: ["septembre", "september", "sept", "سبتمبر", "ايلول"],
    10: ["octobre", "october", "اكتوبر", "تشرين الاول"],
    11: ["novembre", "november", "نوفمبر", "تشرين الثاني"],
    12: ["decembre", "december", "ديسمبر", "كانون الاول"],
}
_MONTH_LOOKUP = {_fold(n): m for m, names in _MONTH_NAMES.items() for n in names}
_MONTH_ALT = "|".join(sorted((re.escape(n) for n in _MONTH_LOOKUP), key=len, reverse=True))
_MONTH_RE = rf"(?<!\w){_AR_PREFIX}(?P<month>{_MONTH_ALT})(?!\w)"
_DAY = r"(?P<d>\d{1,2})(?:er|st|nd|rd|th)?"

_RANGE_RE = re.compile(
    rf"(?<!\w)(?:du|from|de)?\s*(?P<d1>\d{{1,2}})\s*(?:au|to|until|till|-|–|الى|ل)\s*(?P<d2>\d{{1,2}})\s+(?:de\s+|of\s+)?{_MONTH_RE}"
    rf"(?:\s+(?P<year>20\d{{2}}))?"
)
_ABS_NUMERIC_RE = re.compile(r"(?<!\d)(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<year>20\d{2})(?!\d)")
_ABS_RE = re.compile(rf"(?<!\w){_DAY}\s+(?:de\s+|of\s+)?{_MONTH_RE}(?:\s+(?P<year>20\d{{2}}))?")
_ABS_EN_RE = re.compile(rf"{_MONTH_RE}\s+{_DAY}(?!\d)(?:,?\s+(?P<year>20\d{{2}}))?")
_MONTH_YEAR_RE = re.compile(rf"{_MONTH_RE}(?:\s+(?P<year>20\d{{2}}))?")
_MAY_CONTEXT = re.compile(r"(?:\b(?:in|of|on|for|during|until|till|early|late|mid|since|from)|\d)\s+$")

_RELATIVE: list[tuple[str, re.Pattern[str]]] = [
    ("day_after_tomorrow", re.compile(r"apres[- ]?demain|day after tomorrow|بعد غد|بعد غدوة|baad ghodwa|ba3d ghodwa")),
    ("tomorrow", re.compile(rf"(?<!\w){_AR_PREFIX}(?:demain|tomorrow|غدا|غدوة|ghodwa|ghoudwa|ghodoua|bokra|bukra)(?!\w)")),
    ("today", re.compile(r"(?<!\w)(?:aujourd'?hui|today|اليوم|lyoum|el yom)(?!\w)")),
    ("next_week", re.compile(r"(?:la )?semaine prochaine|next week|الاسبوع الجاي|الاسبوع القادم|jemaa jeya|semaine jeya")),
    ("this_week", re.compile(r"cette semaine|this week|هذا الاسبوع|هالاسبوع|esbou3 hedha")),
    ("this_weekend", re.compile(r"week[- ]?end|ويك ?اند|ويكاند|نهاية الاسبوع")),
]
_SEASONS: list[tuple[str, re.Pattern[str]]] = [
    ("summer", re.compile(rf"(?<!\w){_AR_PREFIX}(?:ete|summer|sayf|(?:ال)?صيف)(?!\w)")),
    ("winter", re.compile(rf"(?<!\w){_AR_PREFIX}(?:hiver|winter|chita|(?:ال)?شتاء|(?:ال)?شتا)(?!\w)")),
    ("spring", re.compile(rf"(?<!\w){_AR_PREFIX}(?:printemps|spring|rabii|(?:ال)?ربيع)(?!\w)")),
    ("autumn", re.compile(rf"(?<!\w){_AR_PREFIX}(?:automne|autumn|kharif|(?:ال)?خريف)(?!\w)")),
]

# ---------------------------------------------------------------- travelers / budget
_NUM_WORDS = {"un": 1, "une": 1, "one": 1, "deux": 2, "two": 2, "trois": 3, "three": 3,
              "quatre": 4, "four": 4, "cinq": 5, "five": 5, "six": 6, "sept": 7, "seven": 7, "huit": 8, "eight": 8}
_NUM = r"(?P<n>\d{1,2}|" + "|".join(_NUM_WORDS) + r")"
_PERSON_WORDS = (r"personnes?|persons?|people|pax|adultes?|adults?|voyageurs?|travell?ers?|touristes?|guests?|"
                 r"اشخاص|شخص|اشخاص|نفر|أفراد|افراد")
_NOT_TRAVELERS = r"(?!\s*(?:nuits?|jours?|days?|nights?|euros?|eur|tnd|dt|dinars?|usd|€|\$|h\b|km\b))"
_TRAVELER_PATTERNS: list[tuple[re.Pattern[str], int | None]] = [
    (re.compile(rf"(?<!\w){_NUM}\s*(?:{_PERSON_WORDS})(?!\w)"), None),
    (re.compile(rf"(?:famille|family|عيلة|3a2ila)\s+(?:de|of|من)\s+{_NUM}(?!\w)"), None),
    (re.compile(rf"(?<!\w)(?:pour|for|لـ|ل)\s+{_NUM}(?!\w){_NOT_TRAVELERS}"), None),
    (re.compile(r"(?<!\w)(?:couple|en amoureux|honeymoon|lune de miel|زوج|جوج)(?!\w)"), 2),
    (re.compile(r"(?<!\w)(?:solo|seule?|alone|on my own|وحدي|wahdi|wehdi)(?!\w)"), 1),
]

_AMOUNT_RE = re.compile(
    r"(?<![\w.])(?P<amount>\d{1,3}(?:[ ,.]\d{3})+|\d+(?:[.,]\d+)?)\s*"
    r"(?P<cur>tnd|dt|dinars?|دينار|د\.ت|euros?|eur|€|usd|dollars?|\$)(?!\w)"
)
_AMOUNT_PREFIX_RE = re.compile(r"(?P<cur>€|\$)\s*(?P<amount>\d+(?:[.,]\d+)?)")
_QUALITATIVE_BUDGET = re.compile(
    r"(?<!\w)(pas cher(?:e|s|es)?|bon marche|economique|low[- ]?cost|cheap|petit budget|small budget|"
    r"low budget|budget serre|tight budget|rkhis|rkhiss|رخيص|luxe|luxury|luxueux|luxurious|haut de gamme|"
    r"milieu de gamme|mid[- ]?range|midrange)(?!\w)"
)
_BUDGET_CANON = {"bon marche": "bon marché", "economique": "économique", "budget serre": "budget serré"}


_CURRENCY_WORDS: dict[str, list[str]] = {
    "EUR": ["euro", "euros", "eur", "اورو", "يورو"],
    "USD": ["dollar", "dollars", "usd", "dolar", "دولار"],
    "GBP": ["livre sterling", "livres sterling", "pound", "pounds", "gbp", "جنيه استرليني"],
    "CHF": ["franc suisse", "francs suisses", "chf", "فرنك سويسري"],
    "TND": ["dinar tunisien", "dinars", "dinar", "tnd", "dt", "دينار"],
}
_CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP"}
_CURRENCY_LOOKUP = {_fold(w): code for code, words in _CURRENCY_WORDS.items() for w in words}
_CURRENCY_RE = re.compile(
    rf"(?<!\w)[وفبلك]?(?:ال)?(?P<w>{'|'.join(sorted((re.escape(w) for w in _CURRENCY_LOOKUP), key=len, reverse=True))})(?!\w)"
)


@dataclass(frozen=True)
class DateEntity:
    """A date mention. `normalized` is ISO when resolvable (YYYY-MM, YYYY-MM-DD, a/b range)."""

    raw: str
    normalized: str
    type: str  # month | relative | absolute | range | season
    month: int | None = None


@dataclass
class EntityResult:
    """Entities found in one message."""

    destinations: list[str] = field(default_factory=list)
    dates: list[DateEntity] = field(default_factory=list)
    travelers: int | None = None
    budget: str | None = None
    raw_entities: list[dict[str, Any]] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)  # ISO codes in order of appearance
    origin: str | None = None  # flights only: departure place (slug, or "raw:Name" for non-catalogue places)
    flight_destination: str | None = None  # flights only: arrival place, same representation


@dataclass(frozen=True)
class _Place:
    key: str  # catalogue slug, or "raw:Name" for places outside the Tunisian catalogue
    start: int
    end: int


# spaCy's multilingual NER sometimes tags request words as places ("météo" -> LOC): never treat them as such
_NOT_PLACES = frozenset({
    "meteo", "weather", "vol", "vols", "flight", "flights", "hotel", "hotels", "billet", "billets", "avion",
    "taux", "change", "visa", "budget", "plage", "voyage", "sejour", "destination", "temps", "euro", "euros",
    "dinar", "dinars", "dollar", "dollars", "trip", "travel", "tayyara",
})
_ORIGIN_WORDS = frozenset({"de", "du", "d'", "depuis", "from", "men", "mn", "min", "من"})
_DEST_WORDS = frozenset({"a", "au", "vers", "pour", "to", "for", "lel", "lil", "dans", "الي"})
_ARTICLES = frozenset({"la", "le", "l'", "the", "el", "ville", "city"})
_FLIGHT_WORDS = frozenset({"vol", "vols", "flight", "flights", "billet", "billets", "avion", "fly",
                           "tayyara", "tayara", "طياره", "طيران", "رحله"})
_WORD_TOKEN = re.compile(r"[\w']+")


class EntityExtractor:
    """Rule-based extractor with a spaCy fallback for unknown place names."""

    def __init__(
        self,
        gazetteer_path: str = "data/destinations_gazetteer.json",
        spacy_model: str = "xx_ent_wiki_sm",
        today: date | None = None,
    ) -> None:
        path = Path(gazetteer_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Gazetteer not found at {path}")
        self._today = today
        data = json.loads(path.read_text(encoding="utf-8"))
        self._lookup: dict[str, str] = {}
        self._keys: dict[str, str] = {}  # slug -> key reported in EntityResult.destinations
        for dest in data["destinations"]:
            slug = dest["slug"]
            # Entries with "tunisian": false are international cities, reported as "raw:<display>"
            self._keys[slug] = slug if dest.get("tunisian", True) else f"raw:{dest.get('display') or dest['names'][0].title()}"
            for name in dest["names"]:
                self._lookup[_fold(name)] = slug
        alternatives = "|".join(sorted((re.escape(v) for v in self._lookup), key=len, reverse=True))
        self._dest_re = re.compile(rf"(?<!\w){_AR_PREFIX}(?P<name>{alternatives})(?!\w)")
        self._nlp: Any = None
        try:
            import spacy

            self._nlp = spacy.load(spacy_model, disable=["tagger", "parser", "lemmatizer"])
        except Exception as exc:  # noqa: BLE001 - gazetteer-only mode is a valid degraded state
            logger.warning("spaCy model '%s' unavailable (%s). Run: python -m spacy download %s",
                           spacy_model, exc, spacy_model)

    # ------------------------------------------------------------------ public API
    def extract(self, text: str, language: str = "fr", intent: str | None = None) -> EntityResult:
        """Extract destinations, dates, travelers, budget and (for flights) the route.

        `intent` tells the extractor whether to read the places as a flight route; when it is
        None, flight words in the text ("vol", "flight", ...) trigger route parsing instead.
        """
        normalized = _NORM.normalize(text)
        if not normalized:
            return EntityResult()
        folded = _fold(normalized)
        places = self._find_places(folded)
        spacy_entities = self._spacy_entities(normalized)
        for ent in spacy_entities:
            if (
                ent["label"] in {"LOC", "GPE"}
                and _fold(ent["text"]) not in _NOT_PLACES
                and not any(ent["start"] < p.end and p.start < ent["end"] for p in places)
            ):
                key = f"raw:{ent['text']}"
                if _fold(ent["text"]) not in self._lookup and all(p.key != key for p in places):
                    start = max(folded.find(_fold(ent["text"])), 0)
                    places.append(_Place(key, start, start + len(ent["text"])))
        places.sort(key=lambda p: p.start)
        destinations = list(dict.fromkeys(p.key for p in places))
        has_flight_word = bool(set(_WORD_TOKEN.findall(folded)) & _FLIGHT_WORDS)
        is_flight = intent == "flight_inquiry" or (
            intent is None and (has_flight_word or self._has_explicit_route(folded, places))
        )
        origin, flight_destination = self._flight_route(folded, places) if is_flight else (None, None)
        return EntityResult(
            origin=origin,
            flight_destination=flight_destination,
            destinations=destinations,
            dates=self._extract_dates(normalized, folded),
            travelers=self._extract_travelers(folded),
            budget=self._extract_budget(folded),
            raw_entities=spacy_entities,
            currencies=self._extract_currencies(folded),
        )

    def extract_batch(
        self, texts: list[str], languages: list[str], intents: list[str | None] | None = None
    ) -> list[EntityResult]:
        """Extract entities for each (text, language[, intent]) triple."""
        if len(texts) != len(languages) or (intents is not None and len(intents) != len(texts)):
            raise ValueError("texts, languages and intents must have the same length")
        return [self.extract(t, lang, intents[i] if intents else None) for i, (t, lang) in enumerate(zip(texts, languages))]

    # ------------------------------------------------------------------ destinations
    def _find_places(self, folded: str) -> list[_Place]:
        return [
            _Place(self._keys[self._lookup[m.group("name")]], *m.span("name"))
            for m in self._dest_re.finditer(folded)
        ]

    @staticmethod
    def _role(folded: str, place: _Place) -> str | None:
        """'origin' / 'destination' from the word (or Arabic prefix) right before the place."""
        if place.start > 0 and folded[place.start - 1] == "ل":
            return "destination"  # attached Arabic "to": لجربة
        tokens = _WORD_TOKEN.findall(folded[: place.start])[-2:]
        if len(tokens) == 2 and tokens[1] in _ARTICLES:
            tokens = tokens[:1]
        if not tokens:
            return None
        if tokens[-1] in _ORIGIN_WORDS:
            return "origin"
        if tokens[-1] in _DEST_WORDS:
            return "destination"
        return None

    def _has_explicit_route(self, folded: str, places: list[_Place]) -> bool:
        """True for "de X à Y" style text, where both an origin and a destination marker are present."""
        return {self._role(folded, p) for p in places} >= {"origin", "destination"}

    def _flight_route(self, folded: str, places: list[_Place]) -> tuple[str | None, str | None]:
        """Departure and arrival from prepositions; with none, the first of two places departs
        and the last arrives ("vol Tunis Paris"), and a lone place is the destination."""
        keys = list(dict.fromkeys(p.key for p in places))
        roles: dict[str, str] = {}
        for place in places:
            role = self._role(folded, place)
            if role and place.key not in roles:
                roles[place.key] = role
        origin = next((k for k in keys if roles.get(k) == "origin"), None)
        destination = next((k for k in keys if roles.get(k) == "destination"), None)
        unmarked = [k for k in keys if k not in roles]
        if origin is None and destination is None:
            if len(unmarked) >= 2:
                origin, destination = unmarked[0], unmarked[-1]
            elif unmarked:
                destination = unmarked[0]
        elif origin is None and unmarked:
            origin = unmarked[0]
        elif destination is None and unmarked:
            destination = unmarked[0]
        return origin, destination

    def _spacy_entities(self, text: str) -> list[dict[str, Any]]:
        if self._nlp is None:
            return []
        try:
            return [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                    for e in self._nlp(text).ents]
        except Exception:  # noqa: BLE001
            logger.exception("spaCy failed on input")
            return []

    # ------------------------------------------------------------------ dates
    def _current(self) -> date:
        return self._today or date.today()

    def _infer_year(self, month: int, year: str | None) -> int:
        if year:
            return int(year)
        today = self._current()
        return today.year if month >= today.month else today.year + 1

    def _extract_dates(self, original: str, folded: str) -> list[DateEntity]:
        found: list[tuple[int, DateEntity]] = []
        used: list[tuple[int, int]] = []

        def free(span: tuple[int, int]) -> bool:
            return not any(span[0] < e and s < span[1] for s, e in used)

        def add(span: tuple[int, int], entity: DateEntity) -> None:
            used.append(span)
            if len(original) == len(folded):  # offsets line up: report the user's own spelling
                entity = DateEntity(original[span[0]:span[1]].strip(), entity.normalized, entity.type, entity.month)
            found.append((span[0], entity))

        for m in _RANGE_RE.finditer(folded):
            month = _MONTH_LOOKUP[m.group("month")]
            y = self._infer_year(month, m.group("year"))
            d1, d2 = int(m.group("d1")), int(m.group("d2"))
            if 1 <= d1 <= 31 and 1 <= d2 <= 31 and free(m.span()):
                add(m.span(), DateEntity(m.group().strip(), f"{y}-{month:02d}-{d1:02d}/{y}-{month:02d}-{d2:02d}", "range", month))
        for m in _ABS_NUMERIC_RE.finditer(folded):
            d, mo, y = int(m.group("d")), int(m.group("m")), int(m.group("year"))
            if 1 <= mo <= 12 and 1 <= d <= 31 and free(m.span()):
                add(m.span(), DateEntity(m.group(), f"{y}-{mo:02d}-{d:02d}", "absolute", mo))
        for regex in (_ABS_RE, _ABS_EN_RE):
            for m in regex.finditer(folded):
                month = _MONTH_LOOKUP[m.group("month")]
                d = int(m.group("d"))
                if not (1 <= d <= 31) or not free(m.span()) or not self._month_ok(folded, m, month):
                    continue
                y = self._infer_year(month, m.group("year"))
                add(m.span(), DateEntity(m.group().strip(), f"{y}-{month:02d}-{d:02d}", "absolute", month))
        for m in _MONTH_YEAR_RE.finditer(folded):
            month = _MONTH_LOOKUP[m.group("month")]
            if not free(m.span()) or not self._month_ok(folded, m, month):
                continue
            y = self._infer_year(month, m.group("year"))
            add(m.span(), DateEntity(m.group().strip(), f"{y}-{month:02d}", "month", month))
        for label, regex in _RELATIVE:
            for m in regex.finditer(folded):
                if free(m.span()):
                    add(m.span(), DateEntity(m.group().strip(), label, "relative"))
        for label, regex in _SEASONS:
            for m in regex.finditer(folded):
                if free(m.span()):
                    add(m.span(), DateEntity(m.group().strip(), label, "season"))
        return [e for _, e in sorted(found, key=lambda x: x[0])]

    @staticmethod
    def _month_ok(folded: str, match: re.Match[str], month: int) -> bool:
        """'may' is also an English modal verb: accept it only in a date-like context."""
        if match.group("month") != "may":
            return True
        return bool(_MAY_CONTEXT.search(folded[: match.start("month")])) or bool(re.match(r"\s*\d", folded[match.end():]))

    # ------------------------------------------------------------------ travelers
    def _extract_travelers(self, folded: str) -> int | None:
        best: tuple[int, int] | None = None
        for regex, fixed in _TRAVELER_PATTERNS:
            m = regex.search(folded)
            if not m:
                continue
            if fixed is not None:
                value = fixed
            else:
                raw = m.group("n")
                value = int(raw) if raw.isdigit() else _NUM_WORDS[raw]
            if 1 <= value <= 50 and (best is None or m.start() < best[0]):
                best = (m.start(), value)
        return best[1] if best else None

    # ------------------------------------------------------------------ currencies
    @staticmethod
    def _extract_currencies(folded: str) -> list[str]:
        found = [(m.start(), _CURRENCY_LOOKUP[m.group("w")]) for m in _CURRENCY_RE.finditer(folded)]
        found += [(i, code) for sym, code in _CURRENCY_SYMBOLS.items() if (i := folded.find(sym)) >= 0]
        codes: list[str] = []
        for _, code in sorted(found):
            if code not in codes:
                codes.append(code)
        return codes

    # ------------------------------------------------------------------ budget
    def _extract_budget(self, folded: str) -> str | None:
        m = _AMOUNT_RE.search(folded)
        if m:
            return re.sub(r"\s+", " ", m.group()).strip()
        m = _AMOUNT_PREFIX_RE.search(folded)
        if m:
            return re.sub(r"\s+", " ", m.group()).strip()
        m = _QUALITATIVE_BUDGET.search(folded)
        if m:
            term = re.sub(r"\s+", " ", m.group()).strip()
            return _BUDGET_CANON.get(term, term)
        return None


@lru_cache(maxsize=1)
def get_entity_extractor() -> EntityExtractor:
    """Process-wide extractor instance."""
    return EntityExtractor()
