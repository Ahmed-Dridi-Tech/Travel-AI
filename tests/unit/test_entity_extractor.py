from datetime import date

import pytest

from app.nlp.entity_extractor import EntityExtractor

TODAY = date(2026, 10, 31)


@pytest.fixture(scope="module")
def ex() -> EntityExtractor:
    return EntityExtractor(today=TODAY)


def test_destination_and_month(ex: EntityExtractor) -> None:
    r = ex.extract("Je veux aller à Djerba en décembre")
    assert r.destinations == ["djerba"]
    assert [(d.type, d.month, d.normalized) for d in r.dates] == [("month", 12, "2026-12")]


def test_two_destinations(ex: EntityExtractor) -> None:
    assert ex.extract("Vol pour Tunis et Sousse").destinations == ["tunis", "sousse"]


def test_travelers_and_destination(ex: EntityExtractor) -> None:
    r = ex.extract("Hôtel pour 2 personnes à Hammamet")
    assert r.destinations == ["hammamet"] and r.travelers == 2


@pytest.mark.parametrize("text", ["jerba", "jreba", "nheb nruh ljreba", "جربة", "نحب نسافر لجربة", "Djerba"])
def test_djerba_variants(ex: EntityExtractor, text: str) -> None:
    assert ex.extract(text).destinations == ["djerba"]


def test_no_duplicate_slugs(ex: EntityExtractor) -> None:
    assert ex.extract("Djerba, jerba et Djerba").destinations == ["djerba"]


def test_budget_amount(ex: EntityExtractor) -> None:
    assert ex.extract("budget 500 TND").budget == "500 tnd"
    assert ex.extract("max 300€").budget == "300€"
    assert ex.extract("200 euros").budget == "200 euros"


def test_budget_qualitative(ex: EntityExtractor) -> None:
    assert ex.extract("hôtel pas cher").budget == "pas cher"
    assert ex.extract("a luxury resort").budget == "luxury"
    assert ex.extract("Quel budget pour une semaine ?").budget is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("en couple", 2), ("voyage solo", 1), ("famille de 4", 4), ("for two people", 2),
     ("une famille", None), ("pour 3 nuits à Sousse", None), ("pour 3", 3)],
)
def test_travelers(ex: EntityExtractor, text: str, expected: int | None) -> None:
    assert ex.extract(text).travelers == expected


@pytest.mark.parametrize(
    ("text", "raw", "normalized", "dtype"),
    [
        ("demain", "demain", "tomorrow", "relative"),
        ("ghodwa", "ghodwa", "tomorrow", "relative"),
        ("la semaine prochaine", "la semaine prochaine", "next_week", "relative"),
        ("météo cette semaine", "cette semaine", "this_week", "relative"),
        ("weather this week", "this week", "this_week", "relative"),
        ("ce week-end", "week-end", "this_weekend", "relative"),
        ("en été", "été", "summer", "season"),
        ("in winter", "winter", "winter", "season"),
        ("décembre 2026", "décembre 2026", "2026-12", "month"),
        ("du 5 au 10 janvier", "du 5 au 10 janvier", "2027-01-05/2027-01-10", "range"),
        ("le 15 mars 2027", "15 mars 2027", "2027-03-15", "absolute"),
        ("في ديسمبر", "ديسمبر", "2026-12", "month"),
        ("fi jouillet", None, None, None),
    ],
)
def test_dates(ex: EntityExtractor, text: str, raw: str | None, normalized: str | None, dtype: str | None) -> None:
    dates = ex.extract(text).dates
    if raw is None:
        assert dates == [] or dates[0].type == "month"  # unknown spelling: tolerated either way
        return
    assert len(dates) == 1, dates
    assert (dates[0].raw, dates[0].normalized, dates[0].type) == (raw, normalized, dtype)


def test_modal_may_is_not_a_month(ex: EntityExtractor) -> None:
    assert ex.extract("may I book a hotel?").dates == []
    assert ex.extract("flights in may").dates[0].month == 5


def test_empty_text(ex: EntityExtractor) -> None:
    r = ex.extract("")
    assert (r.destinations, r.dates, r.travelers, r.budget, r.raw_entities) == ([], [], None, None, [])


def test_unknown_place_falls_back_to_spacy(ex: EntityExtractor) -> None:
    if ex._nlp is None:
        pytest.skip("spaCy model not installed")
    r = ex.extract("I want to visit Paris")
    assert "raw:Paris" in r.destinations


def test_batch_length_mismatch(ex: EntityExtractor) -> None:
    with pytest.raises(ValueError):
        ex.extract_batch(["a"], ["fr", "en"])
    assert len(ex.extract_batch(["Djerba", "Sousse"], ["fr", "fr"])) == 2


@pytest.mark.parametrize(
    ("text", "codes"),
    [
        ("Combien vaut 1 euro en dinar ?", ["EUR", "TND"]),
        ("convertir 500 dollars en dinars", ["USD", "TND"]),
        ("What's the exchange rate for USD to TND?", ["USD", "TND"]),
        ("قداش الأورو بالدينار؟", ["EUR", "TND"]),
        ("qaddesh el euro bel dinar", ["EUR", "TND"]),
        ("300€", ["EUR"]),
        ("météo Tunis", []),
    ],
)
def test_currencies(ex: EntityExtractor, text: str, codes: list[str]) -> None:
    assert ex.extract(text).currencies == codes


@pytest.mark.parametrize(
    ("text", "origin", "destination"),
    [
        ("vol Tunis Paris", "tunis", "raw:Paris"),
        ("de Tunis à Rome", "tunis", "raw:Rome"),
        ("nheb vol lel Paris", None, "raw:Paris"),
        ("vol retour Paris Tunis", "raw:Paris", "tunis"),
        ("I want a flight from Sfax to London", "sfax", "raw:Londres"),
        ("billet Paris depuis Monastir", "monastir", "raw:Paris"),
        ("vol pour Djerba", None, "djerba"),
        ("flight Tunis", None, "tunis"),
        ("طيارة من تونس لباريس", "tunis", "raw:Paris"),
        ("nheb vol men Tunis lel Istanbul", "tunis", "raw:Istanbul"),
    ],
)
def test_flight_route(ex: EntityExtractor, text: str, origin: str | None, destination: str | None) -> None:
    r = ex.extract(text)
    assert (r.origin, r.flight_destination) == (origin, destination)


def test_route_only_for_flight_context(ex: EntityExtractor) -> None:
    assert ex.extract("météo Paris").flight_destination is None
    assert ex.extract("hôtel à Rome").origin is None
    assert ex.extract("vol Tunis Paris", intent="hotel_inquiry").origin is None  # classifier says hotel
    assert ex.extract("Tunis Paris", intent="flight_inquiry").flight_destination == "raw:Paris"


def test_international_cities_are_reported_as_raw_places(ex: EntityExtractor) -> None:
    assert ex.extract("hôtel à Rome").destinations == ["raw:Rome"]
    assert ex.extract("Paris et Djerba").destinations == ["raw:Paris", "djerba"]
    assert ex.extract("I want to visit Paris").destinations == ["raw:Paris"]  # no duplicate from spaCy


def test_extract_batch_with_intents(ex: EntityExtractor) -> None:
    results = ex.extract_batch(["Tunis Paris", "Tunis Paris"], ["fr", "fr"], ["flight_inquiry", "weather"])
    assert results[0].origin == "tunis" and results[1].origin is None


@pytest.mark.parametrize("text", ["météo", "vol", "hôtel", "weather", "taux de change", "billet d'avion"])
def test_request_words_are_never_places(ex: EntityExtractor, text: str) -> None:
    assert ex.extract(text).destinations == []
