"""MockNLPPipeline: the keyword-based stand-in used when USE_MOCK_NLP=true (lean deployment)."""

import pytest

from app.core.config import get_settings
from app.nlp import get_pipeline
from app.nlp.mock_pipeline import MockNLPPipeline


@pytest.fixture(scope="module")
def mock() -> MockNLPPipeline:
    return MockNLPPipeline()


# The six messages of docs/demo-script.md, in order
@pytest.mark.parametrize(
    ("text", "language", "intent", "destinations"),
    [
        ("Bonjour, je voudrais savoir la météo à Djerba cette semaine", "fr", "weather", ["djerba"]),
        ("كيفاش الطقس في سوسة ؟", "tun_ar", "weather", ["sousse"]),
        ("nheb naaref el jaw fi Tozeur", "tun_lat", "weather", ["tozeur"]),
        ("What's the exchange rate for euro to dinar?", "en", "currency", []),
        ("Je cherche un vol Tunis Paris pour 2 personnes", "fr", "flight_inquiry", ["tunis", "raw:Paris"]),
        ("hôtel pas cher à Hammamet pour le weekend", "fr", "hotel_inquiry", ["hammamet"]),
    ],
)
def test_demo_script_messages(mock: MockNLPPipeline, text: str, language: str, intent: str, destinations: list[str]) -> None:
    r = mock.analyze(text)
    assert (r.language, r.intent, r.destinations) == (language, intent, destinations)
    assert not r.is_out_of_domain and r.processing_time_ms >= 0


def test_demo_entities(mock: MockNLPPipeline) -> None:
    week = mock.analyze("Bonjour, je voudrais savoir la météo à Djerba cette semaine")
    assert [d.normalized for d in week.dates] == ["this_week"]
    assert mock.analyze("What's the exchange rate for euro to dinar?").currencies == ["EUR", "TND"]
    flight = mock.analyze("Je cherche un vol Tunis Paris pour 2 personnes")
    assert (flight.origin, flight.flight_destination, flight.travelers) == ("tunis", "raw:Paris", 2)
    hotel = mock.analyze("hôtel pas cher à Hammamet pour le weekend")
    assert hotel.budget == "pas cher" and [d.normalized for d in hotel.dates] == ["this_weekend"]


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("météo", "weather"), ("sarf euro tnd", "currency"), ("tayyara lel paris", "flight_inquiry"),
        ("logement Djerba pas cher", "hotel_inquiry"), ("faut-il un visa ?", "travel_faq"),
        ("salam", "smalltalk"), ("merci", "smalltalk"), ("Que visiter ?", "destination"), ("Tozeur", "destination"),
    ],
)
def test_keyword_rules(mock: MockNLPPipeline, text: str, intent: str) -> None:
    assert mock.analyze(text).intent == intent


def test_unmatched_text_is_out_of_domain(mock: MockNLPPipeline) -> None:
    r = mock.analyze("recette de couscous")
    assert r.is_out_of_domain and r.intent == "out_of_domain" and r.ood_reason == "mock_no_match"


def test_whole_words_only(mock: MockNLPPipeline) -> None:
    assert mock.analyze("une révolution volontaire").intent == "out_of_domain"  # 'vol' inside other words


def test_tunisian_flags(mock: MockNLPPipeline) -> None:
    assert mock.analyze("nheb vol Tunis Paris").is_tunisian and mock.analyze("nheb vol Tunis Paris").language == "tun_lat"
    assert not mock.analyze("flight to Rome please").is_tunisian


@pytest.mark.parametrize("text", ["", "   ", "?!", "x" * 5000, "​​"])
def test_never_raises(mock: MockNLPPipeline, text: str) -> None:
    r = mock.analyze(text)
    assert r.intent and r.language


def test_get_pipeline_honours_use_mock_nlp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_NLP", "true")
    get_settings.cache_clear()
    get_pipeline.cache_clear()
    try:
        assert isinstance(get_pipeline(), MockNLPPipeline)
        assert get_pipeline() is get_pipeline()
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
        get_pipeline.cache_clear()
