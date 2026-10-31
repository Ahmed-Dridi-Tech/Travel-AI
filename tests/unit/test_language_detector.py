from pathlib import Path

import pytest

from app.nlp.language_detector import PROJECT_ROOT, LanguageDetector

MODEL = PROJECT_ROOT / "data" / "models" / "lid.176.bin"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/download_models.py first")


@pytest.fixture(scope="module")
def detector() -> LanguageDetector:
    return LanguageDetector()


@pytest.mark.parametrize(
    ("text", "language", "tunisian"),
    [
        ("Je voudrais aller à Djerba en décembre", "fr", False),
        ("I want to book a hotel in Sousse", "en", False),
        ("أريد السفر إلى تونس", "ar", False),
        ("نحب نسافر لجربة", "tun_ar", True),
        ("nheb naaref el jaw fi Djerba", "tun_lat", True),
        ("météo Tunis", "fr", False),
        ("", "unknown", False),
    ],
)
def test_expected_languages(detector: LanguageDetector, text: str, language: str, tunisian: bool) -> None:
    result = detector.detect(text)
    assert result.language == language, result
    assert result.is_tunisian is tunisian
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("météo", "fr"),
        ("convertir 500 euros", "fr"),
        ("meteo Tunis", "fr"),
        ("weather", "en"),
        ("flight to Rome", "en"),
        ("how much", "en"),
        ("طقس", "ar"),
        ("نحب", "tun_ar"),
        ("nheb hotel fi sousse", "tun_lat"),
        ("salam", "tun_lat"),
        ("convert 500 euros", "en"),
    ],
)
def test_short_texts(detector: LanguageDetector, text: str, language: str) -> None:
    result = detector.detect(text)
    assert result.language == language, result
    assert result.confidence >= 0.5


@pytest.mark.parametrize("text", ["asdfgh", "xqz", "Ich möchte nach Tunesien reisen", "12345", "??"])
def test_never_unknown_for_non_empty_text(detector: LanguageDetector, text: str) -> None:
    from app.nlp.language_detector import MIN_RELIABLE_CONFIDENCE

    result = detector.detect(text)
    assert result.language in {"fr", "en", "ar", "tun_ar", "tun_lat"}
    if result.confidence < MIN_RELIABLE_CONFIDENCE:  # last-resort guesses are flagged by low confidence
        assert result.language in {"fr", "ar"}


def test_mixed_script_documented_behaviour(detector: LanguageDetector) -> None:
    """'مرحبا hello' has >30% letters in both scripts: script is 'mixed'; language is whatever
    fastText supports with confidence >= 0.5, otherwise 'unknown'. Both outcomes are acceptable."""
    result = detector.detect("مرحبا hello")
    assert result.script == "mixed"
    assert result.language in {"unknown", "ar", "en"}
    assert not result.is_tunisian


def test_batch_matches_single(detector: LanguageDetector) -> None:
    texts = ["Bonjour", "Hello there", ""]
    assert detector.detect_batch(texts) == [detector.detect(t) for t in texts]


def test_supported_languages(detector: LanguageDetector) -> None:
    assert detector.get_supported_languages() == ["fr", "en", "ar", "tun_ar", "tun_lat"]


def test_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_models"):
        LanguageDetector(str(tmp_path / "nope.bin"))
