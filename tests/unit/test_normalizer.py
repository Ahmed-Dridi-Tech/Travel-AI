import pytest

from app.nlp.normalizer import TextNormalizer

n = TextNormalizer()


def test_normalize_whitespace_and_zero_width() -> None:
    assert n.normalize("  Bonjour \n\n  le\tmonde ​ ") == "Bonjour le monde"
    assert n.normalize("مر‍حبا‏") == "مرحبا"
    assert n.normalize("") == ""


def test_normalize_keeps_case_and_punctuation() -> None:
    assert n.normalize("Quel temps à Djerba ?") == "Quel temps à Djerba ?"


def test_normalize_nfc() -> None:
    assert n.normalize("é") == "é"


def test_arabic_diacritics_tatweel_removed() -> None:
    assert n.normalize_arabic("مَرْحَبًا") == "مرحبا"
    assert n.normalize_arabic("جمـــيل") == "جميل"


def test_arabic_variants_unified() -> None:
    assert n.normalize_arabic("أإآٱ") == "اااا"
    assert n.normalize_arabic("مدرسة") == "مدرسه"
    assert n.normalize_arabic("على") == "علي"


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("أريد السفر إلى تونس", "arabic"),
        ("Je voudrais aller à Djerba", "latin"),
        ("مرحبا hello", "mixed"),
        ("12345 ???", "mixed"),
        ("نحب نسافر pour Paris", "mixed"),
    ],
)
def test_detect_script(text: str, script: str) -> None:
    assert n.detect_script(text) == script


@pytest.mark.parametrize(
    "text",
    [
        "nheb naaref el jaw fi Djerba",
        "chnowa el jaw fi Tozeur",
        "famma barsha blayes fi tounes",
        "7aja behia",
        "nheb vol Tunis Paris",
        "9adech el euro bel dinar",
    ],
)
def test_is_arabizi_true(text: str) -> None:
    assert n.is_arabizi(text)


@pytest.mark.parametrize(
    "text",
    [
        "Je voudrais aller à Djerba en décembre",
        "I want to book a hotel in Sousse",
        "météo Tunis",
        "chambre au 3eme étage",
        "Ken wants a hotel in Paris",
        "نحب نسافر",
    ],
)
def test_is_arabizi_false(text: str) -> None:
    assert not n.is_arabizi(text)
