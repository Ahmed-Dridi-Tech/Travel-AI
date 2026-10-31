"""Curated destination descriptions: shape, coverage and how the seed script merges them."""

import importlib.util
import json
from typing import Any

from app.nlp.intent_classifier import PROJECT_ROOT

DATA = PROJECT_ROOT / "data"
TOP_10 = ["djerba", "tunis", "sousse", "hammamet", "tozeur", "monastir", "tabarka", "kairouan", "sidi-bou-said", "douz"]


def load_seed_module() -> Any:
    spec = importlib.util.spec_from_file_location("seed_destinations", PROJECT_ROOT / "scripts" / "seed_destinations.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_top_ten_destinations_are_described_briefly() -> None:
    texts = json.loads((DATA / "destinations.json").read_text(encoding="utf-8"))
    assert set(TOP_10) <= set(texts)
    for slug in TOP_10:
        for key in ("description_fr", "description_en"):
            text = texts[slug][key]
            assert text.strip() and len(text) <= 300, (slug, key)  # they are shown in chat bubbles
            assert 1 <= text.count(". ") + 1 <= 3, (slug, key)


def test_every_description_belongs_to_a_tunisian_catalogue_entry() -> None:
    texts = json.loads((DATA / "destinations.json").read_text(encoding="utf-8"))
    gazetteer = json.loads((DATA / "destinations_gazetteer.json").read_text(encoding="utf-8"))["destinations"]
    tunisian = {e["slug"] for e in gazetteer if e.get("tunisian", True)}
    assert set(texts) <= tunisian


def test_gazetteer_has_25_tunisian_and_international_entries() -> None:
    entries = json.loads((DATA / "destinations_gazetteer.json").read_text(encoding="utf-8"))["destinations"]
    assert sum(1 for e in entries if e.get("tunisian", True)) == 25
    international = [e for e in entries if e.get("tunisian") is False]
    assert {"paris", "rome", "london", "madrid", "dubai", "istanbul", "barcelona"} <= {e["slug"] for e in international}
    assert all(e.get("display") for e in international)
    assert len({e["slug"] for e in entries}) == len(entries)


def test_to_row_merges_descriptions_only_when_present() -> None:
    seed = load_seed_module()
    entry = {"slug": "djerba", "names": ["djerba", "جربة"], "region": "Médenine", "tags": ["beach"],
             "best_months": [5], "avg_budget_tnd": 800}
    with_text = seed.to_row(entry, {"djerba": {"description_fr": "Texte.", "description_en": "Text."}})
    assert with_text["description_fr"] == "Texte." and with_text["description_en"] == "Text."
    without = seed.to_row(entry, {})
    assert "description_fr" not in without and "description_en" not in without  # existing DB text is never wiped
    assert without["name_fr"] == "Djerba" and without["name_ar"] == "جربة"
