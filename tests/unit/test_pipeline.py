from typing import Any

import pytest

from app.nlp.intent_classifier import PROJECT_ROOT

_MODELS = PROJECT_ROOT / "data" / "models"
pytestmark = pytest.mark.skipif(
    not (_MODELS / "intent_classifier.pkl").exists() or not (_MODELS / "lid.176.bin").exists(),
    reason="run scripts/download_models.py and scripts/train_intent.py first",
)


@pytest.fixture(scope="module")
def pipeline():  # type: ignore[no-untyped-def]
    from app.nlp.pipeline import NLPPipeline

    return NLPPipeline()


def test_full_pipeline_french(pipeline) -> None:  # type: ignore[no-untyped-def]
    r = pipeline.analyze("Quel temps à Djerba en décembre?")
    assert (r.language, r.intent, r.destinations) == ("fr", "weather", ["djerba"])
    assert [d.month for d in r.dates] == [12]
    assert not r.is_out_of_domain and r.processing_time_ms > 0


def test_full_pipeline_arabizi(pipeline) -> None:  # type: ignore[no-untyped-def]
    r = pipeline.analyze("nheb nruh ljreba")
    assert r.language == "tun_lat" and r.is_tunisian
    assert r.intent in {"destination", "flight_inquiry"}
    assert r.destinations == ["djerba"]
    assert not r.is_out_of_domain


@pytest.mark.parametrize("text", ["recette couscous", "chkoun ghala el match?"])
def test_pipeline_ood(pipeline, text: str) -> None:  # type: ignore[no-untyped-def]
    assert pipeline.analyze(text).is_out_of_domain


def test_smalltalk_is_in_domain(pipeline) -> None:  # type: ignore[no-untyped-def]
    r = pipeline.analyze("bonjour")
    assert r.intent == "smalltalk" and not r.is_out_of_domain


def test_pipeline_error_handling(pipeline) -> None:  # type: ignore[no-untyped-def]
    from app.nlp.pipeline import NLPPipeline

    class Boom:
        def detect(self, _: str) -> Any:
            raise RuntimeError("fasttext exploded")

    broken = NLPPipeline(
        language_detector=Boom(),
        intent_classifier=pipeline.intent_classifier,
        ood_detector=pipeline.ood_detector,
        entity_extractor=pipeline.entity_extractor,
    )
    r = broken.analyze("météo Tunis")
    assert (r.language, r.intent, r.is_out_of_domain, r.ood_reason) == (
        "unknown", "out_of_domain", True, "pipeline_error",
    )


def test_empty_input_does_not_raise(pipeline) -> None:  # type: ignore[no-untyped-def]
    r = pipeline.analyze("")
    assert r.language == "unknown"
