import pytest

from app.nlp.intent_classifier import PROJECT_ROOT

_MODELS = PROJECT_ROOT / "data" / "models"
pytestmark = pytest.mark.skipif(
    not (_MODELS / "intent_classifier.pkl").exists(), reason="run scripts/train_intent.py first"
)


@pytest.fixture(scope="module")
def stack():  # type: ignore[no-untyped-def]
    from app.nlp.intent_classifier import get_intent_classifier
    from app.nlp.ood_detector import OODDetector

    clf = get_intent_classifier()
    return clf, OODDetector(embedder=clf.embedder)


def _verdict(stack, text):  # type: ignore[no-untyped-def]
    clf, ood = stack
    return ood.is_out_of_domain(text, clf.classify(text).confidence)


@pytest.mark.parametrize("text", ["recette de couscous", "qui a gagné le match?", "write me a python function"])
def test_out_of_domain(stack, text) -> None:  # type: ignore[no-untyped-def]
    assert _verdict(stack, text).is_ood


@pytest.mark.parametrize("text", ["météo Djerba", "nheb nruh ljreba", "bonjour", "salam", "visa ?", "hotel fi hammamet"])
def test_in_domain(stack, text) -> None:  # type: ignore[no-untyped-def]
    result = _verdict(stack, text)
    assert not result.is_ood and result.reason == "in_domain"


def test_decision_rules_with_fixed_distance() -> None:
    from app.nlp.ood_detector import OODDetector

    class _Stub:
        def encode(self, texts, **_):  # type: ignore[no-untyped-def]
            import numpy as np

            return np.eye(len(texts), 20)[: len(texts)] + 0.0

    det = OODDetector(embedder=_Stub())
    det.distance = lambda text: 0.8  # type: ignore[method-assign]
    assert det.is_out_of_domain("x", 0.9).reason == "embedding_distance"
    assert det.is_out_of_domain("x", 0.3).reason == "combined"
    assert det.is_out_of_domain("x", 0.1).reason == "combined"
    det.distance = lambda text: 0.6  # type: ignore[method-assign]
    assert det.is_out_of_domain("x", 0.2).reason == "intent_confidence"
    assert not det.is_out_of_domain("x", 0.5).is_ood
    det.distance = lambda text: 0.3  # type: ignore[method-assign]
    assert not det.is_out_of_domain("x", 0.1).is_ood
