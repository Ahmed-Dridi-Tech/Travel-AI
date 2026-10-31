import pytest

from app.nlp.intent_classifier import PROJECT_ROOT, IntentClassifier

_MODELS = PROJECT_ROOT / "data" / "models"
pytestmark = pytest.mark.skipif(
    not all((_MODELS / f).exists() for f in ("intent_classifier.pkl", "intent_labels.pkl", "intent_config.json")),
    reason="run scripts/train_intent.py first",
)


@pytest.fixture(scope="module")
def clf() -> IntentClassifier:
    return IntentClassifier()


def test_obvious_weather(clf: IntentClassifier) -> None:
    assert clf.classify("météo Djerba").intent == "weather"


def test_obvious_currency(clf: IntentClassifier) -> None:
    assert clf.classify("taux euro dinar").intent == "currency"


def test_out_of_domain(clf: IntentClassifier) -> None:
    assert clf.classify("recette couscous").is_out_of_domain


def test_confidence_range(clf: IntentClassifier) -> None:
    result = clf.classify("hôtel à Sousse")
    assert 0.0 <= result.confidence <= 1.0
    assert all(0.0 <= p <= 1.0 for p in result.all_scores.values())


def test_all_scores_sum(clf: IntentClassifier) -> None:
    result = clf.classify("vol Tunis Paris")
    assert sum(result.all_scores.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.intent == max(result.all_scores, key=result.all_scores.__getitem__)


def test_batch_matches_single(clf: IntentClassifier) -> None:
    texts = ["météo Tunis", "hotel fi hammamet"]
    assert [r.intent for r in clf.classify_batch(texts)] == [clf.classify(t).intent for t in texts]
    assert clf.classify_batch([]) == []


def test_missing_files_raise(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError, match="train_intent"):
        IntentClassifier(classifier_path=str(tmp_path / "x.pkl"))
