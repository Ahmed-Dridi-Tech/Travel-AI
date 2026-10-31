from app.agent.context import ConversationContext, ContextStore
from app.nlp.pipeline import NLPResult


def nlp(language: str = "fr", confidence: float = 0.9, destinations: list[str] | None = None) -> NLPResult:
    return NLPResult(
        original_text="x", normalized_text="x", language=language, script="latin",
        language_confidence=confidence, is_tunisian=False, intent="weather", intent_confidence=0.9,
        is_out_of_domain=False, ood_reason="in_domain", destinations=destinations or [],
    )


def test_language_only_adopted_when_confident() -> None:
    ctx = ConversationContext("s")
    ctx.update_from_nlp(nlp("fr", 0.5))
    assert ctx.language is None
    ctx.update_from_nlp(nlp("en", 0.95))
    assert ctx.language == "en"
    ctx.update_from_nlp(nlp("unknown", 0.99))
    assert ctx.language == "en" and ctx.turn_count == 3


def test_destinations_keep_last_three_unique_most_recent_first() -> None:
    ctx = ConversationContext("s")
    for dests in (["djerba"], ["tunis", "sousse"], ["djerba"], ["tozeur"]):
        ctx.update_from_nlp(nlp(destinations=dests))
    assert ctx.last_destinations == ["tozeur", "djerba", "tunis"]
    assert ctx.get_implicit_destination() == "tozeur"
    assert ConversationContext("n").get_implicit_destination() is None


def test_store_is_lru_bounded() -> None:
    store = ContextStore(max_sessions=2)
    a = store.get("a")
    store.get("b")
    assert store.get("a") is a
    store.get("c")  # evicts "b", the least recently used
    assert len(store) == 2 and store.get("a") is a and store.get("b") is not None
