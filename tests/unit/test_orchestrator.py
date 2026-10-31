"""Orchestrator with the real NLP stack + MockLLM against the (rolled-back) database."""

from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tool_registry import ToolRegistry
from app.database.repositories import inquiry_repo, message_repo
from app.integrations.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from app.integrations.llm.mock import MockLLMProvider
from app.nlp.intent_classifier import PROJECT_ROOT

_MODELS = PROJECT_ROOT / "data" / "models"
pytestmark = pytest.mark.skipif(
    not (_MODELS / "intent_classifier.pkl").exists() or not (_MODELS / "lid.176.bin").exists(),
    reason="run scripts/download_models.py and scripts/train_intent.py first",
)


class CountingLLM(MockLLMProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, messages: list[LLMMessage], *a: Any, **k: Any) -> LLMResponse:
        self.calls.append(messages[-1].content)
        return await super().complete(messages, *a, **k)


class FailingLLM(BaseLLMProvider):
    def get_model_name(self) -> str:
        return "failing"

    async def complete(self, *a: Any, **k: Any) -> LLMResponse:
        raise RuntimeError("provider down")


@pytest.fixture(scope="module")
def nlp_pipeline() -> Any:
    from app.nlp.pipeline import get_nlp_pipeline

    return get_nlp_pipeline()


def make(nlp_pipeline: Any, llm: BaseLLMProvider | None = None):  # type: ignore[no-untyped-def]
    from app.agent.orchestrator import AgentOrchestrator

    tools = ToolRegistry(weather_provider=lambda: None, exchange_provider=lambda: None)
    return AgentOrchestrator(nlp=nlp_pipeline, tool_registry=tools, llm=llm or MockLLMProvider())


async def test_orchestrator_weather(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    r = await make(nlp_pipeline).process("météo Djerba", "orch-weather", db_session)
    assert r.tool_used == "get_weather" and r.tool_success and r.intent == "weather"
    assert "Djerba" in r.response_text and "°C" in r.response_text and "Je cherche la météo" not in r.response_text
    assert r.destinations == ["djerba"] and r.llm_tokens_used > 0 and r.message_id


async def test_orchestrator_ood_skips_tool_and_llm(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    llm = CountingLLM()
    r = await make(nlp_pipeline, llm).process("recette de couscous", "orch-ood", db_session)
    assert r.is_out_of_domain and r.tool_used == "none" and r.llm_tokens_used == 0
    assert llm.calls == [] and "voyage" in r.response_text


async def test_orchestrator_llm_failure_returns_fallback(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    r = await make(nlp_pipeline, FailingLLM()).process("météo Djerba", "orch-fail", db_session)
    assert "difficulté technique" in r.response_text and r.llm_tokens_used == 0 and r.tool_used == "get_weather"
    en = await make(nlp_pipeline, FailingLLM()).process("What's the weather like in Sousse?", "orch-fail-en", db_session)
    assert "technical issue" in en.response_text


async def test_orchestrator_persists_both_turns(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    from app.database.repositories import conversation_repo

    orch = make(nlp_pipeline)
    await orch.process("météo Tunis", "orch-persist", db_session)
    await orch.process("bonjour", "orch-persist", db_session)
    conv = await conversation_repo.get_by_session_id(db_session, "orch-persist")
    msgs = await message_repo.get_by_conversation(db_session, conv.id)  # type: ignore[union-attr]
    assert [m.role.value for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0].intent == "weather" and msgs[2].intent == "smalltalk"


async def test_orchestrator_reuses_previous_destination(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    orch = make(nlp_pipeline)
    await orch.process("météo Djerba", "orch-implicit", db_session)
    r = await orch.process("et la météo ?", "orch-implicit", db_session)
    assert r.intent == "weather" and r.destinations == ["djerba"] and r.tool_success


async def test_orchestrator_prompt_contains_history_and_no_secrets(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    llm = CountingLLM()
    orch = make(nlp_pipeline, llm)
    await orch.process("météo Djerba", "orch-prompt", db_session)
    await orch.process("taux euro dinar", "orch-prompt", db_session)
    assert "Recent conversation:" in llm.calls[1] and "météo Djerba" in llm.calls[1]
    assert len(llm.calls[1]) < 3200  # ~800 tokens


async def test_orchestrator_flight_inquiry_is_stored(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    from app.database.repositories import conversation_repo

    orch = make(nlp_pipeline)
    r = await orch.process("je cherche un vol pour Londres pour 2 personnes", "orch-flight", db_session)
    assert r.tool_used == "submit_flight_inquiry" and r.tool_success
    conv = await conversation_repo.get_by_session_id(db_session, "orch-flight")
    stored = await inquiry_repo.list_by_conversation(db_session, conv.id)  # type: ignore[union-attr]
    assert len(stored) == 1 and stored[0].travelers == 2 and "Londres" in (stored[0].destination or "")
    assert "réf" in r.response_text


async def test_orchestrator_destination_tool_reads_the_database(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    r = await make(nlp_pipeline).process("Que visiter à Tozeur ?", "orch-dest", db_session)
    assert r.tool_used == "get_destination_info" and r.tool_success
    assert "Tozeur" in r.response_text and "TND" in r.response_text


class StubNLP:
    """Returns a fixed analysis per message so language fallback can be tested deterministically."""

    def __init__(self, *results: tuple[str, float]) -> None:
        self.results = list(results)

    def analyze(self, text: str):  # type: ignore[no-untyped-def]
        from app.nlp.pipeline import NLPResult

        language, confidence = self.results.pop(0)
        return NLPResult(
            original_text=text, normalized_text=text, language=language, script="latin",
            language_confidence=confidence, is_tunisian=False, intent="currency", intent_confidence=0.9,
            is_out_of_domain=False, ood_reason="in_domain", currencies=["EUR", "TND"],
        )


async def test_second_message_with_unknown_language_keeps_session_language(
    nlp_pipeline: Any, db_session: AsyncSession
) -> None:
    from app.agent.orchestrator import AgentOrchestrator

    orch = make(nlp_pipeline)
    first = await orch.process("météo Djerba", "orch-lang-fr", db_session)
    second = await orch.process("convertir 500 euros", "orch-lang-fr", db_session)
    assert first.language == "fr" and second.language == "fr"

    stub = AgentOrchestrator(
        nlp=StubNLP(("en", 0.95), ("unknown", 0.2)),
        tool_registry=ToolRegistry(weather_provider=lambda: None, exchange_provider=lambda: None),
        llm=MockLLMProvider(),
    )
    await stub.process("What is the exchange rate?", "orch-lang-en", db_session)
    follow_up = await stub.process("500 euros", "orch-lang-en", db_session)
    assert follow_up.language == "en"  # session language reused, never 'unknown'


async def test_first_message_with_unknown_language_defaults_to_french(db_session: AsyncSession) -> None:
    from app.agent.orchestrator import AgentOrchestrator

    stub = AgentOrchestrator(
        nlp=StubNLP(("unknown", 0.1)),
        tool_registry=ToolRegistry(weather_provider=lambda: None, exchange_provider=lambda: None),
        llm=MockLLMProvider(),
    )
    r = await stub.process("500 euros", "orch-lang-first", db_session)
    assert r.language == "fr"


async def test_flight_route_is_stored_as_arrow(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    from app.database.repositories import conversation_repo

    r = await make(nlp_pipeline).process("vol Tunis Paris", "orch-route", db_session)
    assert r.tool_used == "submit_flight_inquiry"
    conv = await conversation_repo.get_by_session_id(db_session, "orch-route")
    stored = await inquiry_repo.list_by_conversation(db_session, conv.id)  # type: ignore[union-attr]
    assert stored[0].destination == "Tunis → Paris" and stored[0].raw_message == "vol Tunis Paris"
    assert "travelers" not in r.response_text  # field ids never leak into the customer-facing text


async def test_bare_weather_request_asks_for_the_destination(nlp_pipeline: Any, db_session: AsyncSession) -> None:
    r = await make(nlp_pipeline).process("météo", "orch-bare-weather", db_session)
    assert r.intent == "weather" and r.destinations == [] and not r.tool_success
    assert "quelle destination" in r.response_text.lower()
