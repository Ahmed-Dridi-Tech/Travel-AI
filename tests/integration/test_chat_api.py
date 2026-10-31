"""Chat API against the real NLP stack and PostgreSQL (rolled back after each test)."""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.database.session import get_db
from app.nlp.intent_classifier import PROJECT_ROOT

_MODELS = PROJECT_ROOT / "data" / "models"
pytestmark = pytest.mark.skipif(
    not (_MODELS / "intent_classifier.pkl").exists() or not (_MODELS / "lid.176.bin").exists(),
    reason="run scripts/download_models.py and scripts/train_intent.py first",
)


@pytest.fixture(scope="module")
def nlp_pipeline() -> Any:
    from app.nlp.pipeline import get_nlp_pipeline

    return get_nlp_pipeline()


@pytest_asyncio.fixture
async def client(nlp_pipeline: Any) -> AsyncIterator[httpx.AsyncClient]:
    from app.agent.orchestrator import AgentOrchestrator
    from app.agent.tool_registry import ToolRegistry
    from app.api.routes.chat import get_orchestrator
    from app.integrations.llm.mock import MockLLMProvider
    from app.main import create_app

    engine = create_async_engine(
        get_settings().database_url, poolclass=NullPool, connect_args={"connect_timeout": 5}
    )
    try:
        conn = await engine.connect()
        await conn.execute(text("SELECT 1 FROM messages LIMIT 1"))
        await conn.rollback()
    except Exception as exc:  # noqa: BLE001
        await engine.dispose()
        pytest.skip(f"database/schema unavailable: {exc}")
    trans = await conn.begin()
    session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)

    app = create_app()

    async def override_db() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_db] = override_db
    orchestrator = AgentOrchestrator(
        nlp=nlp_pipeline,
        tool_registry=ToolRegistry(weather_provider=lambda: None, exchange_provider=lambda: None),
        llm=MockLLMProvider(),
    )
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        await session.close()
        await trans.rollback()
        await conn.close()
        await engine.dispose()


def _sid() -> str:
    return f"test-{uuid.uuid4()}"


async def _chat(client: httpx.AsyncClient, message: str, sid: str | None = None) -> httpx.Response:
    return await client.post("/api/v1/chat", json={"session_id": sid or _sid(), "message": message})


async def test_chat_french(client: httpx.AsyncClient) -> None:
    r = await _chat(client, "météo Djerba décembre")
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "fr" and body["intent"] == "weather"
    assert body["destinations"] == ["djerba"] and not body["is_out_of_domain"]
    assert body["tool_used"] == "get_weather" and body["tool_success"] and body["llm_tokens_used"] > 0
    assert "Djerba" in body["response"] and "°C" in body["response"]


async def test_chat_arabizi(client: httpx.AsyncClient) -> None:
    r = await _chat(client, "nheb naaref el jaw fi Djerba")
    body = r.json()
    assert r.status_code == 200 and body["language"] == "tun_lat" and body["is_tunisian"] is True


async def test_chat_ood(client: httpx.AsyncClient) -> None:
    r = await _chat(client, "recette de couscous")
    body = r.json()
    assert r.status_code == 200 and body["is_out_of_domain"] is True
    assert "voyage" in body["response"] and body["tool_used"] == "none" and body["llm_tokens_used"] == 0


async def test_chat_english_and_arabic_replies(client: httpx.AsyncClient) -> None:
    en = (await _chat(client, "Do you know any nice recipe for pizza?")).json()
    assert en["language"] == "en" and en["is_out_of_domain"] and "specialise in travel" in en["response"]
    ar = (await _chat(client, "اكتب لي قصيدة عن الحب")).json()
    assert ar["language"] == "ar" and ar["is_out_of_domain"] and "متخصص" in ar["response"]


async def test_chat_currency_and_faq_and_destination(client: httpx.AsyncClient) -> None:
    fx = (await _chat(client, "Combien vaut 1 euro en dinar ?")).json()
    assert fx["tool_used"] == "get_exchange_rate" and "EUR" in fx["response"] and "TND" in fx["response"]
    faq = (await _chat(client, "Faut-il un visa ?")).json()
    assert faq["tool_used"] == "get_faq_answer" and "visa" in faq["response"].lower()
    dest = (await _chat(client, "Que visiter à Tozeur ?")).json()
    assert dest["tool_used"] == "get_destination_info" and "Tozeur" in dest["response"]


async def test_chat_history(client: httpx.AsyncClient) -> None:
    sid = _sid()
    await _chat(client, "météo Tunis", sid)
    await _chat(client, "bonjour", sid)
    r = await client.get(f"/api/v1/chat/{sid}/history")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "météo Tunis" and msgs[0]["intent"] == "weather"
    assert msgs[2]["content"] == "bonjour"


async def test_history_unknown_session_is_empty(client: httpx.AsyncClient) -> None:
    r = await client.get(f"/api/v1/chat/{_sid()}/history")
    assert r.status_code == 200 and r.json()["messages"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s", "message": ""},
        {"session_id": "s", "message": "   "},
        {"session_id": "s", "message": "x" * 1001},
        {"session_id": "", "message": "hello"},
        {"message": "hello"},
    ],
)
async def test_chat_invalid(client: httpx.AsyncClient, payload: dict[str, str]) -> None:
    assert (await client.post("/api/v1/chat", json=payload)).status_code == 422
