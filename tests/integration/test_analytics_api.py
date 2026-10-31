"""Analytics endpoint over HTTP, backed by a rolled-back database session."""

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Conversation, Inquiry, InquiryTypeEnum, Message, RoleEnum
from app.database.session import get_db


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncIterator[httpx.AsyncClient]:
    from app.main import create_app

    app = create_app()

    async def override_db() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_db] = override_db
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_analytics_endpoint_returns_200(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/analytics?days=7")
    assert r.status_code == 200


async def test_analytics_structure(client: httpx.AsyncClient) -> None:
    body = (await client.get("/api/v1/analytics")).json()
    assert set(body) == {
        "period_days", "generated_at", "summary", "conversations_by_day", "language_distribution",
        "intent_distribution", "recent_inquiries", "recent_conversations",
    }
    assert set(body["summary"]) == {"total_conversations", "total_messages", "automation_rate", "avg_messages_per_conversation"}
    assert body["period_days"] == 7 and len(body["conversations_by_day"]) == 7


@pytest.mark.parametrize("days", [0, 91, "abc"])
async def test_analytics_rejects_bad_days(client: httpx.AsyncClient, days: object) -> None:
    assert (await client.get(f"/api/v1/analytics?days={days}")).status_code == 422


async def test_analytics_with_data(client: httpx.AsyncClient, db_session: AsyncSession) -> None:
    before = (await client.get("/api/v1/analytics")).json()["summary"]
    first = Conversation(session_id="api-a", language="tun_lat")
    second = Conversation(session_id="api-b", language="fr")
    db_session.add_all([first, second])
    await db_session.flush()
    db_session.add_all([
        Message(conversation_id=first.id, role=RoleEnum.USER, content="nheb hotel fi sousse", intent="hotel_inquiry"),
        Message(conversation_id=second.id, role=RoleEnum.USER, content="météo Djerba", intent="weather"),
        Inquiry(conversation_id=first.id, inquiry_type=InquiryTypeEnum.HOTEL, destination="Sousse",
                language="tun_lat", raw_message="nheb hotel fi sousse"),
    ])
    await db_session.flush()
    body = (await client.get("/api/v1/analytics")).json()
    assert body["summary"]["total_conversations"] == before["total_conversations"] + 2
    assert body["summary"]["total_messages"] == before["total_messages"] + 2
    assert {"api-a"[:8]} <= {c["session_id"] for c in body["recent_conversations"]}
    assert body["recent_inquiries"][0]["destination"] == "Sousse" and body["recent_inquiries"][0]["inquiry_type"] == "hotel"
    assert {"tun_lat", "fr"} <= {x["language"] for x in body["language_distribution"]}


async def test_analytics_open_when_no_key_configured(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import get_settings

    monkeypatch.delenv("ANALYTICS_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        assert (await client.get("/api/v1/analytics")).status_code == 200
    finally:
        get_settings.cache_clear()


async def test_analytics_requires_key_when_configured(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import get_settings

    monkeypatch.setenv("ANALYTICS_API_KEY", "k3y-for-tests")
    get_settings.cache_clear()
    try:
        assert (await client.get("/api/v1/analytics")).status_code == 401
        assert (await client.get("/api/v1/analytics", headers={"X-API-Key": "wrong"})).status_code == 401
        assert (await client.get("/api/v1/analytics", headers={"X-API-Key": "k3y-for-tests"})).status_code == 200
        assert (await client.get("/health")).status_code == 200  # only analytics is protected
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
