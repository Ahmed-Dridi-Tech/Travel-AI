"""Demo-data seeding against the real schema (rolled back), using the mock NLP so no models are needed."""

import importlib.util
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import AgentOrchestrator
from app.agent.tool_registry import ToolRegistry
from app.database.models import Conversation, Inquiry, InquiryTypeEnum, Message
from app.integrations.llm.mock import MockLLMProvider
from app.nlp.intent_classifier import PROJECT_ROOT
from app.nlp.mock_pipeline import MockNLPPipeline


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location("seed_demo_data", PROJECT_ROOT / "scripts" / "seed_demo_data.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(
        nlp=MockNLPPipeline(),
        tool_registry=ToolRegistry(weather_provider=lambda: None, exchange_provider=lambda: None),
        llm=MockLLMProvider(),
    )


async def demo_rows(db: AsyncSession) -> list[Conversation]:
    result = await db.execute(select(Conversation).where(Conversation.session_id.like("demo-%")).order_by(Conversation.created_at))
    return list(result.scalars().all())


async def test_seed_creates_five_conversations_and_two_inquiries(db_session: AsyncSession, orchestrator: AgentOrchestrator) -> None:
    seed = load_script()
    now = datetime.now(timezone.utc)
    report = await seed.seed_demo(db_session, orchestrator, now)
    assert [r["intent"] for r in report] == [r["expected_intent"] for r in report]
    assert [r["language"] for r in report] == ["fr", "tun_lat", "en", "tun_ar", "fr"]
    assert await seed.count_demo(db_session) == (5, 2)

    inquiries = (await db_session.execute(select(Inquiry).join(Conversation).where(Conversation.session_id.like("demo-%")))).scalars().all()
    assert sorted(i.inquiry_type for i in inquiries) == [InquiryTypeEnum.FLIGHT, InquiryTypeEnum.HOTEL]
    assert {i.destination for i in inquiries} == {"Tunis → Paris", "Sousse"}


async def test_seed_spreads_timestamps_over_three_days(db_session: AsyncSession, orchestrator: AgentOrchestrator) -> None:
    seed = load_script()
    now = datetime.now(timezone.utc)
    await seed.seed_demo(db_session, orchestrator, now)
    rows = await demo_rows(db_session)
    assert len(rows) == 5
    created = [c.created_at for c in rows]
    assert created == sorted(created) and len(set(created)) == 5
    assert now - timedelta(days=3) < created[0] and created[-1] < now
    messages = (await db_session.execute(select(Message).join(Conversation).where(Conversation.session_id.like("demo-%")))).scalars().all()
    assert all(now - timedelta(days=3) < m.created_at < now for m in messages) and len(messages) == 10


async def test_seed_is_idempotent_and_reset_removes_only_demo_data(db_session: AsyncSession, orchestrator: AgentOrchestrator) -> None:
    seed = load_script()
    other = Conversation(session_id="real-customer-session", language="fr")
    db_session.add(other)
    await db_session.flush()
    await seed.seed_demo(db_session, orchestrator)
    await seed.seed_demo(db_session, orchestrator)  # second run must not double the data
    assert await seed.count_demo(db_session) == (5, 2)
    assert await seed.reset_demo(db_session) == 5
    assert await seed.count_demo(db_session) == (0, 0)
    assert (await db_session.execute(select(Conversation).where(Conversation.session_id == "real-customer-session"))).scalar_one()
    assert await seed.reset_demo(db_session) == 0  # nothing left to remove
