"""Analytics against the real schema inside a rolled-back transaction.

Tests use a fixed `now` in 2100 so the window never overlaps real data in the dev database.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Conversation, Inquiry, InquiryTypeEnum, Message, RoleEnum
from app.services.analytics_service import AnalyticsService

NOW = datetime(2100, 6, 15, 12, 0, tzinfo=timezone.utc)
service = AnalyticsService()


async def add_conversation(
    db: AsyncSession, sid: str, language: str = "fr", age_days: float = 1, messages: tuple[tuple[RoleEnum, str, str | None], ...] = (),
    inquiry: bool = False,
) -> Conversation:
    created = NOW - timedelta(days=age_days)
    conv = Conversation(session_id=sid, language=language, created_at=created, updated_at=created)
    db.add(conv)
    await db.flush()
    for i, (role, content, intent) in enumerate(messages):
        db.add(Message(conversation_id=conv.id, role=role, content=content, intent=intent, created_at=created + timedelta(seconds=i)))
    if inquiry:
        db.add(Inquiry(conversation_id=conv.id, inquiry_type=InquiryTypeEnum.FLIGHT, destination="Paris",
                       language=language, raw_message="vol Paris", created_at=created, updated_at=created))
    await db.flush()
    return conv


async def test_empty_database(db_session: AsyncSession) -> None:
    result = await service.get_analytics(db_session, days=7, now=NOW)
    s = result.summary
    assert (s.total_conversations, s.total_messages, s.automation_rate, s.avg_messages_per_conversation) == (0, 0, 0.0, 0.0)
    assert result.language_distribution == [] and result.intent_distribution == []
    assert len(result.conversations_by_day) == 7 and all(d.count == 0 for d in result.conversations_by_day)


async def test_automation_rate_calculation(db_session: AsyncSession) -> None:
    for sid, with_inquiry in (("a", False), ("b", False), ("c", True)):
        await add_conversation(db_session, sid, inquiry=with_inquiry)
    s = (await service.get_analytics(db_session, days=7, now=NOW)).summary
    assert s.total_conversations == 3 and s.automation_rate == 0.67


async def test_language_distribution(db_session: AsyncSession) -> None:
    for sid, lang in (("a", "fr"), ("b", "fr"), ("c", "tun_lat"), ("d", "en")):
        await add_conversation(db_session, sid, language=lang)
    dist = {r.language: r.count for r in (await service.get_analytics(db_session, days=7, now=NOW)).language_distribution}
    assert dist == {"fr": 2, "tun_lat": 1, "en": 1}


async def test_period_filter_and_daily_buckets(db_session: AsyncSession) -> None:
    await add_conversation(db_session, "recent", age_days=1)
    await add_conversation(db_session, "recent2", age_days=1)
    await add_conversation(db_session, "old", age_days=30)  # outside the 7-day window
    result = await service.get_analytics(db_session, days=7, now=NOW)
    assert result.summary.total_conversations == 2
    by_day = {d.date: d.count for d in result.conversations_by_day}
    assert by_day["2100-06-14"] == 2 and sum(by_day.values()) == 2
    assert list(by_day)[-1] == "2100-06-15"  # ends today, oldest first


async def test_messages_intents_and_recent_lists(db_session: AsyncSession) -> None:
    await add_conversation(
        db_session, "abcdefghijkl", language="fr", age_days=0.5,
        messages=((RoleEnum.USER, "météo Djerba " + "x" * 80, "weather"), (RoleEnum.ASSISTANT, "ok", "weather"),
                  (RoleEnum.USER, "vol Tunis Paris", "flight_inquiry")),
        inquiry=True,
    )
    result = await service.get_analytics(db_session, days=7, now=NOW)
    assert result.summary.total_messages == 3 and result.summary.avg_messages_per_conversation == 3.0
    assert {i.intent: i.count for i in result.intent_distribution} == {"weather": 1, "flight_inquiry": 1}  # user turns only
    conv = result.recent_conversations[0]
    assert conv.session_id == "abcdefgh" and conv.message_count == 3
    assert len(conv.first_message) <= 50 and conv.first_message.startswith("météo Djerba")
    inquiry = result.recent_inquiries[0]
    assert (inquiry.inquiry_type, inquiry.destination, inquiry.status) == ("flight", "Paris", "pending")
    assert inquiry.raw_message == "vol Paris"
    assert result.summary.automation_rate == 0.0


@pytest.mark.parametrize("days", [1, 30])
async def test_daily_series_length(db_session: AsyncSession, days: int) -> None:
    assert len((await service.get_analytics(db_session, days=days, now=NOW)).conversations_by_day) == days
