"""Agency analytics: aggregate queries over conversations, messages and inquiries."""

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import Date, cast, distinct, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Conversation, Inquiry, Message, RoleEnum
from app.schemas.analytics import (
    AnalyticsResponse,
    ConversationRow,
    DayCount,
    InquiryRow,
    IntentCount,
    LanguageCount,
    Summary,
)

TUNISIA_TZ = timezone(timedelta(hours=1))  # no DST since 2008
TUNISIA_TZ_NAME = "Africa/Tunis"
RECENT_LIMIT = 10
SESSION_ID_CHARS = 8
FIRST_MESSAGE_CHARS = 50


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class AnalyticsService:
    """Read-only reporting queries. `now` is injectable so tests can use an isolated time window."""

    async def get_analytics(self, db: AsyncSession, days: int = 7, now: datetime | None = None) -> AnalyticsResponse:
        """Build the full dashboard payload for the last `days` days."""
        now = now or datetime.now(timezone.utc)
        since = now - timedelta(days=days)

        total_conversations = await self._scalar(
            db, select(func.count()).select_from(Conversation).where(Conversation.created_at >= since)
        )
        total_messages = await self._scalar(
            db, select(func.count()).select_from(Message).where(Message.created_at >= since)
        )
        active_conversations = await self._scalar(
            db, select(func.count(distinct(Message.conversation_id))).where(Message.created_at >= since)
        )
        with_inquiry = await self._scalar(
            db,
            select(func.count()).select_from(Conversation).where(
                Conversation.created_at >= since,
                exists().where(Inquiry.conversation_id == Conversation.id),
            ),
        )
        automation = (total_conversations - with_inquiry) / total_conversations if total_conversations else 0.0
        average = total_messages / active_conversations if active_conversations else 0.0

        return AnalyticsResponse(
            period_days=days,
            generated_at=now,
            summary=Summary(
                total_conversations=total_conversations,
                total_messages=total_messages,
                automation_rate=round(automation, 2),
                avg_messages_per_conversation=round(average, 1),
            ),
            conversations_by_day=await self._by_day(db, since, now, days),
            language_distribution=await self._languages(db, since),
            intent_distribution=await self._intents(db, since),
            recent_inquiries=await self._recent_inquiries(db),
            recent_conversations=await self._recent_conversations(db),
        )

    @staticmethod
    async def _scalar(db: AsyncSession, stmt) -> int:  # type: ignore[no-untyped-def]
        return int((await db.execute(stmt)).scalar_one() or 0)

    @staticmethod
    async def _by_day(db: AsyncSession, since: datetime, now: datetime, days: int) -> list[DayCount]:
        local_day = cast(func.date_trunc("day", func.timezone(TUNISIA_TZ_NAME, Conversation.created_at)), Date)
        rows = (
            await db.execute(
                select(local_day, func.count()).where(Conversation.created_at >= since).group_by(local_day)
            )
        ).all()
        counts = {row[0]: row[1] for row in rows}
        today: date = now.astimezone(TUNISIA_TZ).date()
        return [  # zero-filled so the chart has one bar per day
            DayCount(date=(d := today - timedelta(days=offset)).isoformat(), count=counts.get(d, 0))
            for offset in range(days - 1, -1, -1)
        ]

    @staticmethod
    async def _languages(db: AsyncSession, since: datetime) -> list[LanguageCount]:
        count = func.count()
        rows = (
            await db.execute(
                select(Conversation.language, count)
                .where(Conversation.created_at >= since)
                .group_by(Conversation.language)
                .order_by(count.desc(), Conversation.language)
            )
        ).all()
        return [LanguageCount(language=lang, count=n) for lang, n in rows]

    @staticmethod
    async def _intents(db: AsyncSession, since: datetime) -> list[IntentCount]:
        count = func.count()
        rows = (
            await db.execute(
                select(Message.intent, count)
                .where(Message.created_at >= since, Message.role == RoleEnum.USER, Message.intent.is_not(None))
                .group_by(Message.intent)
                .order_by(count.desc(), Message.intent)
            )
        ).all()
        return [IntentCount(intent=intent, count=n) for intent, n in rows]

    @staticmethod
    async def _recent_inquiries(db: AsyncSession) -> list[InquiryRow]:
        rows = (await db.execute(select(Inquiry).order_by(Inquiry.created_at.desc()).limit(RECENT_LIMIT))).scalars()
        return [
            InquiryRow(
                id=str(i.id), inquiry_type=i.inquiry_type.value, destination=i.destination,
                raw_message=_truncate(i.raw_message, 80), created_at=i.created_at, status=i.status.value,
            )
            for i in rows
        ]

    @staticmethod
    async def _recent_conversations(db: AsyncSession) -> list[ConversationRow]:
        message_count = (
            select(func.count()).where(Message.conversation_id == Conversation.id).correlate(Conversation).scalar_subquery()
        )
        first_message = (
            select(Message.content)
            .where(Message.conversation_id == Conversation.id, Message.role == RoleEnum.USER)
            .order_by(Message.created_at)
            .limit(1)
            .correlate(Conversation)
            .scalar_subquery()
        )
        rows = (
            await db.execute(
                select(Conversation.session_id, Conversation.language, message_count, first_message, Conversation.created_at)
                .order_by(Conversation.created_at.desc())
                .limit(RECENT_LIMIT)
            )
        ).all()
        return [
            ConversationRow(
                session_id=sid[:SESSION_ID_CHARS], language=lang, message_count=int(n or 0),
                first_message=_truncate(first or "", FIRST_MESSAGE_CHARS), created_at=created,
            )
            for sid, lang, n, first, created in rows
        ]


analytics_service = AnalyticsService()
