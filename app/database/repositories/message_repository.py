"""Message queries."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Message, RoleEnum
from app.database.repositories.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """Data access for chat messages."""

    def __init__(self) -> None:
        super().__init__(Message)

    async def get_by_conversation(
        self, db: AsyncSession, conversation_id: uuid.UUID, limit: int = 20
    ) -> list[Message]:
        """Return the most recent `limit` messages in chronological order (oldest first)."""
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))

    async def add_message(
        self,
        db: AsyncSession,
        conversation_id: uuid.UUID,
        role: RoleEnum,
        content: str,
        intent: str | None = None,
        confidence: float | None = None,
    ) -> Message:
        """Append a message to a conversation."""
        return await self.create(
            db,
            conversation_id=conversation_id,
            role=role,
            content=content,
            intent=intent,
            confidence=confidence,
        )
