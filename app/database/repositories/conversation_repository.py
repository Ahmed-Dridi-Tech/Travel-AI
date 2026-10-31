"""Conversation queries."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import Conversation, Message, RoleEnum
from app.database.repositories.base import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    """Data access for conversations."""

    def __init__(self) -> None:
        super().__init__(Conversation)

    async def get_by_session_id(self, db: AsyncSession, session_id: str) -> Conversation | None:
        """Return the most recent conversation of a client session, or None."""
        result = await db.execute(
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_with_messages(
        self, db: AsyncSession, conversation_id: uuid.UUID
    ) -> Conversation | None:
        """Return the conversation with `messages` eager-loaded in chronological order."""
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

    async def create_with_first_message(
        self, db: AsyncSession, session_id: str, language: str, content: str
    ) -> Conversation:
        """Create a conversation and its first user message in the caller's transaction."""
        conversation = Conversation(session_id=session_id, language=language)
        conversation.messages.append(Message(role=RoleEnum.USER, content=content))
        db.add(conversation)
        await db.flush()
        return await self.get_with_messages(db, conversation.id) or conversation
