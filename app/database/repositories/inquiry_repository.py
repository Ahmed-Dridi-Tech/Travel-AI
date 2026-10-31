"""Inquiry queries."""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Inquiry, InquiryStatusEnum
from app.database.repositories.base import BaseRepository


class InquiryRepository(BaseRepository[Inquiry]):
    """Data access for flight/hotel/package inquiries."""

    def __init__(self) -> None:
        super().__init__(Inquiry)

    async def list_by_conversation(self, db: AsyncSession, conversation_id: uuid.UUID) -> Sequence[Inquiry]:
        """Inquiries recorded in one conversation, oldest first."""
        result = await db.execute(
            select(Inquiry).where(Inquiry.conversation_id == conversation_id).order_by(Inquiry.created_at)
        )
        return list(result.scalars().all())

    async def list_by_status(
        self, db: AsyncSession, status: InquiryStatusEnum, limit: int = 100
    ) -> Sequence[Inquiry]:
        """Most recent inquiries with the given status."""
        result = await db.execute(
            select(Inquiry).where(Inquiry.status == status).order_by(Inquiry.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
