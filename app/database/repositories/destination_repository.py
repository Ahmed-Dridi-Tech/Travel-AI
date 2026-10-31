"""Destination queries."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from app.database.models import Destination
from app.database.repositories.base import BaseRepository

_NAME_COLUMN_BY_LANGUAGE: dict[str, InstrumentedAttribute[str]] = {
    "fr": Destination.name_fr,
    "tun_lat": Destination.name_fr,  # Arabizi has no stored names; Latin spellings follow French
    "en": Destination.name_en,
    "ar": Destination.name_ar,
    "tun_ar": Destination.name_ar,
}


class DestinationRepository(BaseRepository[Destination]):
    """Data access for the local destinations catalogue."""

    def __init__(self) -> None:
        super().__init__(Destination)

    async def get_by_slug(self, db: AsyncSession, slug: str) -> Destination | None:
        """Return the destination with this slug, or None."""
        result = await db.execute(select(Destination).where(Destination.slug == slug))
        return result.scalar_one_or_none()

    async def search_by_name(
        self, db: AsyncSession, query: str, language: str = "fr", active_only: bool = True
    ) -> list[Destination]:
        """Case-insensitive substring match on the name column matching `language`."""
        column = _NAME_COLUMN_BY_LANGUAGE.get(language, Destination.name_fr)
        stmt = select(Destination).where(column.icontains(query.strip(), autoescape=True))
        if active_only:
            stmt = stmt.where(Destination.active.is_(True))
        result = await db.execute(stmt.order_by(Destination.name_fr))
        return list(result.scalars().all())

    async def get_by_tags(
        self, db: AsyncSession, tags: list[str], active_only: bool = True
    ) -> list[Destination]:
        """Destinations sharing at least one tag (PostgreSQL array overlap, &&)."""
        if not tags:
            return []
        stmt = select(Destination).where(Destination.tags.overlap(tags))
        if active_only:
            stmt = stmt.where(Destination.active.is_(True))
        result = await db.execute(stmt.order_by(Destination.name_fr))
        return list(result.scalars().all())

    async def get_all_active(self, db: AsyncSession) -> list[Destination]:
        """All active destinations ordered by French name."""
        result = await db.execute(
            select(Destination).where(Destination.active.is_(True)).order_by(Destination.name_fr)
        )
        return list(result.scalars().all())
