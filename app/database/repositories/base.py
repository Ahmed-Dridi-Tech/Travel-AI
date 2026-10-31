"""Generic async repository. Sessions are always passed in; commit is the caller's job."""

import uuid
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.base import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """CRUD operations for one model. Methods flush (so ids/defaults exist) but never commit."""

    def __init__(self, model: type[T]) -> None:
        self.model = model

    async def get_by_id(self, db: AsyncSession, id_: uuid.UUID) -> T | None:
        """Return the row with this primary key, or None."""
        return await db.get(self.model, id_)

    async def create(self, db: AsyncSession, **values: Any) -> T:
        """Insert a new row and return it with defaults populated."""
        obj = self.model(**values)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def update(self, db: AsyncSession, obj: T, **values: Any) -> T:
        """Set the given attributes on obj; unknown attribute names raise AttributeError."""
        for key, value in values.items():
            if not hasattr(obj, key):
                raise AttributeError(f"{self.model.__name__} has no attribute '{key}'")
            setattr(obj, key, value)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def delete(self, db: AsyncSession, obj: T) -> None:
        """Delete obj (DB-level cascades remove dependent rows)."""
        await db.delete(obj)
        await db.flush()

    async def list(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> Sequence[T]:
        """Return rows ordered by primary key, paginated."""
        result = await db.execute(
            select(self.model).order_by(self.model.id).offset(skip).limit(limit)  # type: ignore[attr-defined]
        )
        return list(result.scalars().all())
