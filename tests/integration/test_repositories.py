"""Repository tests against the real PostgreSQL schema (run `alembic upgrade head` first).

Each test runs inside an outer transaction + SAVEPOINT that is rolled back, so nothing persists.
"""

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.database.models import RoleEnum
from app.database.repositories import conversation_repo, destination_repo, message_repo


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(
        get_settings().database_url, poolclass=NullPool, connect_args={"connect_timeout": 5}
    )
    try:
        conn = await engine.connect()
    except Exception as exc:  # noqa: BLE001
        await engine.dispose()
        pytest.skip(f"PostgreSQL unreachable: {exc}")
    try:
        await conn.execute(text("SELECT 1 FROM destinations LIMIT 1"))
    except Exception:
        await conn.close()
        await engine.dispose()
        pytest.skip("Schema missing: run `alembic upgrade head`")
    await conn.rollback()
    trans = await conn.begin()
    session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    try:
        yield session
    finally:
        await session.close()
        await trans.rollback()
        await conn.close()
        await engine.dispose()


async def test_conversation_create_and_get_by_session(db: AsyncSession) -> None:
    conv = await conversation_repo.create_with_first_message(db, "sess-1", "fr", "Bonjour")
    assert conv.id is not None and conv.language == "fr"
    assert [m.content for m in conv.messages] == ["Bonjour"]
    assert conv.messages[0].role == RoleEnum.USER

    found = await conversation_repo.get_by_session_id(db, "sess-1")
    assert found is not None and found.id == conv.id
    assert await conversation_repo.get_by_session_id(db, "missing") is None


async def test_get_with_messages_ordered(db: AsyncSession) -> None:
    conv = await conversation_repo.create_with_first_message(db, "sess-2", "en", "first")
    await message_repo.add_message(db, conv.id, RoleEnum.ASSISTANT, "second", "faq", 0.9)
    await message_repo.add_message(db, conv.id, RoleEnum.USER, "third")

    loaded = await conversation_repo.get_with_messages(db, conv.id)
    assert loaded is not None
    assert [m.content for m in loaded.messages] == ["first", "second", "third"]
    assert await conversation_repo.get_with_messages(db, uuid.uuid4()) is None


async def test_message_add_and_get_by_conversation_limit(db: AsyncSession) -> None:
    conv = await conversation_repo.create(db, session_id="sess-3", language="fr")
    for i in range(5):
        await message_repo.add_message(db, conv.id, RoleEnum.USER, f"m{i}")
    msg = await message_repo.add_message(db, conv.id, RoleEnum.ASSISTANT, "reply", "weather", 0.8)
    assert msg.intent == "weather" and msg.confidence == 0.8

    last3 = await message_repo.get_by_conversation(db, conv.id, limit=3)
    assert [m.content for m in last3] == ["m3", "m4", "reply"]
    assert len(await message_repo.get_by_conversation(db, conv.id)) == 6


async def test_destination_slug_search_tags_active(db: AsyncSession) -> None:
    djerba = await destination_repo.create(
        db,
        slug="djerba-test",
        name_fr="Djerba Test",
        name_ar="جربة اختبار",
        name_en="Djerba Island",
        region="Sud-Est",
        tags=["beach", "family"],
        best_months=[6, 7, 8],
        avg_budget_tnd=900,
    )
    await destination_repo.create(
        db,
        slug="tozeur-test",
        name_fr="Tozeur Test",
        name_ar="توزر",
        name_en="Tozeur",
        region="Sud-Ouest",
        tags=["desert"],
        best_months=[11, 12],
        active=False,
    )

    assert (await destination_repo.get_by_slug(db, "djerba-test")).id == djerba.id  # type: ignore[union-attr]
    assert await destination_repo.get_by_slug(db, "nope") is None

    assert [d.slug for d in await destination_repo.search_by_name(db, "djerba test")] == ["djerba-test"]
    assert [d.slug for d in await destination_repo.search_by_name(db, "island", "en")] == ["djerba-test"]
    assert [d.slug for d in await destination_repo.search_by_name(db, "جربة اختبار", "tun_ar")] == ["djerba-test"]
    assert await destination_repo.search_by_name(db, "tozeur test") == []  # inactive hidden
    assert len(await destination_repo.search_by_name(db, "tozeur test", active_only=False)) == 1
    assert await destination_repo.search_by_name(db, "%") == []  # wildcard is escaped

    assert "djerba-test" in [d.slug for d in await destination_repo.get_by_tags(db, ["family", "x"])]
    assert "tozeur-test" not in [d.slug for d in await destination_repo.get_by_tags(db, ["desert"])]  # inactive
    assert await destination_repo.get_by_tags(db, []) == []
    assert "tozeur-test" not in [d.slug for d in await destination_repo.get_all_active(db)]


async def test_base_update_delete_list(db: AsyncSession) -> None:
    conv = await conversation_repo.create(db, session_id="sess-4", language="fr")
    updated = await conversation_repo.update(db, conv, language="tun_lat")
    assert updated.language == "tun_lat" and updated.updated_at >= updated.created_at
    with pytest.raises(AttributeError):
        await conversation_repo.update(db, conv, bogus=1)
    assert conv.id in [c.id for c in await conversation_repo.list(db, skip=0, limit=1000)]
    await conversation_repo.delete(db, conv)
    assert await conversation_repo.get_by_id(db, conv.id) is None
