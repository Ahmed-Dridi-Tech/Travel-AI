import asyncio
import sys
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

# psycopg's async mode cannot run on Windows' default ProactorEventLoop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Session inside an outer transaction + SAVEPOINT; everything is rolled back after the test."""
    from app.core.config import get_settings

    engine = create_async_engine(
        get_settings().database_url, poolclass=NullPool, connect_args={"connect_timeout": 5}
    )
    try:
        conn = await engine.connect()
        await conn.execute(text("SELECT 1 FROM messages LIMIT 1"))
        await conn.rollback()
    except Exception as exc:  # noqa: BLE001
        await engine.dispose()
        pytest.skip(f"database/schema unavailable: {exc}")
    trans = await conn.begin()
    session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    try:
        yield session
    finally:
        await session.close()
        await trans.rollback()
        await conn.close()
        await engine.dispose()
