"""FastAPI application entrypoint."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import get_agent_orchestrator
from app.api.routes.analytics import router as analytics_router
from app.api.routes.chat import router as chat_router
from app.core.config import get_settings
from app.database.session import engine, get_db, init_db
from app.integrations.exchange import get_exchange_service
from app.integrations.weather import get_weather_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging on startup; resource init (DB, NLP models) is added in later days."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.app_env)
    db_ok = await init_db()
    logger.info("Startup DB check: %s", "connected" if db_ok else "UNREACHABLE (running degraded)")
    try:
        await run_in_threadpool(get_agent_orchestrator)  # load models now, not on the first user request
    except Exception:  # noqa: BLE001 - missing models/keys must not stop the API from booting
        logger.exception("Agent failed to initialise; /api/v1/chat will fail until models/config are fixed")
    yield
    for getter in (get_weather_service, get_exchange_service):
        service = getter()
        if service is not None:
            await service.close()
    await engine.dispose()
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Application factory so tests can build isolated instances."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(analytics_router, prefix="/api/v1")

    @app.get("/health", tags=["system"])
    async def health(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
        """Liveness + DB probe. Always returns 200; 'degraded' signals an unreachable DB."""
        try:
            await db.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception as exc:  # noqa: BLE001 - /health must never raise
            logger.warning("Health check: DB unreachable (%s)", type(exc).__name__)
            db_status = "unreachable"
        return {
            "status": "ok" if db_status == "connected" else "degraded",
            "db": db_status,
            "app": settings.app_name,
            "version": settings.app_version,
            "env": settings.app_env,
        }

    return app


app = create_app()
