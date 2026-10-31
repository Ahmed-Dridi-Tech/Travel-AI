"""Agency analytics endpoint. Protected by a shared key when ANALYTICS_API_KEY is set."""

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.database.session import get_db
from app.schemas.analytics import AnalyticsResponse
from app.services.analytics_service import analytics_service

router = APIRouter(tags=["analytics"])


def require_analytics_key(x_api_key: str | None = Header(default=None)) -> None:
    """If ANALYTICS_API_KEY is configured, the request must carry the same value in X-API-Key."""
    expected = get_settings().analytics_api_key
    if not expected:
        return  # open (local development)
    if not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.get("/analytics", response_model=AnalyticsResponse, dependencies=[Depends(require_analytics_key)])
async def get_analytics(
    days: int = Query(7, ge=1, le=90, description="Look-back window in days"),
    db: AsyncSession = Depends(get_db),
) -> AnalyticsResponse:
    """KPIs, daily volume, language/intent distribution, recent inquiries and conversations."""
    return await analytics_service.get_analytics(db, days)
