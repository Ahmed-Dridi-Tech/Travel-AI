"""ExchangeRate-API integration (pair conversion rates) with an in-memory TTL cache."""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://v6.exchangerate-api.com/v6"
DEFAULT_CACHE_TTL = 3600
COMMON_PAIRS = [("EUR", "TND"), ("USD", "TND"), ("GBP", "TND"), ("CHF", "TND")]
_CODE_RE = re.compile(r"^[A-Z]{3}$")


class ExchangeServiceError(Exception):
    """Exchange-rate lookup failed (bad code, auth, network, provider error)."""


@dataclass(frozen=True)
class ExchangeResult:
    """Rate for 1 unit of from_currency expressed in to_currency."""

    from_currency: str
    to_currency: str
    rate: float
    last_updated: datetime


class ExchangeService:
    """Async ExchangeRate-API client. The API key sits in the URL path, so it is never logged
    and error messages carry only exception class names."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=timeout)
        self._cache: dict[str, tuple[float, ExchangeResult]] = {}
        self._cache_ttl = cache_ttl

    async def get_rate(self, from_currency: str, to_currency: str) -> ExchangeResult:
        """Current rate for the pair, cached for the TTL."""
        src, dst = from_currency.strip().upper(), to_currency.strip().upper()
        if not (_CODE_RE.match(src) and _CODE_RE.match(dst)):  # also blocks path injection
            raise ExchangeServiceError(f"invalid currency code: {from_currency!r}/{to_currency!r}")
        key = f"{src}:{dst}"
        entry = self._cache.get(key)
        if entry and time.monotonic() - entry[0] < self._cache_ttl:
            return entry[1]
        try:
            response = await self.client.get(f"{self.base_url}/{self.api_key}/pair/{src}/{dst}")
        except httpx.HTTPError as exc:
            raise ExchangeServiceError(f"network error: {type(exc).__name__}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExchangeServiceError(f"invalid response (HTTP {response.status_code})") from exc
        if response.status_code != 200 or payload.get("result") != "success":
            raise ExchangeServiceError(f"provider error: {payload.get('error-type', response.status_code)}")
        try:
            result = ExchangeResult(
                from_currency=src,
                to_currency=dst,
                rate=float(payload["conversion_rate"]),
                last_updated=datetime.fromtimestamp(int(payload["time_last_update_unix"]), tz=timezone.utc),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExchangeServiceError(f"unexpected response shape: {exc!r}") from exc
        self._cache[key] = (time.monotonic(), result)
        return result

    async def get_common_rates(self) -> list[ExchangeResult]:
        """EUR, USD, GBP and CHF against TND (cache keeps repeat calls cheap)."""
        return list(await asyncio.gather(*(self.get_rate(a, b) for a, b in COMMON_PAIRS)))

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()


@lru_cache(maxsize=1)
def get_exchange_service() -> ExchangeService | None:
    """Shared service, or None when EXCHANGE_RATE_API_KEY is not configured."""
    settings = get_settings()
    if not settings.exchange_rate_api_key:
        return None
    return ExchangeService(
        settings.exchange_rate_api_key,
        base_url=settings.exchangerate_base_url,
        timeout=settings.http_timeout_seconds,
    )
