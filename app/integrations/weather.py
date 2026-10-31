"""OpenWeatherMap integration (current weather + short forecast) with an in-memory TTL cache."""

import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openweathermap.org/data/2.5"
DEFAULT_CACHE_TTL = 1800


class CityNotFoundError(Exception):
    """OpenWeatherMap does not know this city."""


class WeatherServiceError(Exception):
    """Any other weather failure. status_code is 0 for network errors."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class WeatherResult:
    """Current conditions."""

    city: str
    country: str
    temperature_c: float
    feels_like_c: float
    humidity_percent: int
    description: str
    wind_speed_ms: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(frozen=True)
class DayForecast:
    """Daily summary built from 3-hourly forecast slots."""

    date: str
    temp_min_c: float
    temp_max_c: float
    description: str


@dataclass(frozen=True)
class ForecastResult:
    """Multi-day forecast."""

    city: str
    days: list[DayForecast] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WeatherService:
    """Async OpenWeatherMap client. The API key is never logged or included in error messages."""

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
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl

    async def _get(
        self, path: str, city: str, lat: float | None, lon: float | None, extra: dict[str, Any] | None = None,
        lang: str = "fr",
    ) -> dict[str, Any]:
        where: dict[str, Any] = {"lat": lat, "lon": lon} if lat is not None and lon is not None else {"q": city}
        params = {**where, "appid": self.api_key, "units": "metric", "lang": lang, **(extra or {})}
        try:
            response = await self.client.get(f"{self.base_url}/{path}", params=params)
        except httpx.HTTPError as exc:
            raise WeatherServiceError(0, f"network error: {type(exc).__name__}") from exc
        if response.status_code == 404:
            raise CityNotFoundError(city)
        if response.status_code == 401:
            raise WeatherServiceError(401, "invalid or not-yet-activated API key")
        if response.status_code != 200:
            try:
                message = str(response.json().get("message", "unknown error"))
            except ValueError:
                message = "unknown error"
            raise WeatherServiceError(response.status_code, message)
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise WeatherServiceError(response.status_code, "invalid JSON from provider") from exc
        return payload

    def _cached(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry and time.monotonic() - entry[0] < self._cache_ttl:
            return entry[1]
        return None

    async def get_current_weather(
        self, city: str, lat: float | None = None, lon: float | None = None, lang: str = "fr"
    ) -> WeatherResult:
        """Current conditions for `city` (e.g. 'Tunis,TN'); coordinates, when given, take precedence
        (OpenWeather does not know every tourist name, e.g. 'Djerba,TN' returns 404)."""
        key = f"current:{city.lower()}:{lat}:{lon}:{lang}"
        if (hit := self._cached(key)) is not None:
            return hit  # type: ignore[no-any-return]
        data = await self._get("weather", city, lat, lon, lang=lang)
        try:
            result = WeatherResult(
                city=data["name"],
                country=data.get("sys", {}).get("country", ""),
                temperature_c=float(data["main"]["temp"]),
                feels_like_c=float(data["main"]["feels_like"]),
                humidity_percent=int(data["main"]["humidity"]),
                description=data["weather"][0]["description"],
                wind_speed_ms=float(data.get("wind", {}).get("speed", 0.0)),
                timestamp=datetime.fromtimestamp(data.get("dt", time.time()), tz=timezone.utc),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise WeatherServiceError(200, f"unexpected response shape: {exc!r}") from exc
        self._cache[key] = (time.monotonic(), result)
        return result

    async def get_forecast(
        self, city: str, days: int = 3, lat: float | None = None, lon: float | None = None, lang: str = "fr"
    ) -> ForecastResult:
        """Daily min/max and dominant condition for the next `days` days (max 5)."""
        days = max(1, min(days, 5))
        key = f"forecast:{city.lower()}:{days}:{lat}:{lon}:{lang}"
        if (hit := self._cached(key)) is not None:
            return hit  # type: ignore[no-any-return]
        data = await self._get("forecast", city, lat, lon, {"cnt": days * 8}, lang=lang)
        try:
            by_day: dict[str, list[dict[str, Any]]] = {}
            for slot in data["list"]:
                by_day.setdefault(slot["dt_txt"][:10], []).append(slot)
            summaries = []
            for day, slots in sorted(by_day.items())[:days]:
                condition = Counter(s["weather"][0]["description"] for s in slots).most_common(1)[0][0]
                summaries.append(
                    DayForecast(
                        date=day,
                        temp_min_c=min(float(s["main"]["temp_min"]) for s in slots),
                        temp_max_c=max(float(s["main"]["temp_max"]) for s in slots),
                        description=condition,
                    )
                )
            result = ForecastResult(city=data.get("city", {}).get("name", city), days=summaries)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise WeatherServiceError(200, f"unexpected response shape: {exc!r}") from exc
        self._cache[key] = (time.monotonic(), result)
        return result

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()


@lru_cache(maxsize=1)
def get_weather_service() -> WeatherService | None:
    """Shared service, or None when OPENWEATHER_API_KEY is not configured."""
    settings = get_settings()
    if not settings.openweather_api_key:
        return None
    return WeatherService(
        settings.openweather_api_key,
        base_url=f"{settings.openweather_base_url.rstrip('/')}/data/2.5",
        timeout=settings.http_timeout_seconds,
    )
