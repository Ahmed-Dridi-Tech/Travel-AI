"""Live OpenWeatherMap calls. Skipped without a working key (new keys can take ~2 h to activate)."""

import httpx
import pytest

from app.core.config import get_settings
from app.integrations.weather import CityNotFoundError, WeatherService

KEY = get_settings().openweather_api_key


def _key_works() -> bool:
    if not KEY:
        return False
    try:
        r = httpx.get("https://api.openweathermap.org/data/2.5/weather", params={"q": "Tunis,TN", "appid": KEY}, timeout=8)
    except httpx.HTTPError:
        return False
    return r.status_code == 200


pytestmark = pytest.mark.skipif(not _key_works(), reason="OPENWEATHER_API_KEY missing, invalid or not yet active")


async def test_real_weather_tunis() -> None:
    svc = WeatherService(KEY)
    try:
        r = await svc.get_current_weather("Tunis,TN")
    finally:
        await svc.close()
    assert r.country == "TN" and -10 < r.temperature_c < 55 and 0 <= r.humidity_percent <= 100 and r.description


async def test_real_forecast() -> None:
    svc = WeatherService(KEY)
    try:
        f = await svc.get_forecast("Djerba", days=3, lat=33.875, lon=10.8575)
    finally:
        await svc.close()
    assert 2 <= len(f.days) <= 4 and all(d.temp_min_c <= d.temp_max_c for d in f.days)


async def test_real_current_by_coordinates() -> None:
    svc = WeatherService(KEY)
    try:
        r = await svc.get_current_weather("Djerba", lat=33.875, lon=10.8575)
    finally:
        await svc.close()
    assert r.country == "TN" and -10 < r.temperature_c < 55


async def test_real_city_not_found() -> None:
    svc = WeatherService(KEY)
    try:
        with pytest.raises(CityNotFoundError):
            await svc.get_current_weather("XXINVALIDXX")
    finally:
        await svc.close()
