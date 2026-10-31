"""Weather/exchange services against httpx.MockTransport: caching, error mapping, no key leaks."""

import json

import httpx
import pytest

from app.integrations.exchange import ExchangeService, ExchangeServiceError
from app.integrations.weather import CityNotFoundError, WeatherService, WeatherServiceError

OWM_CURRENT = {
    "name": "Tunis", "sys": {"country": "TN"}, "dt": 1790000000,
    "main": {"temp": 21.4, "feels_like": 21.0, "humidity": 64},
    "weather": [{"description": "ciel dégagé"}], "wind": {"speed": 3.6},
}


def weather_service(handler, ttl: int = 1800) -> tuple[WeatherService, list[httpx.Request]]:  # type: ignore[no-untyped-def]
    calls: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return WeatherService("SECRET-KEY", client=client, cache_ttl=ttl), calls


async def test_weather_current_maps_response_and_caches() -> None:
    svc, calls = weather_service(lambda r: httpx.Response(200, json=OWM_CURRENT))
    first = await svc.get_current_weather("Tunis,TN")
    second = await svc.get_current_weather("tunis,tn")
    assert (first.city, first.country, first.temperature_c, first.humidity_percent) == ("Tunis", "TN", 21.4, 64)
    assert first is second and len(calls) == 1
    assert calls[0].url.params["units"] == "metric" and calls[0].url.params["lang"] == "fr"


async def test_weather_coordinates_replace_name_query() -> None:
    svc, calls = weather_service(lambda r: httpx.Response(200, json=OWM_CURRENT))
    await svc.get_current_weather("Djerba", lat=33.875, lon=10.8575)
    params = calls[0].url.params
    assert params["lat"] == "33.875" and params["lon"] == "10.8575" and "q" not in params


async def test_weather_cache_expires() -> None:
    svc, calls = weather_service(lambda r: httpx.Response(200, json=OWM_CURRENT), ttl=0)
    await svc.get_current_weather("Tunis")
    await svc.get_current_weather("Tunis")
    assert len(calls) == 2


async def test_weather_error_mapping() -> None:
    svc, _ = weather_service(lambda r: httpx.Response(404, json={"message": "city not found"}))
    with pytest.raises(CityNotFoundError):
        await svc.get_current_weather("XXINVALIDXX")
    svc, _ = weather_service(lambda r: httpx.Response(401, json={"message": "Invalid API key"}))
    with pytest.raises(WeatherServiceError) as exc:
        await svc.get_current_weather("Tunis")
    assert exc.value.status_code == 401 and "SECRET-KEY" not in str(exc.value)
    svc, _ = weather_service(lambda r: httpx.Response(500, json={"message": "boom"}))
    with pytest.raises(WeatherServiceError, match="500"):
        await svc.get_current_weather("Tunis")


async def test_weather_network_error_does_not_leak_key() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    svc, _ = weather_service(boom)
    with pytest.raises(WeatherServiceError) as exc:
        await svc.get_current_weather("Tunis")
    assert exc.value.status_code == 0 and "SECRET-KEY" not in str(exc.value)


async def test_weather_forecast_groups_by_day() -> None:
    slots = []
    for day, base in (("2026-11-05", 15.0), ("2026-11-06", 18.0)):
        for hour, desc in (("00", "nuageux"), ("03", "nuageux"), ("12", "ciel dégagé")):
            slots.append({"dt_txt": f"{day} {hour}:00:00", "main": {"temp_min": base, "temp_max": base + 4},
                          "weather": [{"description": desc}]})
    svc, calls = weather_service(lambda r: httpx.Response(200, json={"city": {"name": "Tunis"}, "list": slots}))
    result = await svc.get_forecast("Tunis", days=2)
    assert [d.date for d in result.days] == ["2026-11-05", "2026-11-06"]
    assert result.days[0].temp_min_c == 15.0 and result.days[1].temp_max_c == 22.0
    assert result.days[0].description == "nuageux"
    assert calls[0].url.params["cnt"] == "16"
    await svc.get_forecast("Tunis", days=2)
    assert len(calls) == 1


def exchange_service(handler, ttl: int = 3600) -> tuple[ExchangeService, list[httpx.Request]]:  # type: ignore[no-untyped-def]
    calls: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    return ExchangeService("SECRET-KEY", client=httpx.AsyncClient(transport=httpx.MockTransport(wrapped)), cache_ttl=ttl), calls


def ok(request: httpx.Request) -> httpx.Response:
    pair = request.url.path.rsplit("/", 2)
    rate = {"EUR": 3.4, "USD": 3.1, "GBP": 3.9, "CHF": 3.6}.get(pair[-2], 1.0)
    return httpx.Response(200, json={"result": "success", "conversion_rate": rate, "time_last_update_unix": 1790000000})


async def test_exchange_rate_and_cache() -> None:
    svc, calls = exchange_service(ok)
    first = await svc.get_rate("eur", "tnd")
    again = await svc.get_rate("EUR", "TND")
    assert (first.from_currency, first.to_currency, first.rate) == ("EUR", "TND", 3.4)
    assert first is again and len(calls) == 1
    assert calls[0].url.path.endswith("/pair/EUR/TND")


async def test_exchange_common_rates() -> None:
    svc, calls = exchange_service(ok)
    rates = await svc.get_common_rates()
    assert [r.from_currency for r in rates] == ["EUR", "USD", "GBP", "CHF"] and len(calls) == 4
    await svc.get_common_rates()
    assert len(calls) == 4


async def test_exchange_rejects_bad_codes_before_any_request() -> None:
    svc, calls = exchange_service(ok)
    for bad in ("EU", "EURO", "../x", "E1R"):
        with pytest.raises(ExchangeServiceError):
            await svc.get_rate(bad, "TND")
    assert calls == []


async def test_exchange_provider_error_and_no_key_leak() -> None:
    svc, _ = exchange_service(lambda r: httpx.Response(200, json={"result": "error", "error-type": "unsupported-code"}))
    with pytest.raises(ExchangeServiceError, match="unsupported-code"):
        await svc.get_rate("EUR", "ZZZ")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    svc, _ = exchange_service(boom)
    with pytest.raises(ExchangeServiceError) as exc:
        await svc.get_rate("EUR", "TND")
    assert "SECRET-KEY" not in str(exc.value)
    svc, _ = exchange_service(lambda r: httpx.Response(200, content=b"not json"))
    with pytest.raises(ExchangeServiceError):
        await svc.get_rate("EUR", "TND")
    assert json.dumps({}) == "{}"


async def test_weather_language_is_sent_and_cached_per_language() -> None:
    svc, calls = weather_service(lambda r: httpx.Response(200, json=OWM_CURRENT))
    await svc.get_current_weather("Tunis", lang="en")
    await svc.get_current_weather("Tunis", lang="en")
    await svc.get_current_weather("Tunis", lang="ar")
    assert [c.url.params["lang"] for c in calls] == ["en", "ar"]
