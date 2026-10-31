from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.tool_registry import DEMO_NOTE, INTENT_TO_TOOL, ToolRegistry
from app.integrations.exchange import ExchangeResult
from app.integrations.weather import CityNotFoundError, WeatherResult, WeatherServiceError
from app.nlp.entity_extractor import DateEntity, EntityResult

DB: Any = object()  # the fake repositories never touch it


def dest(slug: str = "djerba", active: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        slug=slug, name_fr=slug.title(), name_ar="جربة", region="Médenine", tags=["beach", "family"],
        best_months=[4, 5, 6], avg_budget_tnd=800, active=active, description_fr=None,
    )


class FakeDestinations:
    def __init__(self, items: list[SimpleNamespace]) -> None:
        self.items = {d.slug: d for d in items}

    async def get_by_slug(self, db: Any, slug: str) -> SimpleNamespace | None:
        return self.items.get(slug)

    async def get_all_active(self, db: Any) -> list[SimpleNamespace]:
        return [d for d in self.items.values() if d.active]


class FakeInquiries:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create(self, db: Any, **values: Any) -> SimpleNamespace:
        self.created.append(values)
        return SimpleNamespace(id="12345678-1234-5678-1234-567812345678", **values)


class FakeWeather:
    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.calls = error, []

    async def get_current_weather(self, city: str, lat: float | None = None, lon: float | None = None, lang: str = "fr") -> WeatherResult:
        self.calls.append(city)
        self.coords = (lat, lon)
        self.lang = lang
        if self.error:
            raise self.error
        return WeatherResult("Djerba", "TN", 24.0, 23.5, 55, "ciel dégagé", 4.1, datetime(2026, 11, 1, tzinfo=timezone.utc))


class FakeForecastWeather(FakeWeather):
    async def get_forecast(self, city: str, days: int = 3, lat: float | None = None, lon: float | None = None, lang: str = "fr"):  # type: ignore[no-untyped-def]
        from app.integrations.weather import DayForecast, ForecastResult

        self.forecast_args = (city, days, lat, lon)
        return ForecastResult("Djerba", [DayForecast("2026-11-02", 15.0, 22.0, "nuageux"), DayForecast("2026-11-03", 16.0, 23.0, "ciel dégagé")])


class FakeExchange:
    async def get_rate(self, a: str, b: str) -> ExchangeResult:
        return ExchangeResult(a, b, 3.4, datetime(2026, 11, 1, tzinfo=timezone.utc))


def registry(weather: Any = None, exchange: Any = None, dests: Any = None, inq: Any = None) -> ToolRegistry:
    return ToolRegistry(
        weather_provider=lambda: weather, exchange_provider=lambda: exchange,
        destinations=dests or FakeDestinations([dest()]), inquiries=inq or FakeInquiries(),
    )


CONV = "00000000-0000-0000-0000-000000000001"


async def test_weather_tool_no_destination() -> None:
    r = await registry().execute("weather", EntityResult(), CONV, DB)
    assert (r.success, r.error, r.tool_name) == (False, "no_destination", "get_weather")


async def test_weather_tool_no_api_key_returns_flagged_demo_data() -> None:
    r = await registry().execute("weather", EntityResult(destinations=["djerba"]), CONV, DB)
    assert r.success and r.data["note"] == DEMO_NOTE and r.data["requested_destination"] == "Djerba"


async def test_weather_tool_real_service_and_period_note() -> None:
    svc = FakeWeather()
    entities = EntityResult(destinations=["ain-draham"], dates=[DateEntity("décembre", "2026-12", "month", 12)])
    r = await registry(weather=svc).execute("weather", entities, CONV, DB)
    assert svc.calls == ["Ain Draham,TN"] and svc.coords == (36.7775, 8.6872)  # catalogue place: queried by coordinates
    assert r.success and "note" not in r.data and r.data["temperature_c"] == 24.0
    assert r.data["period_note"] == "décembre"


async def test_weather_tool_non_tunisian_city_has_no_country_suffix() -> None:
    svc = FakeWeather()
    await registry(weather=svc).execute("weather", EntityResult(destinations=["raw:Paris"]), CONV, DB)
    assert svc.calls == ["Paris"] and svc.coords == (None, None)


@pytest.mark.parametrize(
    ("error", "prefix"),
    [(CityNotFoundError("x"), "city_not_found"), (WeatherServiceError(401, "bad key"), "weather_unavailable")],
)
async def test_weather_tool_failures_are_results_not_exceptions(error: Exception, prefix: str) -> None:
    r = await registry(weather=FakeWeather(error)).execute("weather", EntityResult(destinations=["djerba"]), CONV, DB)
    assert not r.success and r.error and r.error.startswith(prefix)


async def test_exchange_tool_no_api_key_returns_demo_rate() -> None:
    r = await registry().execute("currency", EntityResult(), CONV, DB)
    assert r.success and r.data["note"] == DEMO_NOTE
    assert (r.data["from"], r.data["to"], r.data["rate"]) == ("EUR", "TND", 3.35)


async def test_exchange_tool_demo_refuses_other_pairs() -> None:
    r = await registry().execute("currency", EntityResult(currencies=["USD", "TND"]), CONV, DB)
    assert not r.success and r.error == "demo_mode_supports_only_eur_tnd"


async def test_exchange_tool_pair_and_amount_conversion() -> None:
    entities = EntityResult(currencies=["USD", "TND"], budget="500 dollars")
    r = await registry(exchange=FakeExchange()).execute("currency", entities, CONV, DB)
    assert r.success and (r.data["from"], r.data["to"]) == ("USD", "TND")
    assert r.data["amount"] == 500 and r.data["converted"] == 1700.0


async def test_destination_tool_known() -> None:
    entities = EntityResult(destinations=["djerba"], dates=[DateEntity("mai", "2027-05", "month", 5)])
    r = await registry().execute("destination", entities, CONV, DB)
    assert r.success and r.data["region"] == "Médenine" and r.data["requested_month_is_recommended"] is True


async def test_destination_tool_unknown_place_and_suggestions() -> None:
    r = await registry().execute("destination", EntityResult(destinations=["raw:Paris"]), CONV, DB)
    assert not r.success and r.error == "destination_not_in_catalogue:Paris"
    r = await registry().execute("destination", EntityResult(), CONV, DB)
    assert r.success and r.data["suggestions"][0]["name"] == "Djerba"


async def test_faq_visa() -> None:
    r = await registry().execute("travel_faq", EntityResult(), CONV, DB, message="Faut-il un visa pour la France ?")
    assert r.success and r.data["topic"] == "visa" and "visa" in r.data["answer"].lower()


async def test_faq_unknown_question() -> None:
    r = await registry().execute("travel_faq", EntityResult(), CONV, DB, message="blabla xyz")
    assert r.success and r.data["found"] is False and r.data["available_topics"]


@pytest.mark.parametrize(("text", "topic"), [("numéro de la police ?", "emergency"), ("lazem viza lel france?", "visa"), ("prise électrique ?", "electricity")])
async def test_faq_multilingual_keywords(text: str, topic: str) -> None:
    r = await registry().execute("travel_faq", EntityResult(), CONV, DB, message=text)
    assert r.data["topic"] == topic


async def test_flight_inquiry_is_recorded_with_missing_fields() -> None:
    inq = FakeInquiries()
    entities = EntityResult(destinations=["raw:Londres"], dates=[DateEntity("du 5 au 10 janvier", "2027-01-05/2027-01-10", "range", 1)])
    r = await registry(inq=inq).execute("flight_inquiry", entities, CONV, DB, message="vol Londres", language="fr")
    assert r.success and r.data["status"] == "recorded" and r.data["missing_fields"] == ["travelers"]
    created = inq.created[0]
    assert created["inquiry_type"].value == "flight" and created["destination"] == "Londres"
    assert str(created["departure_date"]) == "2027-01-05" and str(created["return_date"]) == "2027-01-10"


async def test_inquiry_without_any_detail_is_not_recorded() -> None:
    inq = FakeInquiries()
    r = await registry(inq=inq).execute("hotel_inquiry", EntityResult(), CONV, DB, message="un hôtel")
    assert r.success and r.data["status"] == "need_details" and inq.created == []


async def test_routing_covers_all_eight_intents() -> None:
    expected = {
        "weather": "get_weather", "currency": "get_exchange_rate", "destination": "get_destination_info",
        "flight_inquiry": "submit_flight_inquiry", "hotel_inquiry": "submit_hotel_inquiry",
        "travel_faq": "get_faq_answer", "smalltalk": "none", "out_of_domain": "none",
    }
    assert set(INTENT_TO_TOOL) == set(expected)
    reg = registry()
    entities = EntityResult(destinations=["djerba"])
    for intent, tool in expected.items():
        r = await reg.execute(intent, entities, CONV, DB, message="visa")
        assert r.tool_name == tool, intent
    assert (await reg.execute("unheard_of", entities, CONV, DB)).tool_name == "none"


async def test_unexpected_exception_becomes_failed_result() -> None:
    class Boom:
        async def get_by_slug(self, *_: Any) -> None:
            raise RuntimeError("db exploded")

    r = await registry(dests=Boom()).execute("destination", EntityResult(destinations=["djerba"]), CONV, DB)
    assert not r.success and "db exploded" in (r.error or "")


async def test_flight_inquiry_stores_the_route() -> None:
    inq = FakeInquiries()
    entities = EntityResult(destinations=["tunis", "raw:Paris"], origin="tunis", flight_destination="raw:Paris")
    r = await registry(inq=inq).execute("flight_inquiry", entities, CONV, DB, message="vol Tunis Paris")
    assert inq.created[0]["destination"] == "Tunis → Paris"
    assert r.data["destination"] == "Tunis → Paris" and r.data["missing_fields"] == ["dates", "travelers"]


async def test_flight_inquiry_single_city_and_missing_origin() -> None:
    inq = FakeInquiries()
    entities = EntityResult(destinations=["raw:Paris"], flight_destination="raw:Paris")
    r = await registry(inq=inq).execute("flight_inquiry", entities, CONV, DB, message="vol pour Paris")
    assert inq.created[0]["destination"] == "Paris" and "origin" in r.data["missing_fields"]
    only_origin = EntityResult(destinations=["tunis"], origin="tunis")
    r = await registry(inq=FakeInquiries()).execute("flight_inquiry", only_origin, CONV, DB, message="depuis Tunis")
    assert r.data["destination"] == "Tunis" and r.data["missing_fields"][0] == "destination"


async def test_hotel_inquiry_ignores_route_fields() -> None:
    inq = FakeInquiries()
    entities = EntityResult(destinations=["sousse"], origin="tunis", flight_destination="sousse")
    await registry(inq=inq).execute("hotel_inquiry", entities, CONV, DB, message="hotel sousse")
    assert inq.created[0]["destination"] == "Sousse"


async def test_destination_description_follows_language() -> None:
    d = dest()
    d.description_fr, d.description_en = "Île du sud.", "Island in the south."
    reg = registry(dests=FakeDestinations([d]))
    fr = await reg.execute("destination", EntityResult(destinations=["djerba"]), CONV, DB, language="fr")
    en = await reg.execute("destination", EntityResult(destinations=["djerba"]), CONV, DB, language="en")
    assert fr.data["description"] == "Île du sud." and en.data["description"] == "Island in the south."
    d.description_en = None
    assert (await reg.execute("destination", EntityResult(destinations=["djerba"]), CONV, DB, language="en")).data["description"] == "Île du sud."


async def test_weather_tool_adds_forecast_for_near_term_periods() -> None:
    svc = FakeForecastWeather()
    entities = EntityResult(destinations=["djerba"], dates=[DateEntity("cette semaine", "this_week", "relative")])
    r = await registry(weather=svc).execute("weather", entities, CONV, DB)
    assert r.success and len(r.data["forecast"]) == 2 and "period_note" not in r.data
    assert svc.forecast_args[1:] == (5, 33.875, 10.8575)


async def test_weather_tool_far_period_gets_note_instead_of_forecast() -> None:
    svc = FakeForecastWeather()
    entities = EntityResult(destinations=["djerba"], dates=[DateEntity("décembre", "2026-12", "month", 12)])
    r = await registry(weather=svc).execute("weather", entities, CONV, DB)
    assert "forecast" not in r.data and r.data["period_note"] == "décembre"


async def test_weather_tool_forecast_failure_degrades_to_current_conditions() -> None:
    entities = EntityResult(destinations=["djerba"], dates=[DateEntity("demain", "tomorrow", "relative")])
    r = await registry(weather=FakeWeather()).execute("weather", entities, CONV, DB)  # FakeWeather has no get_forecast
    assert r.success and "forecast" not in r.data and r.data["period_note"] == "période demandée"


@pytest.mark.parametrize(("language", "owm"), [("fr", "fr"), ("tun_lat", "fr"), ("en", "en"), ("ar", "ar"), ("tun_ar", "ar")])
async def test_weather_description_language_follows_the_customer(language: str, owm: str) -> None:
    svc = FakeWeather()
    await registry(weather=svc).execute("weather", EntityResult(destinations=["djerba"]), CONV, DB, language=language)
    assert svc.lang == owm
