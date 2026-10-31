"""Tools the agent may call. Each tool returns real data (API or local DB) or an explicit failure."""

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.faq import find_faq, list_topics
from app.database.models import InquiryTypeEnum
from app.database.repositories import destination_repo, inquiry_repo
from app.integrations.exchange import ExchangeService, get_exchange_service
from app.integrations.weather import CityNotFoundError, WeatherService, get_weather_service
from app.nlp.entity_extractor import EntityResult

logger = logging.getLogger(__name__)

DEMO_NOTE = "demo_data_no_api_key"
FORECAST_PERIODS = frozenset({"tomorrow", "day_after_tomorrow", "this_week", "this_weekend"})  # inside the 5-day forecast
DEMO_EUR_TND_RATE = 3.35

INTENT_TO_TOOL: dict[str, str | None] = {
    "weather": "get_weather",
    "currency": "get_exchange_rate",
    "destination": "get_destination_info",
    "flight_inquiry": "submit_flight_inquiry",
    "hotel_inquiry": "submit_hotel_inquiry",
    "travel_faq": "get_faq_answer",
    "smalltalk": None,
    "out_of_domain": None,
}


@dataclass
class ToolResult:
    """Outcome of one tool call; `data` is what the LLM is allowed to talk about."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    tool_name: str = "none"
    execution_time_ms: float = 0.0


@lru_cache(maxsize=1)
def _coordinates() -> dict[str, tuple[float, float]]:
    """slug -> (lat, lon) from the gazetteer; weather is queried by coordinates for catalogue places."""
    path = Path(__file__).resolve().parents[2] / "data" / "destinations_gazetteer.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["destinations"]
    except (OSError, ValueError, KeyError):
        logger.warning("Gazetteer coordinates unavailable; weather falls back to name queries")
        return {}
    return {e["slug"]: (e["lat"], e["lon"]) for e in entries if "lat" in e and "lon" in e}


def _clean(name: str) -> str:
    return name.removeprefix("raw:").replace("-", " ").title()


def _parse_amount(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"\d[\d\s.,]*", text)
    if not match:
        return None
    raw = match.group().strip().replace(" ", "")
    if "," in raw and "." not in raw:
        raw = raw.replace(",", ".") if re.search(r",\d{1,2}$", raw) else raw.replace(",", "")
    else:
        raw = raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


class ToolRegistry:
    """Maps intents to tools. Services are injected as providers so tests can swap them."""

    def __init__(
        self,
        weather_provider: Callable[[], WeatherService | None] = get_weather_service,
        exchange_provider: Callable[[], ExchangeService | None] = get_exchange_service,
        destinations: Any = destination_repo,
        inquiries: Any = inquiry_repo,
    ) -> None:
        self._weather_provider = weather_provider
        self._exchange_provider = exchange_provider
        self._destinations = destinations
        self._inquiries = inquiries

    async def execute(
        self,
        intent: str,
        entities: EntityResult,
        conversation_id: str,
        db: AsyncSession | None = None,
        *,
        message: str = "",
        language: str = "fr",
    ) -> ToolResult:
        """Run the tool mapped to `intent`. Never raises: failures become ToolResult(success=False)."""
        tool = INTENT_TO_TOOL.get(intent)
        if tool is None:
            return ToolResult(success=True, data={}, tool_name="none")
        started = time.perf_counter()
        try:
            if tool == "get_weather":
                result = await self._get_weather(entities, language)
            elif tool == "get_exchange_rate":
                result = await self._get_exchange_rate(entities)
            elif tool == "get_destination_info":
                result = await self._get_destination_info(entities, db, language)
            elif tool == "submit_flight_inquiry":
                result = await self._submit_inquiry(InquiryTypeEnum.FLIGHT, tool, entities, conversation_id, db, message, language)
            elif tool == "submit_hotel_inquiry":
                result = await self._submit_inquiry(InquiryTypeEnum.HOTEL, tool, entities, conversation_id, db, message, language)
            else:
                result = self._get_faq_answer(message)
        except Exception as exc:  # noqa: BLE001 - a tool failure must never break the conversation
            logger.exception("Tool %s failed", tool)
            result = ToolResult(False, error=f"{type(exc).__name__}: {exc}", tool_name=tool)
        result.execution_time_ms = (time.perf_counter() - started) * 1000
        return result

    # ------------------------------------------------------------------ weather
    async def _get_weather(self, entities: EntityResult, language: str = "fr") -> ToolResult:
        name = "get_weather"
        if not entities.destinations:
            return ToolResult(False, error="no_destination", tool_name=name)
        target = entities.destinations[0]
        label = _clean(target)
        city = label if target.startswith("raw:") else f"{label},TN"
        wants_forecast = any(d.normalized in FORECAST_PERIODS for d in entities.dates)
        period = [d.raw for d in entities.dates if d.normalized != "today" and d.normalized not in FORECAST_PERIODS]
        lat, lon = _coordinates().get(target, (None, None))
        owm_lang = {"en": "en", "ar": "ar", "tun_ar": "ar"}.get(language, "fr")  # descriptions in the customer's language
        service = self._weather_provider()
        if service is None:
            data: dict[str, Any] = {
                "requested_destination": label, "city": label, "country": "TN",
                "temperature_c": 22.0, "feels_like_c": 21.0, "humidity_percent": 60,
                "description": "ciel dégagé", "wind_speed_ms": 3.0, "note": DEMO_NOTE,
            }
        else:
            try:
                weather = await service.get_current_weather(city, lat=lat, lon=lon, lang=owm_lang)
            except CityNotFoundError:
                return ToolResult(False, error=f"city_not_found:{label}", tool_name=name)
            except Exception as exc:  # noqa: BLE001
                return ToolResult(False, error=f"weather_unavailable:{exc}", tool_name=name)
            data = {**weather.to_dict(), "requested_destination": label}
            if wants_forecast:
                try:
                    forecast = await service.get_forecast(city, days=5, lat=lat, lon=lon, lang=owm_lang)
                    data["forecast"] = [
                        {"date": d.date, "temp_min_c": d.temp_min_c, "temp_max_c": d.temp_max_c, "description": d.description}
                        for d in forecast.days
                    ]
                except Exception:  # noqa: BLE001 - current conditions are still worth answering with
                    logger.warning("Forecast unavailable for %s", label)
                    period.append("période demandée")
        if period:
            data["period_note"] = ", ".join(period)  # current conditions only, flagged for the LLM
        return ToolResult(True, data, tool_name=name)

    # ------------------------------------------------------------------ exchange
    async def _get_exchange_rate(self, entities: EntityResult) -> ToolResult:
        name = "get_exchange_rate"
        codes = entities.currencies
        if len(codes) >= 2:
            src, dst = codes[0], codes[1]
        elif len(codes) == 1 and codes[0] != "TND":
            src, dst = codes[0], "TND"
        elif len(codes) == 1:
            src, dst = "TND", "EUR"
        else:
            src, dst = "EUR", "TND"
        amount = _parse_amount(entities.budget) if entities.budget and entities.currencies else None
        service = self._exchange_provider()
        if service is None:
            if {src, dst} != {"EUR", "TND"}:
                return ToolResult(False, error="demo_mode_supports_only_eur_tnd", tool_name=name)
            rate = DEMO_EUR_TND_RATE if src == "EUR" else 1 / DEMO_EUR_TND_RATE
            data: dict[str, Any] = {"from": src, "to": dst, "rate": rate, "note": DEMO_NOTE}
        else:
            try:
                result = await service.get_rate(src, dst)
            except Exception as exc:  # noqa: BLE001
                return ToolResult(False, error=f"exchange_unavailable:{exc}", tool_name=name)
            data = {"from": src, "to": dst, "rate": result.rate, "last_updated": result.last_updated.isoformat()}
        if amount is not None:
            data["amount"] = amount
            data["converted"] = round(amount * data["rate"], 2)  # computed here, never by the LLM
        return ToolResult(True, data, tool_name=name)

    # ------------------------------------------------------------------ destinations
    async def _get_destination_info(
        self, entities: EntityResult, db: AsyncSession | None, language: str = "fr"
    ) -> ToolResult:
        name = "get_destination_info"
        if db is None:
            return ToolResult(False, error="no_database", tool_name=name)
        slugs = [d for d in entities.destinations if not d.startswith("raw:")]
        if not slugs:
            if entities.destinations:  # a place we do not carry
                return ToolResult(False, error=f"destination_not_in_catalogue:{_clean(entities.destinations[0])}", tool_name=name)
            top = (await self._destinations.get_all_active(db))[:3]
            if not top:
                return ToolResult(False, error="catalogue_empty", tool_name=name)
            return ToolResult(True, {"suggestions": [{"name": d.name_fr, "region": d.region, "tags": d.tags} for d in top]}, tool_name=name)
        dest = await self._destinations.get_by_slug(db, slugs[0])
        if dest is None or not dest.active:
            return ToolResult(False, error=f"destination_not_in_catalogue:{_clean(slugs[0])}", tool_name=name)
        data: dict[str, Any] = {
            "slug": dest.slug, "name": dest.name_fr, "name_ar": dest.name_ar, "region": dest.region,
            "tags": list(dest.tags), "best_months": list(dest.best_months),
            "avg_budget_tnd": dest.avg_budget_tnd,
            "budget_basis": "estimation indicative, 3 nuits pour 2 personnes",
        }
        description = (dest.description_en if language == "en" else None) or dest.description_fr
        if description:
            data["description"] = description
        months = [d.month for d in entities.dates if d.month]
        if months:
            data["requested_months"] = months
            data["requested_month_is_recommended"] = all(m in dest.best_months for m in months)
        return ToolResult(True, data, tool_name=name)

    # ------------------------------------------------------------------ inquiries
    async def _submit_inquiry(
        self,
        kind: InquiryTypeEnum,
        tool_name: str,
        entities: EntityResult,
        conversation_id: str,
        db: AsyncSession | None,
        message: str,
        language: str,
    ) -> ToolResult:
        if db is None:
            return ToolResult(False, error="no_database", tool_name=tool_name)
        is_flight = kind == InquiryTypeEnum.FLIGHT
        if is_flight and (entities.origin or entities.flight_destination):
            # "Tunis → Paris" when both ends are known, otherwise the single city we have
            destination = " → ".join(_clean(p) for p in (entities.origin, entities.flight_destination) if p)
        else:
            destination = ", ".join(_clean(d) for d in entities.destinations) or None
        departure: date | None = None
        return_date: date | None = None
        for d in entities.dates:
            try:
                if d.type == "range":
                    first, second = d.normalized.split("/")
                    departure, return_date = date.fromisoformat(first), date.fromisoformat(second)
                    break
                if d.type == "absolute" and departure is None:
                    departure = date.fromisoformat(d.normalized)
            except ValueError:
                continue
        has_destination = bool(entities.flight_destination) if is_flight and entities.origin else bool(destination)
        missing = [
            label
            for label, present in (
                ("destination", has_destination),
                ("origin", not is_flight or entities.origin or not entities.flight_destination),
                ("dates", departure or entities.dates),
                ("travelers", entities.travelers),
            )
            if not present
        ]
        base = {"inquiry_type": kind.value, "missing_fields": missing}
        if not (destination or entities.dates or entities.travelers):
            return ToolResult(True, {**base, "status": "need_details"}, tool_name=tool_name)
        inquiry = await self._inquiries.create(
            db,
            conversation_id=uuid.UUID(conversation_id),
            inquiry_type=kind,
            destination=destination,
            departure_date=departure,
            return_date=return_date,
            travelers=entities.travelers,
            budget=entities.budget,
            language=language,
            raw_message=message,
        )
        data = {
            **base, "status": "recorded", "inquiry_id": str(inquiry.id),
            "reference": str(inquiry.id)[:8].upper(), "destination": destination,
            "dates": [d.raw for d in entities.dates], "travelers": entities.travelers,
            "budget": entities.budget, "next_step": "advisor_callback",
        }
        return ToolResult(True, data, tool_name=tool_name)

    # ------------------------------------------------------------------ FAQ
    @staticmethod
    def _get_faq_answer(message: str) -> ToolResult:
        name = "get_faq_answer"
        entry = find_faq(message)
        if entry is None:
            return ToolResult(True, {"found": False, "available_topics": list_topics()}, tool_name=name)
        return ToolResult(True, {"found": True, "topic": entry.topic, "answer": entry.answer}, tool_name=name)
