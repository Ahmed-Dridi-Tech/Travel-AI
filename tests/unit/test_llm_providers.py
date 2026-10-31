from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import get_settings
from app.integrations.llm.anthropic_provider import AnthropicProvider
from app.integrations.llm.base import LLMMessage, LLMProviderError
from app.integrations.llm.mock import MockLLMProvider
from app.integrations.llm.provider import get_llm_provider


async def test_mock_formats_weather_from_tool_data() -> None:
    prompt = (
        'Customer message: "météo Djerba"\nTOOL_NAME: get_weather\n'
        'TOOL_DATA: {"requested_destination": "Djerba", "temperature_c": 24.0, "feels_like_c": 23.0, '
        '"humidity_percent": 55, "description": "ciel dégagé", "wind_speed_ms": 4.1}\nGuidance: x'
    )
    r = await MockLLMProvider().complete([LLMMessage("user", prompt)], system_prompt="sys")
    assert "Djerba" in r.content and "24°C" in r.content and "ciel dégagé" in r.content
    assert r.latency_ms == 50.0 and r.model == "mock-llm"
    assert r.input_tokens > 0 and r.output_tokens > 0


async def test_mock_flags_demo_data_and_errors() -> None:
    demo = 'TOOL_NAME: get_exchange_rate\nTOOL_DATA: {"from": "EUR", "to": "TND", "rate": 3.35, "note": "demo_data_no_api_key"}'
    r = await MockLLMProvider().complete([LLMMessage("user", demo)])
    assert "3.3500" in r.content and "démonstration" in r.content
    err = await MockLLMProvider().complete([LLMMessage("user", "TOOL_NAME: get_weather\nTOOL_ERROR: no_destination")])
    assert "destination" in err.content


@pytest.mark.parametrize(
    ("text", "needle"),
    [("quelle météo ?", "météo"), ("le taux de l'euro", "change"), ("parle moi de Djerba", "destination"), ("bla", "Tunisie")],
)
async def test_mock_keyword_fallbacks(text: str, needle: str) -> None:
    assert needle in (await MockLLMProvider().complete([LLMMessage("user", text)])).content


def test_factory_mock_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    for provider, key, expected in (("mock", "", MockLLMProvider), ("anthropic", "sk-test", AnthropicProvider)):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
        get_settings.cache_clear()
        get_llm_provider.cache_clear()
        assert isinstance(get_llm_provider(), expected)
        assert get_llm_provider() is get_llm_provider()
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_llm_provider()
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    with pytest.raises(ValueError, match="Unsupported"):
        get_llm_provider()
    monkeypatch.undo()
    get_settings.cache_clear()
    get_llm_provider.cache_clear()


class _FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response, self.error, self.kwargs = response, error, {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _provider(messages: _FakeMessages) -> AnthropicProvider:
    provider = AnthropicProvider("sk-test", "claude-haiku-5-5")
    provider.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]
    return provider


def _response(text: str = "Bonjour", stop: str = "end_turn") -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], model="claude-haiku-5-5", stop_reason=stop,
        usage=SimpleNamespace(input_tokens=120, output_tokens=18),
    )


async def test_anthropic_request_shape_and_accounting() -> None:
    fake = _FakeMessages(_response())
    r = await _provider(fake).complete([LLMMessage("user", "salut")], system_prompt="SYS", max_tokens=300, temperature=0.9)
    assert (r.content, r.input_tokens, r.output_tokens) == ("Bonjour", 120, 18) and r.latency_ms >= 0
    assert fake.kwargs["model"] == "claude-haiku-5-5" and fake.kwargs["system"] == "SYS"
    assert fake.kwargs["messages"] == [{"role": "user", "content": "salut"}] and fake.kwargs["max_tokens"] == 300
    assert "temperature" not in fake.kwargs  # rejected by the API since 2026-10-21


async def test_anthropic_errors_become_provider_errors() -> None:
    with pytest.raises(LLMProviderError, match="RuntimeError"):
        await _provider(_FakeMessages(error=RuntimeError("down"))).complete([LLMMessage("user", "x")])
    with pytest.raises(LLMProviderError, match="refusal"):
        await _provider(_FakeMessages(_response(stop="refusal"))).complete([LLMMessage("user", "x")])
    with pytest.raises(LLMProviderError, match="Empty"):
        await _provider(_FakeMessages(_response(text=""))).complete([LLMMessage("user", "x")])


def test_anthropic_requires_key() -> None:
    with pytest.raises(ValueError):
        AnthropicProvider("")


_WEATHER = (
    'TOOL_NAME: get_weather\nTOOL_DATA: {"requested_destination": "Sousse", "temperature_c": 24.0, "feels_like_c": 23.0, '
    '"humidity_percent": 55, "description": "%s", "wind_speed_ms": 4.1}'
)


@pytest.mark.parametrize(
    ("instruction", "description", "needle"),
    [
        ("Réponds en français.", "ciel dégagé", "il fait actuellement 24°C"),
        ("Respond in English.", "clear sky", "it is currently 24°C"),
        ("أجب باللغة العربية.", "سماء صافية", "الطقس الآن"),
        ("أجب بالدارجة التونسية.", "سماء صافية", "الطقس الآن"),
        ("Réponds en français (le client écrit en dialecte tunisien latin).", "ciel dégagé", "il fait actuellement"),
    ],
)
async def test_mock_replies_in_the_customer_language(instruction: str, description: str, needle: str) -> None:
    r = await MockLLMProvider().complete([LLMMessage("user", f"{instruction}\n" + _WEATHER % description)])
    assert needle in r.content and description in r.content


async def test_mock_translates_inquiries_errors_and_demo_flag() -> None:
    recorded = (
        'TOOL_NAME: submit_flight_inquiry\nTOOL_DATA: {"inquiry_type": "flight", "status": "recorded", '
        '"reference": "AB12CD34", "missing_fields": ["dates"]}'
    )
    en = await MockLLMProvider().complete([LLMMessage("user", "Respond in English.\n" + recorded)])
    assert "flight request (ref. AB12CD34)" in en.content and "the dates" in en.content
    ar = await MockLLMProvider().complete([LLMMessage("user", "أجب باللغة العربية.\n" + recorded)])
    assert "AB12CD34" in ar.content and "التواريخ" in ar.content
    demo = 'TOOL_NAME: get_exchange_rate\nTOOL_DATA: {"from": "EUR", "to": "TND", "rate": 3.35, "note": "demo_data_no_api_key"}'
    assert "Demo values" in (await MockLLMProvider().complete([LLMMessage("user", "Respond in English.\n" + demo)])).content
    err = await MockLLMProvider().complete([LLMMessage("user", "Respond in English.\nTOOL_NAME: get_weather\nTOOL_ERROR: no_destination")])
    assert err.content == "Which destination do you mean?"


async def test_mock_destination_uses_translated_tags_and_months() -> None:
    data = (
        'TOOL_NAME: get_destination_info\nTOOL_DATA: {"name": "Tozeur", "region": "Tozeur", "tags": ["desert", "nature"], '
        '"best_months": [3, 11], "avg_budget_tnd": 750}'
    )
    fr = (await MockLLMProvider().complete([LLMMessage("user", "Réponds en français.\n" + data)])).content
    assert "désert, nature" in fr and "mars, novembre" in fr
    en = (await MockLLMProvider().complete([LLMMessage("user", "Respond in English.\n" + data)])).content
    assert "desert, nature" in en and "March, November" in en and "750 TND" in en
    ar = (await MockLLMProvider().complete([LLMMessage("user", "أجب باللغة العربية.\n" + data)])).content
    assert "صحراء" in ar and "مارس" in ar
