"""Build the configured LLM provider."""

from functools import lru_cache

from app.core.config import get_settings
from app.integrations.llm.base import BaseLLMProvider
from app.integrations.llm.mock import MockLLMProvider


@lru_cache(maxsize=1)
def get_llm_provider() -> BaseLLMProvider:
    """Singleton provider chosen by LLM_PROVIDER ('mock' or 'anthropic')."""
    settings = get_settings()
    name = settings.llm_provider
    if name == "mock":
        return MockLLMProvider()
    if name == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        from app.integrations.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            settings.anthropic_api_key, settings.anthropic_model, settings.llm_timeout_seconds
        )
    raise ValueError(f"Unsupported LLM_PROVIDER '{name}' (expected 'mock' or 'anthropic')")
