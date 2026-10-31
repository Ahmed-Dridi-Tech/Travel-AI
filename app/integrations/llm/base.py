"""Provider-agnostic LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMMessage:
    """One chat message. role is 'system', 'user' or 'assistant'."""

    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Completion plus accounting data."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMProviderError(Exception):
    """Any failure talking to an LLM provider (network, auth, rate limit, bad response)."""


class BaseLLMProvider(ABC):
    """Interface every provider implements."""

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Return a completion for the conversation. Raises LLMProviderError on failure."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Identifier of the underlying model."""
