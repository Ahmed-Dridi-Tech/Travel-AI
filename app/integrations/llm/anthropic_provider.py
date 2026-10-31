"""Anthropic Claude provider (native async client)."""

import time
from typing import Any

import anthropic

from app.integrations.llm.base import BaseLLMProvider, LLMMessage, LLMProviderError, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Calls the Messages API.

    `temperature` is accepted for interface symmetry but never sent: the API rejects sampling
    parameters (400) since 2026-10-21. Thinking is disabled and effort is low because this is a
    plain formatting task and the whole token budget should go to the answer.
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-5-5", timeout: float = 20.0) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=1)

    def get_model_name(self) -> str:
        return self.model

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role in {"user", "assistant"}]
        system = "\n\n".join(
            filter(None, [system_prompt, *(m.content for m in messages if m.role == "system")])
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": chat,
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "low"},
        }
        if system:
            kwargs["system"] = system
        started = time.perf_counter()
        try:
            response = await self.client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - anthropic.APIError and transport errors alike
            raise LLMProviderError(f"{type(exc).__name__}: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMProviderError("Anthropic declined the request (stop_reason=refusal)")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMProviderError(f"Empty completion (stop_reason={response.stop_reason})")
        return LLMResponse(
            content=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
