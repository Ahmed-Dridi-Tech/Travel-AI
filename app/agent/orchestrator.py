"""Agent orchestrator: NLP -> persistence -> tool -> LLM formatting -> persistence."""

import logging
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import ContextStore
from app.agent.prompts import (
    FALLBACK_MESSAGES,
    OOD_MESSAGES,
    SYSTEM_PROMPT,
    build_user_prompt,
    message_group,
)
from app.agent.tool_registry import ToolRegistry, ToolResult
from app.core.config import get_settings
from app.database.models import RoleEnum
from app.database.repositories import conversation_repo, message_repo
from app.integrations.llm.base import BaseLLMProvider, LLMMessage
from app.integrations.llm.provider import get_llm_provider
from app.nlp.language_detector import MIN_RELIABLE_CONFIDENCE
from app.nlp import get_pipeline
from app.nlp.pipeline import NLPResult

logger = logging.getLogger(__name__)

HISTORY_MESSAGES = 5
IMPLICIT_DESTINATION_INTENTS = {"weather", "destination", "hotel_inquiry", "flight_inquiry"}


@dataclass
class AgentResponse:
    """Everything the API layer needs to answer one chat turn."""

    response_text: str
    language: str
    intent: str
    is_out_of_domain: bool
    destinations: list[str]
    tool_used: str
    tool_success: bool
    is_tunisian: bool
    processing_time_ms: float
    llm_tokens_used: int
    message_id: str = ""
    extras: dict[str, str] = field(default_factory=dict)


class AgentOrchestrator:
    """Runs one conversational turn. The LLM only formats what NLP and tools produced."""

    def __init__(
        self,
        nlp: Any | None = None,
        tool_registry: ToolRegistry | None = None,
        llm: BaseLLMProvider | None = None,
        contexts: ContextStore | None = None,
    ) -> None:
        self.nlp = nlp or get_pipeline()
        self.tool_registry = tool_registry or ToolRegistry()
        self.llm = llm or get_llm_provider()
        self.contexts = contexts or ContextStore()

    async def process(self, user_message: str, session_id: str, db: AsyncSession) -> AgentResponse:
        """Handle one user message end to end. LLM or tool failures never propagate."""
        started = time.perf_counter()
        settings = get_settings()
        nlp: NLPResult = await run_in_threadpool(self.nlp.analyze, user_message)

        ctx = self.contexts.get(session_id)
        implicit = ctx.get_implicit_destination()
        entities = nlp.entities
        if nlp.intent in IMPLICIT_DESTINATION_INTENTS and not entities.destinations and implicit:
            entities = replace(entities, destinations=[implicit])  # "et la météo ?" reuses the last place
        ctx.update_from_nlp(nlp)

        # Conversation + history (loaded before saving, so the current turn is not duplicated).
        history: list[LLMMessage] = []
        conversation = await conversation_repo.get_by_session_id(db, session_id)
        language = self._resolve_language(nlp, ctx.language, ctx.turn_count, conversation)
        group = message_group(language)
        if conversation is None:
            conversation = await conversation_repo.create_with_first_message(db, session_id, language, user_message)
            await message_repo.update(db, conversation.messages[0], intent=nlp.intent, confidence=nlp.intent_confidence)
        else:
            recent = await message_repo.get_by_conversation(db, conversation.id, limit=HISTORY_MESSAGES)
            history = [LLMMessage(m.role.value, m.content) for m in recent]
            if language != "unknown" and conversation.language != language:
                await conversation_repo.update(db, conversation, language=language)
            await message_repo.add_message(
                db, conversation.id, RoleEnum.USER, user_message, nlp.intent, nlp.intent_confidence
            )
        ctx.conversation_id = str(conversation.id)

        tool_result = ToolResult(success=True, tool_name="none")
        tokens = 0
        if nlp.ood_reason == "pipeline_error":
            reply = FALLBACK_MESSAGES[group]
        elif nlp.is_out_of_domain:
            reply = OOD_MESSAGES[group]  # canned: no LLM cost, no prompt-injection surface
        else:
            tool_result = await self.tool_registry.execute(
                nlp.intent, entities, str(conversation.id), db, message=user_message, language=language
            )
            reply, tokens = await self._format_reply(nlp, tool_result, history, language, group, settings)

        assistant = await message_repo.add_message(
            db, conversation.id, RoleEnum.ASSISTANT, reply, nlp.intent, nlp.intent_confidence
        )
        return AgentResponse(
            response_text=reply,
            language=language,
            intent=nlp.intent,
            is_out_of_domain=nlp.is_out_of_domain,
            destinations=entities.destinations,
            tool_used=tool_result.tool_name,
            tool_success=tool_result.success,
            is_tunisian=nlp.is_tunisian,
            processing_time_ms=(time.perf_counter() - started) * 1000,
            llm_tokens_used=tokens,
            message_id=str(assistant.id),
        )

    @staticmethod
    def _resolve_language(nlp: NLPResult, ctx_language: str | None, turn_count: int, conversation) -> str:  # type: ignore[no-untyped-def]
        """Never return 'unknown'. A reliable detection wins; otherwise reuse the session's language
        (context, else the stored conversation) from the second message on; the first message keeps
        the detector's script-based guess ('fr' for Latin text)."""
        if nlp.language != "unknown" and nlp.language_confidence >= MIN_RELIABLE_CONFIDENCE:
            return nlp.language
        stored = conversation.language if conversation is not None and conversation.language != "unknown" else None
        if turn_count > 1 or stored:
            return ctx_language or stored or "fr"
        return nlp.language if nlp.language != "unknown" else "fr"  # first message: keep the script-based guess

    async def _format_reply(
        self, nlp: NLPResult, tool_result: ToolResult, history: list[LLMMessage],
        language: str, group: str, settings,  # type: ignore[no-untyped-def]
    ) -> tuple[str, int]:
        prompt = build_user_prompt(nlp, tool_result, history, language)
        try:
            response = await self.llm.complete(
                [LLMMessage("user", prompt)],
                system_prompt=SYSTEM_PROMPT,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
        except Exception:  # noqa: BLE001 - any provider failure degrades to a canned reply
            logger.exception("LLM call failed; sending fallback reply")
            return FALLBACK_MESSAGES[group], 0
        return response.content, response.input_tokens + response.output_tokens


@lru_cache(maxsize=1)
def get_agent_orchestrator() -> AgentOrchestrator:
    """Process-wide orchestrator (loads NLP models and builds the configured LLM provider)."""
    return AgentOrchestrator()
