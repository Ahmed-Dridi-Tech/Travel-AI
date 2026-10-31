"""Chat service: thin adapter between the API schemas and the agent orchestrator."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import AgentOrchestrator
from app.database.repositories import conversation_repo, message_repo
from app.schemas.chat import ChatRequest, ChatResponse, HistoryMessage, HistoryResponse

HISTORY_LIMIT = 20


class ChatService:
    """Validated request in, API response out."""

    async def handle_message(
        self, db: AsyncSession, request: ChatRequest, orchestrator: AgentOrchestrator
    ) -> ChatResponse:
        """Run the agent for one message and map its result to the API schema."""
        agent = await orchestrator.process(request.message, request.session_id, db)
        return ChatResponse(
            session_id=request.session_id,
            message_id=agent.message_id,
            response=agent.response_text,
            language=agent.language,
            is_tunisian=agent.is_tunisian,
            intent=agent.intent,
            is_out_of_domain=agent.is_out_of_domain,
            destinations=agent.destinations,
            processing_time_ms=round(agent.processing_time_ms, 2),
            tool_used=agent.tool_used,
            tool_success=agent.tool_success,
            llm_tokens_used=agent.llm_tokens_used,
        )

    async def get_history(self, db: AsyncSession, session_id: str) -> HistoryResponse:
        """Last HISTORY_LIMIT messages of the session's most recent conversation."""
        conversation = await conversation_repo.get_by_session_id(db, session_id)
        if conversation is None:
            return HistoryResponse(session_id=session_id, messages=[])
        messages = await message_repo.get_by_conversation(db, conversation.id, limit=HISTORY_LIMIT)
        return HistoryResponse(
            session_id=session_id,
            messages=[
                HistoryMessage(
                    id=str(m.id), role=m.role.value, content=m.content,
                    intent=m.intent, confidence=m.confidence, created_at=m.created_at,
                )
                for m in messages
            ],
        )


chat_service = ChatService()
