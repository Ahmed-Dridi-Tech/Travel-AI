"""Chat endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import AgentOrchestrator, get_agent_orchestrator
from app.database.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse, HistoryResponse
from app.services.chat_service import chat_service

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def get_orchestrator() -> AgentOrchestrator:
    """Dependency returning the shared agent (overridable in tests)."""
    return get_agent_orchestrator()


@router.post("", response_model=ChatResponse)
async def post_chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """Analyze the message, call the right tool, persist the turn and return the agent's reply."""
    return await chat_service.handle_message(db, request, orchestrator)


@router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str, db: AsyncSession = Depends(get_db)) -> HistoryResponse:
    """Return the last 20 messages of a session (empty list if the session is unknown)."""
    return await chat_service.get_history(db, session_id)
