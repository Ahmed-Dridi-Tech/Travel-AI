"""Repository classes and ready-to-use singletons (repositories are stateless)."""

from app.database.repositories.base import BaseRepository
from app.database.repositories.conversation_repository import ConversationRepository
from app.database.repositories.destination_repository import DestinationRepository
from app.database.repositories.inquiry_repository import InquiryRepository
from app.database.repositories.message_repository import MessageRepository

conversation_repo = ConversationRepository()
message_repo = MessageRepository()
destination_repo = DestinationRepository()
inquiry_repo = InquiryRepository()

__all__ = [
    "BaseRepository",
    "ConversationRepository",
    "DestinationRepository",
    "InquiryRepository",
    "MessageRepository",
    "conversation_repo",
    "destination_repo",
    "inquiry_repo",
    "message_repo",
]
