"""In-memory per-session conversation context (MVP: lost on restart, bounded in size)."""

from collections import OrderedDict
from dataclasses import dataclass, field

from app.nlp.pipeline import NLPResult

LANGUAGE_CONFIDENCE_THRESHOLD = 0.7
MAX_REMEMBERED_DESTINATIONS = 3
MAX_SESSIONS = 1000


@dataclass
class ConversationContext:
    """What the agent remembers within a session."""

    session_id: str
    conversation_id: str | None = None
    language: str | None = None
    last_destinations: list[str] = field(default_factory=list)
    turn_count: int = 0

    def update_from_nlp(self, nlp_result: NLPResult) -> None:
        """Adopt a confidently detected language, remember up to 3 recent destinations, count the turn."""
        if nlp_result.language != "unknown" and nlp_result.language_confidence > LANGUAGE_CONFIDENCE_THRESHOLD:
            self.language = nlp_result.language
        merged: list[str] = []
        for dest in [*nlp_result.destinations, *self.last_destinations]:
            if dest not in merged:
                merged.append(dest)
        self.last_destinations = merged[:MAX_REMEMBERED_DESTINATIONS]
        self.turn_count += 1

    def get_implicit_destination(self) -> str | None:
        """Most recent destination, for follow-ups like 'et la météo ?'."""
        return self.last_destinations[0] if self.last_destinations else None


class ContextStore:
    """LRU map session_id -> ConversationContext."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._max = max_sessions
        self._items: OrderedDict[str, ConversationContext] = OrderedDict()

    def get(self, session_id: str) -> ConversationContext:
        """Fetch or create the context, marking it most recently used."""
        ctx = self._items.get(session_id)
        if ctx is None:
            ctx = ConversationContext(session_id=session_id)
            self._items[session_id] = ctx
            while len(self._items) > self._max:
                self._items.popitem(last=False)
        else:
            self._items.move_to_end(session_id)
        return ctx

    def __len__(self) -> int:
        return len(self._items)
