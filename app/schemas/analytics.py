"""Analytics API schemas."""

from datetime import datetime

from pydantic import BaseModel


class Summary(BaseModel):
    """Headline KPIs for the period."""

    total_conversations: int
    total_messages: int
    automation_rate: float  # share of conversations with no flight/hotel/package inquiry
    avg_messages_per_conversation: float


class DayCount(BaseModel):
    date: str  # YYYY-MM-DD, Tunisia local date
    count: int


class LanguageCount(BaseModel):
    language: str
    count: int


class IntentCount(BaseModel):
    intent: str
    count: int


class InquiryRow(BaseModel):
    id: str
    inquiry_type: str
    destination: str | None
    raw_message: str  # what the customer wrote, truncated to 80 characters
    created_at: datetime
    status: str


class ConversationRow(BaseModel):
    session_id: str  # truncated to 8 characters
    language: str
    message_count: int
    first_message: str  # truncated to 50 characters
    created_at: datetime


class AnalyticsResponse(BaseModel):
    period_days: int
    generated_at: datetime
    summary: Summary
    conversations_by_day: list[DayCount]
    language_distribution: list[LanguageCount]
    intent_distribution: list[IntentCount]
    recent_inquiries: list[InquiryRow]
    recent_conversations: list[ConversationRow]
