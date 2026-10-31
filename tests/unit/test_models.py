import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import configure_mappers

from app.database.models import (
    Base,
    Conversation,
    Destination,
    Feedback,
    Inquiry,
    InquiryStatusEnum,
    InquiryTypeEnum,
    Message,
    RoleEnum,
)


def _col_default(model: type, column: str, ctx: object = None) -> object:
    """Evaluate a column's Python-side default without a database."""
    return getattr(model, "__table__").c[column].default.arg(ctx)


def test_five_tables_registered() -> None:
    assert set(Base.metadata.tables) == {
        "conversations",
        "messages",
        "inquiries",
        "destinations",
        "feedback",
    }


def test_mappers_configure() -> None:
    configure_mappers()


def test_enum_values() -> None:
    assert [e.value for e in RoleEnum] == ["user", "assistant"]
    assert [e.value for e in InquiryTypeEnum] == ["flight", "hotel", "package", "other"]
    assert [e.value for e in InquiryStatusEnum] == ["pending", "handled", "closed"]


def test_instantiation_without_db() -> None:
    conv = Conversation(session_id="s1", language="fr")
    msg = Message(conversation=conv, role=RoleEnum.USER, content="Bonjour")
    inq = Inquiry(
        conversation=conv,
        inquiry_type=InquiryTypeEnum.HOTEL,
        language="fr",
        raw_message="hotel djerba",
    )
    dest = Destination(slug="djerba", name_fr="Djerba", name_ar="جربة", name_en="Djerba", region="Sud")
    fb = Feedback(message=msg, rating=1)
    assert conv.messages == [msg] and conv.inquiries == [inq]
    assert msg.feedback == [fb]
    assert dest.slug == "djerba"


def test_uuid_mixin_generates_valid_uuid() -> None:
    value = _col_default(Conversation, "id")
    assert isinstance(value, uuid.UUID) and value.version == 4
    assert _col_default(Conversation, "id") != value


def test_timestamp_mixin_defaults() -> None:
    created = _col_default(Conversation, "created_at")
    updated = _col_default(Conversation, "updated_at")
    assert isinstance(created, datetime) and created.tzinfo == timezone.utc
    assert isinstance(updated, datetime)
    assert Conversation.__table__.c.updated_at.onupdate is not None
    assert "updated_at" not in Message.__table__.c  # messages are immutable


def test_destination_array_columns() -> None:
    cols = Destination.__table__.c
    assert cols.tags.type.item_type.__class__.__name__ == "String"
    assert cols.best_months.type.item_type.__class__.__name__ == "Integer"


def test_utcnow_strictly_increasing() -> None:
    from app.database.models.base import utcnow

    stamps = [utcnow() for _ in range(2000)]
    assert all(a < b for a, b in zip(stamps, stamps[1:]))
