"""Database schema (SQLAlchemy 2, typed)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ai_operator.config import EMBEDDING_DIM
from ai_operator.core.message import DeliveryStatus, LeadStatus, MessageStatus


class Base(DeclarativeBase):
    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: JSONB,
        list[Any]: JSONB,
        datetime: DateTime(timezone=True),
    }


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


class Message(Base):
    """An inbound customer message. Never deleted: this is the system of record."""

    __tablename__ = "messages"
    __table_args__ = (
        # Idempotency: the same channel message can be delivered twice, stored once.
        UniqueConstraint("channel", "external_id", name="uq_messages_channel_external"),
        Index("ix_messages_sender", "channel", "sender_id", "received_at"),
        Index("ix_messages_status", "status"),
        Index("ix_messages_received_at", "received_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    channel: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(255))
    sender_id: Mapped[str] = mapped_column(String(255))
    sender_name: Mapped[str | None] = mapped_column(String(255))
    subject: Mapped[str | None] = mapped_column(String(500))
    text: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)
    received_at: Mapped[datetime] = _now()

    status: Mapped[str] = mapped_column(String(20), default=MessageStatus.RECEIVED.value)
    language: Mapped[str | None] = mapped_column(String(5))
    category: Mapped[str | None] = mapped_column(String(20))
    confidence: Mapped[float | None] = mapped_column(Float)
    injection_suspected: Mapped[bool] = mapped_column(Boolean, default=False)
    analysis: Mapped[dict[str, Any] | None] = mapped_column()
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    processing_started_at: Mapped[datetime | None] = mapped_column()
    processed_at: Mapped[datetime | None] = mapped_column()

    outbound: Mapped[list[Outbound]] = relationship(back_populates="message")
    handoff: Mapped[Handoff | None] = relationship(back_populates="message")


class Outbound(Base):
    """Transactional outbox for everything we send to a customer."""

    __tablename__ = "outbound_messages"
    __table_args__ = (Index("ix_outbound_status", "status", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id", ondelete="SET NULL"))
    # Makes "send the AI reply for message 42" safe to run twice.
    dedupe_key: Mapped[str] = mapped_column(String(100), unique=True)
    channel: Mapped[str] = mapped_column(String(32))
    recipient: Mapped[str] = mapped_column(String(255))
    subject: Mapped[str | None] = mapped_column(String(500))
    text: Mapped[str] = mapped_column(Text)
    attachments: Mapped[list[Any]] = mapped_column(default=list)
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)
    kind: Mapped[str] = mapped_column(String(20))  # ai_reply | human_reply
    status: Mapped[str] = mapped_column(String(20), default=DeliveryStatus.PENDING.value)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()
    sent_at: Mapped[datetime | None] = mapped_column()

    message: Mapped[Message | None] = relationship(back_populates="outbound")


class Lead(Base):
    """A booking / order extracted from a conversation."""

    __tablename__ = "leads"
    __table_args__ = (
        Index("ix_leads_sender", "channel", "sender_id", "status"),
        Index("ix_leads_created_at", "created_at"),
        Index("ix_leads_scheduled_at", "scheduled_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id", ondelete="SET NULL"))
    channel: Mapped[str] = mapped_column(String(32))
    sender_id: Mapped[str] = mapped_column(String(255))
    name: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(40))
    email: Mapped[str | None] = mapped_column(String(255))
    service_code: Mapped[str | None] = mapped_column(String(64))
    service_text: Mapped[str | None] = mapped_column(String(255))
    scheduled_at: Mapped[datetime | None] = mapped_column()
    # We proposed ``scheduled_at`` ourselves (e.g. next year's occurrence of a date
    # that has already passed) and are waiting for the customer's "yes".
    needs_confirmation: Mapped[bool] = mapped_column(default=False, server_default=false())
    amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(3), default="AMD")
    language: Mapped[str] = mapped_column(String(5), default="en")
    status: Mapped[str] = mapped_column(String(20), default=LeadStatus.INCOMPLETE.value)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    sheet_synced_at: Mapped[datetime | None] = mapped_column()
    sheet_error: Mapped[str | None] = mapped_column(Text)

    invoice: Mapped[Invoice | None] = relationship(back_populates="lead")


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(primary_key=True)
    number: Mapped[str] = mapped_column(String(32), unique=True)
    lead_id: Mapped[int] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), unique=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(3))
    language: Mapped[str] = mapped_column(String(5))
    issued_at: Mapped[datetime] = _now()

    lead: Mapped[Lead] = relationship(back_populates="invoice")


class Handoff(Base):
    """A conversation the AI gave to a human, and why."""

    __tablename__ = "handoffs"
    __table_args__ = (Index("ix_handoffs_status", "status", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), unique=True
    )
    reason: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="open")  # open | resolved
    resolution: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()
    resolved_at: Mapped[datetime | None] = mapped_column()

    message: Mapped[Message] = relationship(back_populates="handoff")


class KBDocument(Base):
    __tablename__ = "kb_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    language: Mapped[str] = mapped_column(String(5))
    source: Mapped[str] = mapped_column(String(255), unique=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    chunks: Mapped[list[KBChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class KBChunk(Base):
    __tablename__ = "kb_chunks"
    __table_args__ = (
        Index(
            "ix_kb_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("kb_documents.id", ondelete="CASCADE"))
    ord: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Vectors from different models are not comparable, so we remember who made them.
    embedding_model: Mapped[str] = mapped_column(String(100))

    document: Mapped[KBDocument] = relationship(back_populates="chunks")


class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"
    __table_args__ = (Index("ix_prices_item", "competitor", "item_name", "scraped_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    competitor: Mapped[str] = mapped_column(String(100))
    item_name: Mapped[str] = mapped_column(String(255))
    service_code: Mapped[str | None] = mapped_column(String(64))
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(3))
    scraped_at: Mapped[datetime] = _now()


class PriceReport(Base):
    __tablename__ = "price_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = _now()
    changes: Mapped[list[Any]] = mapped_column(default=list)
    summary: Mapped[str] = mapped_column(Text)


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (UniqueConstraint("kind", "report_date", name="uq_reports_kind_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    report_date: Mapped[date] = mapped_column(Date)
    language: Mapped[str] = mapped_column(String(5))
    content: Mapped[str] = mapped_column(Text)
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = _now()


class Notification(Base):
    """Outbox for messages to the business owner."""

    __tablename__ = "notifications"
    __table_args__ = (Index("ix_notifications_status", "status", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dedupe_key: Mapped[str] = mapped_column(String(100), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default=DeliveryStatus.PENDING.value)
    delivered_via: Mapped[str | None] = mapped_column(String(32))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()
    sent_at: Mapped[datetime | None] = mapped_column()


class Event(Base):
    """Audit trail of what the pipeline did and why (PII-masked)."""

    __tablename__ = "events"
    __table_args__ = (Index("ix_events_ts", "ts"), Index("ix_events_message", "message_id"))

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = _now()
    level: Mapped[str] = mapped_column(String(10), default="info")
    kind: Mapped[str] = mapped_column(String(64))
    message_id: Mapped[int | None] = mapped_column(Integer)
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)


class DeadLetter(Base):
    """Jobs that failed every retry. Nothing is dropped silently."""

    __tablename__ = "dead_letters"

    id: Mapped[int] = mapped_column(primary_key=True)
    job: Mapped[str] = mapped_column(String(64))
    ref_id: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = _now()
    resolved_at: Mapped[datetime | None] = mapped_column()


class Setting(Base):
    """Runtime overrides editable from the panel."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column()
