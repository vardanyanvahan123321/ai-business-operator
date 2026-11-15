"""The single message format every channel is translated into.

This is the contract that keeps the pipeline channel-agnostic: a Telegram
update, an email and a web-form submission all become an ``InboundMessage``
before anything else looks at them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Category(StrEnum):
    QUESTION = "question"
    ORDER = "order"
    COMPLAINT = "complaint"
    SPAM = "spam"
    OTHER = "other"


class MessageStatus(StrEnum):
    RECEIVED = "received"  # stored, waiting for the worker
    PROCESSING = "processing"
    ANSWERED = "answered"  # AI replied on its own
    HANDOFF = "handoff"  # a human has to take over
    IGNORED = "ignored"  # spam: stored, never answered
    DEAD = "dead"  # all retries exhausted, see dead_letters

    @property
    def is_final(self) -> bool:
        return self in {
            MessageStatus.ANSWERED,
            MessageStatus.HANDOFF,
            MessageStatus.IGNORED,
            MessageStatus.DEAD,
        }


class HandoffReason(StrEnum):
    COMPLAINT = "complaint"
    LOW_CONFIDENCE = "low_confidence"
    NO_ANSWER = "no_answer"
    INJECTION = "injection"
    OTHER = "other"
    FAILED = "failed"


class LeadStatus(StrEnum):
    INCOMPLETE = "incomplete"  # still missing mandatory fields
    NEW = "new"
    CONFIRMED = "confirmed"
    DONE = "done"
    CANCELLED = "cancelled"


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class InboundMessage(BaseModel):
    channel: str
    # Unique id of the message inside its channel; the idempotency key.
    external_id: str
    # Stable address of the customer inside the channel (chat id, email, form token).
    sender_id: str
    sender_name: str | None = None
    subject: str | None = None
    text: str
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    meta: dict[str, Any] = Field(default_factory=dict)


class Attachment(BaseModel):
    filename: str
    path: str
    mime_type: str = "application/pdf"


class OutboundMessage(BaseModel):
    channel: str
    recipient: str
    text: str
    subject: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
