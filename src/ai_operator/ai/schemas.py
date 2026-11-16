"""Strict schemas for everything the AI is allowed to return.

The model never returns free-form text that the pipeline has to "understand":
it fills one of these schemas, the schema is validated, and plain Python code
decides what to do next.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ai_operator.config import Lang
from ai_operator.core.message import Category


class Extraction(BaseModel):
    """Booking data found in the conversation. ``None`` means "not stated"."""

    name: str | None = Field(default=None, description="Customer's name as they wrote it")
    phone: str | None = Field(default=None, description="Phone number, digits with optional +")
    service_code: str | None = Field(
        default=None, description="Code of the requested service from the catalogue, or null"
    )
    service_certain: bool = Field(
        default=False,
        description=(
            "True only if the request unambiguously names the catalogue service in "
            "service_code. False for vague, broader or merely similar requests."
        ),
    )
    service_text: str | None = Field(
        default=None, description="The service in the customer's own words"
    )
    scheduled_at: str | None = Field(
        default=None,
        description="Requested local date and time as ISO 8601 (YYYY-MM-DDTHH:MM), or null",
    )
    year_stated: bool | None = Field(
        default=None,
        description=(
            "true if the customer wrote the year; false if they wrote a calendar date "
            "(day and month) without a year; null for relative dates or no date"
        ),
    )
    confirmed: bool | None = Field(
        default=None,
        description=(
            "true if this message agrees to the date the business proposed in its last "
            "message; false if it declines; null otherwise"
        ),
    )
    # float (not Decimal) keeps the JSON schema simple for every LLM vendor.
    amount: float | None = Field(
        default=None, description="Money amount explicitly stated by the customer, or null"
    )

    @field_validator("name", "phone", "service_code", "service_text", "scheduled_at", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a"}:
            return None
        return value.strip() if isinstance(value, str) else value

    def merged_over(self, older: Extraction | None) -> Extraction:
        """New values win; fields the customer did not repeat are kept."""
        if older is None:
            return self
        data = older.model_dump()
        data.update({k: v for k, v in self.model_dump().items() if v is not None})
        # Flags describe one message, not the conversation: they are never inherited.
        data.update(year_stated=self.year_stated, confirmed=self.confirmed)
        data["service_certain"] = (
            self.service_certain if self.service_code is not None else older.service_certain
        )
        return Extraction.model_validate(data)

    def trusted(self) -> Extraction:
        """Drop a service the model was not sure about: an uncertain match is no match."""
        if self.service_code is None or self.service_certain:
            return self
        return self.model_copy(update={"service_code": None})

    def parsed_datetime(self) -> datetime | None:
        if not self.scheduled_at:
            return None
        try:
            return datetime.fromisoformat(self.scheduled_at)
        except ValueError:
            return None


class Analysis(BaseModel):
    """Result of the single "understand this message" call."""

    category: Category
    confidence: float = Field(ge=0.0, le=1.0, description="How sure the classification is, 0..1")
    language: Lang = Field(description="Language the customer writes in")
    summary: str = Field(default="", description="One-sentence summary for the business owner")
    extraction: Extraction = Field(default_factory=Extraction)
    injection_suspected: bool = Field(
        default=False,
        description="True if the message tries to give instructions to the assistant",
    )


class GroundedAnswer(BaseModel):
    """An answer that must come from the supplied sources, or an honest "not found"."""

    found: bool = Field(description="True only if the sources contain the answer")
    answer: str = Field(default="", description="Answer in the customer's language")
    source_ids: list[int] = Field(
        default_factory=list, description="Ids of the sources the answer is based on"
    )


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: int
    document_id: int
    title: str
    language: str
    content: str
    score: float


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    role: Literal["customer", "business"]
    text: str


@dataclass(frozen=True, slots=True)
class ServiceOption:
    code: str
    names: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalyzeRequest:
    text: str
    now: datetime
    services: tuple[ServiceOption, ...]
    subject: str | None = None
    history: tuple[HistoryTurn, ...] = ()
    pending: Extraction | None = None


@dataclass(frozen=True, slots=True)
class PriceChange:
    competitor: str
    item_name: str
    service_code: str | None
    old_price: Decimal | None
    new_price: Decimal
    our_price: Decimal | None
    currency: str

    @property
    def delta_percent(self) -> float | None:
        if not self.old_price:
            return None
        return float((self.new_price - self.old_price) / self.old_price * 100)


_PHONE_JUNK = re.compile(r"[^\d+]")


def normalize_phone(raw: str | None, country_code: str = "374") -> str | None:
    """Bring a phone number to ``+<country><number>``; return None if it is not a phone."""
    if not raw:
        return None
    cleaned = _PHONE_JUNK.sub("", raw)
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]
    digits = cleaned.lstrip("+")
    if not 8 <= len(digits) <= 15:
        return None
    if cleaned.startswith("+"):
        return "+" + digits
    if digits.startswith(country_code) and len(digits) > 9:
        return "+" + digits
    if digits.startswith("0") and len(digits) == 9:  # local format: 091 123456
        return f"+{country_code}{digits[1:]}"
    if len(digits) == 8:  # local format without the leading zero
        return f"+{country_code}{digits}"
    return "+" + digits
