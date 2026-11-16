"""Configuration.

Two layers, on purpose:

* ``Settings`` - infrastructure and secrets, read from environment / ``.env``.
* ``BusinessConfig`` - everything that describes *the business* (name, services,
  prices, competitors, schedule). It lives in a YAML file so the system can be
  re-pointed at another company without touching code.
"""

from __future__ import annotations

import os
import re
from datetime import date, time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Lang = Literal["en", "ru", "hy"]
LANGS: tuple[Lang, ...] = ("en", "ru", "hy")
AIProviderName = Literal["gemini", "anthropic", "openai", "mock"]

EMBEDDING_DIM = 768


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- runtime -----------------------------------------------------------
    demo_mode: bool = False
    log_level: str = "INFO"
    log_json: bool = True
    data_dir: Path = Path("var")
    business_config: Path = Path("config/business.yaml")
    public_base_url: str = "http://localhost:8088"

    # --- storage -----------------------------------------------------------
    database_url: str = "postgresql+asyncpg://operator:operator@localhost:5432/operator"
    redis_url: str = "redis://localhost:6379/0"

    # --- AI ----------------------------------------------------------------
    ai_provider: AIProviderName = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # Free tiers are tight: we throttle ourselves instead of hitting 429s.
    ai_requests_per_minute: int = 10
    ai_max_retries: int = 4
    ai_cache_ttl_seconds: int = 24 * 3600
    ai_timeout_seconds: float = 45.0

    # --- decision thresholds -----------------------------------------------
    confidence_threshold: float = 0.65
    rag_min_score: float = 0.30
    rag_top_k: int = 4

    # --- queue -------------------------------------------------------------
    max_attempts: int = 5
    retry_base_seconds: int = 5
    stuck_after_seconds: int = 300

    # --- Telegram ----------------------------------------------------------
    telegram_bot_token: str = ""
    telegram_owner_chat_id: str = ""
    telegram_webhook_secret: str = ""

    # --- Email -------------------------------------------------------------
    email_inbox: Literal["none", "imap", "mailpit"] = "none"
    email_address: str = "service@demo-auto.example"
    imap_host: str = ""
    imap_port: int = 993
    imap_user: str = ""
    imap_password: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    mailpit_api_url: str = "http://mailpit:8025"

    # --- Google Sheets -----------------------------------------------------
    google_sheets_id: str = ""
    google_service_account_file: str = ""

    # --- web panel ---------------------------------------------------------
    admin_password: str = ""
    session_secret: str = ""
    # Public form: messages accepted from one client address per window (0 = no limit).
    webform_max_messages: int = 10
    webform_window_seconds: int = 300

    @field_validator("database_url")
    @classmethod
    def _force_asyncpg(cls, value: str) -> str:
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value

    @property
    def invoices_dir(self) -> Path:
        return self.data_dir / "invoices"

    @property
    def leads_csv(self) -> Path:
        return self.data_dir / "leads.csv"

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token)

    @property
    def sheets_enabled(self) -> bool:
        return bool(self.google_sheets_id and self.google_service_account_file)


class LocalizedText(BaseModel):
    """A string in all three supported languages."""

    en: str
    ru: str
    hy: str

    def get(self, lang: str) -> str:
        return str(getattr(self, lang, self.en))


class Service(BaseModel):
    code: str
    name: LocalizedText
    # Words that identify the service in a customer's message (any language).
    keywords: list[str] = Field(default_factory=list)
    price: Decimal | None = None
    price_from: bool = False
    duration_minutes: int = 60


class Competitor(BaseModel):
    name: str
    url: str
    item_selector: str
    name_selector: str
    price_selector: str
    # Maps a competitor's item title to our service code, for side-by-side comparison.
    service_map: dict[str, str] = Field(default_factory=dict)


class ScrapeConfig(BaseModel):
    user_agent: str = "AIBusinessOperatorBot/1.0 (+price monitoring; respects robots.txt)"
    min_delay_seconds: float = 2.0
    timeout_seconds: float = 15.0
    hour: int = 7
    minute: int = 30


class ReportConfig(BaseModel):
    hour: int = 8
    minute: int = 0


class Seller(BaseModel):
    legal_name: str
    tax_id: str = ""
    address: LocalizedText
    phone: str
    email: str
    bank_details: str = ""


WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _default_hours() -> dict[str, tuple[time, time] | None]:
    return {day: (time(9), time(19)) for day in WEEKDAYS[:6]}


class Schedule(BaseModel):
    """Machine-readable opening hours: what the booking rules check slots against.

    ``working_hours`` above stays the human-readable text shown to customers.
    """

    # Weekday -> (open, close); a missing or null day is a day off.
    hours: dict[str, tuple[time, time] | None] = Field(default_factory=_default_hours)
    slot_minutes: int = Field(default=30, ge=5, le=240)
    # How many customers can be served at the same time (lifts, mechanics, chairs...).
    parallel_bookings: int = Field(default=1, ge=1)

    @field_validator("hours")
    @classmethod
    def _known_days(
        cls, value: dict[str, tuple[time, time] | None]
    ) -> dict[str, tuple[time, time] | None]:
        for day, span in value.items():
            if day not in WEEKDAYS:
                raise ValueError(f"unknown weekday {day!r}; use one of {', '.join(WEEKDAYS)}")
            if span is not None and span[0] >= span[1]:
                raise ValueError(f"{day}: opening time must be before closing time")
        return value

    def hours_on(self, day: date) -> tuple[time, time] | None:
        return self.hours.get(WEEKDAYS[day.weekday()])


class BusinessConfig(BaseModel):
    name: LocalizedText
    description: LocalizedText
    timezone: str = "Asia/Yerevan"
    currency: str = "AMD"
    default_language: Lang = "en"
    owner_language: Lang = "ru"
    working_hours: LocalizedText
    schedule: Schedule = Field(default_factory=Schedule)
    seller: Seller
    services: list[Service]
    # Offered when a request matches no service unambiguously ("something is knocking").
    fallback_service: str | None = None
    competitors: list[Competitor] = Field(default_factory=list)
    scrape: ScrapeConfig = Field(default_factory=ScrapeConfig)
    morning_report: ReportConfig = Field(default_factory=ReportConfig)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @model_validator(mode="after")
    def _fallback_exists(self) -> BusinessConfig:
        if self.fallback_service and self.service(self.fallback_service) is None:
            raise ValueError(f"fallback_service {self.fallback_service!r} is not in services")
        return self

    def service(self, code: str | None) -> Service | None:
        if not code:
            return None
        return next((s for s in self.services if s.code == code), None)


@lru_cache
def get_settings() -> Settings:
    return Settings()


_ENV_REF = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)(?::-(?P<default>[^}]*))?\}")


def expand_env(raw: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` so one YAML works locally and in Docker."""
    return _ENV_REF.sub(lambda m: os.environ.get(m["name"], m["default"] or ""), raw)


@lru_cache
def load_business_config(path: Path | None = None) -> BusinessConfig:
    target = path or get_settings().business_config
    raw = Path(target).read_text(encoding="utf-8")
    return BusinessConfig.model_validate(yaml.safe_load(expand_env(raw)))


def get_business() -> BusinessConfig:
    return load_business_config(get_settings().business_config)
