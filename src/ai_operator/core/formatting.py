"""Human-readable money and dates (same numeric style in all three languages)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from ai_operator.core.i18n import t


def format_money(amount: Decimal | float | int | None) -> str:
    if amount is None:
        return "-"
    return f"{Decimal(amount):,.0f}".replace(",", " ")


def format_datetime(value: datetime | None, lang: str, tz: object | None = None) -> str:
    """``16.11.2026 10:00``; midnight means the customer gave a date without a time."""
    if value is None:
        return t(lang, "common.not_set")
    local = value.astimezone(tz) if tz is not None and value.tzinfo else value  # type: ignore[arg-type]
    if local.hour == 0 and local.minute == 0:
        return f"{local:%d.%m.%Y} ({t(lang, 'common.time_not_set')})"
    return f"{local:%d.%m.%Y %H:%M}"
