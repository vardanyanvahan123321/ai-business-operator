"""Masking of personal data before it reaches logs.

The database keeps the real values (the business needs the phone number to call
the customer back). Logs do not: they are shipped, tailed and pasted into
tickets, so phones and emails are masked there by default.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any

_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
# A phone is 9-15 digits, optionally with +, spaces, dashes or brackets between them.
_PHONE = re.compile(r"(?<![\w.])\+?\d(?:[\s\-()]{0,2}\d){8,14}(?![\w])")

# Keys whose values are personal by nature, regardless of their shape.
_SENSITIVE_KEYS = frozenset(
    {"name", "sender_name", "customer_name", "password", "token", "api_key"}
)


def _mask_phone(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = [c for c in raw if c.isdigit()]
    keep = "".join(digits[-2:])
    prefix = "+" if raw.startswith("+") else ""
    return f"{prefix}{'*' * (len(digits) - 2)}{keep}"


def mask_pii(text: str) -> str:
    """Mask emails and phone numbers inside free text."""
    masked = _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)
    return _PHONE.sub(_mask_phone, masked)


def mask_value(key: str, value: Any) -> Any:
    """Recursively mask a structured log value."""
    if key in _SENSITIVE_KEYS and isinstance(value, str) and value:
        return value[0] + "***"
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {k: mask_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [mask_value(key, v) for v in value]
    return value


def mask_event(_logger: Any, _method: str, event_dict: MutableMapping[str, Any]) -> dict[str, Any]:
    """structlog processor: nothing personal leaves the process unmasked."""
    return {k: mask_value(k, v) for k, v in event_dict.items()}
