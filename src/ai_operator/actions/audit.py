"""Audit trail and owner notifications (both are plain database rows)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ai_operator.core.pii import mask_value
from ai_operator.db.models import Event, Notification


def audit(
    session: AsyncSession,
    kind: str,
    *,
    message_id: int | None = None,
    level: str = "info",
    **data: Any,
) -> None:
    """Record what happened and why. Personal data is masked before it is stored."""
    masked = {key: mask_value(key, value) for key, value in data.items()}
    session.add(Event(kind=kind, level=level, message_id=message_id, data=masked))


async def notify_owner(
    session: AsyncSession, *, kind: str, text: str, dedupe_key: str
) -> int | None:
    """Queue a notification for the owner. Returns its id, or None if it already exists."""
    result = await session.execute(
        insert(Notification)
        .values(kind=kind, text=text, dedupe_key=dedupe_key)
        .on_conflict_do_nothing(index_elements=[Notification.dedupe_key])
        .returning(Notification.id)
    )
    return result.scalar_one_or_none()


async def notification_id(session: AsyncSession, dedupe_key: str) -> int | None:
    return await session.scalar(
        select(Notification.id).where(Notification.dedupe_key == dedupe_key)
    )
