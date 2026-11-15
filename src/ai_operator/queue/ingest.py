"""Entry point of the conveyor: store first, then queue.

A message is written to PostgreSQL *before* anything else happens to it. From
that moment it cannot be lost: if Redis or the worker is down, the row waits
with status ``received`` and the sweeper enqueues it later.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from ai_operator.actions.audit import audit
from ai_operator.context import AppContext
from ai_operator.core.message import InboundMessage
from ai_operator.db.models import Message
from ai_operator.db.session import session_scope
from ai_operator.queue import jobs

MAX_TEXT_CHARS = 10_000


async def ingest(ctx: AppContext, inbound: InboundMessage) -> tuple[int, bool]:
    """Store an inbound message and queue it. Returns ``(message_id, created)``.

    Idempotent: channels redeliver (Telegram after a timeout, IMAP after a
    restart), and the unique ``(channel, external_id)`` key turns a duplicate
    into a no-op.
    """
    async with session_scope(ctx.sessions) as session:
        message_id = await session.scalar(
            insert(Message)
            .values(
                channel=inbound.channel,
                external_id=inbound.external_id[:255],
                sender_id=inbound.sender_id[:255],
                sender_name=(inbound.sender_name or None) and inbound.sender_name[:255],
                subject=(inbound.subject or None) and inbound.subject[:500],
                text=inbound.text[:MAX_TEXT_CHARS],
                meta=inbound.meta,
                received_at=inbound.received_at,
            )
            .on_conflict_do_nothing(constraint="uq_messages_channel_external")
            .returning(Message.id)
        )
        if message_id is None:
            existing = await session.scalar(
                select(Message.id).where(
                    Message.channel == inbound.channel,
                    Message.external_id == inbound.external_id[:255],
                )
            )
            assert existing is not None
            return existing, False
        audit(session, "message.received", message_id=message_id, channel=inbound.channel)

    await jobs.safe_enqueue(ctx.queue, jobs.PROCESS_MESSAGE, message_id)
    return message_id, True
