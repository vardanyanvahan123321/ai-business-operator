"""Operator actions from the panel: reply as a human, retry a failed job."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import delete

from ai_operator.actions.audit import audit
from ai_operator.context import AppContext
from ai_operator.core.i18n import t
from ai_operator.core.message import DeliveryStatus, HandoffReason, MessageStatus
from ai_operator.db.models import DeadLetter, Handoff, Lead, Message, Notification, Outbound
from ai_operator.db.session import session_scope
from ai_operator.queue import jobs

HUMAN_REPLY = "human_reply"


async def resolve_handoff(ctx: AppContext, handoff_id: int, reply: str | None) -> bool:
    """Close a handoff; with ``reply`` the text is sent to the customer through the outbox."""
    outbound_id: int | None = None
    async with session_scope(ctx.sessions) as session:
        handoff = await session.get(Handoff, handoff_id, with_for_update=True)
        if handoff is None or handoff.status != "open":
            return False
        message = await session.get(Message, handoff.message_id)
        assert message is not None
        text = (reply or "").strip()
        if text:
            language = message.language or ctx.business.default_language
            subject = None
            if message.channel == "email":
                subject = (
                    f"Re: {message.subject}"
                    if message.subject
                    else t(language, "reply.email.subject", company=ctx.business.name.get(language))
                )
            outbound = Outbound(
                message_id=message.id,
                dedupe_key=f"human:{handoff.id}:{uuid.uuid4().hex[:12]}",
                channel=message.channel,
                recipient=message.sender_id,
                subject=subject,
                text=text,
                meta={"in_reply_to": (message.meta or {}).get("message_id")},
                kind=HUMAN_REPLY,
            )
            session.add(outbound)
            await session.flush()
            outbound_id = outbound.id
        handoff.status = "resolved"
        handoff.resolution = text or None
        handoff.resolved_at = datetime.now(UTC)
        audit(session, "handoff.resolved", message_id=message.id, replied=bool(text))
    if outbound_id is not None:
        await jobs.safe_enqueue(ctx.queue, jobs.SEND_OUTBOUND, outbound_id)
    return True


async def retry_dead_letter(ctx: AppContext, dead_letter_id: int) -> bool:
    """Give a buried job another life: reset its row and enqueue it again."""
    async with session_scope(ctx.sessions) as session:
        dead = await session.get(DeadLetter, dead_letter_id, with_for_update=True)
        if dead is None or dead.resolved_at is not None or dead.ref_id is None:
            return False
        if dead.job == jobs.PROCESS_MESSAGE:
            message = await session.get(Message, dead.ref_id)
            if message is not None:
                message.status = MessageStatus.RECEIVED.value
                message.attempts = 0
                message.error = None
                await session.execute(
                    delete(Handoff).where(
                        Handoff.message_id == message.id,
                        Handoff.reason == HandoffReason.FAILED.value,
                    )
                )
        elif dead.job == jobs.SEND_OUTBOUND:
            outbound = await session.get(Outbound, dead.ref_id)
            if outbound is not None:
                outbound.status = DeliveryStatus.PENDING.value
                outbound.error = None
        elif dead.job == jobs.DELIVER_NOTIFICATION:
            notification = await session.get(Notification, dead.ref_id)
            if notification is not None:
                notification.status = DeliveryStatus.PENDING.value
                notification.error = None
        elif dead.job == jobs.SYNC_LEAD:
            lead = await session.get(Lead, dead.ref_id)
            if lead is not None:
                lead.sheet_error = None
        else:
            return False
        dead.resolved_at = datetime.now(UTC)
        job, ref_id = dead.job, dead.ref_id
        audit(session, "dead_letter.retried", job=job, ref_id=ref_id)
    await jobs.safe_enqueue(ctx.queue, job, ref_id)
    return True
