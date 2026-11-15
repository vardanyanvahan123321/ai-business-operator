"""Worker jobs.

Every job follows the same contract:

* it is **idempotent** - it first checks whether the work is already done;
* on failure it is retried with exponential backoff;
* when retries run out it is **buried**, not dropped: a ``dead_letters`` row is
  written, the owner is notified, and the panel offers a "retry" button.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from arq import Retry
from sqlalchemy import select

from ai_operator.actions.audit import audit, notify_owner
from ai_operator.actions.invoice import ensure_invoice_pdf
from ai_operator.actions.sheets import lead_to_row
from ai_operator.context import AppContext
from ai_operator.core.i18n import t
from ai_operator.core.logging import get_logger
from ai_operator.core.message import (
    Attachment,
    DeliveryStatus,
    LeadStatus,
    MessageStatus,
    OutboundMessage,
)
from ai_operator.db.models import DeadLetter, Invoice, Lead, Message, Notification, Outbound
from ai_operator.db.session import session_scope
from ai_operator.pipeline import Pipeline
from ai_operator.pricing.monitor import run_price_check
from ai_operator.queue import jobs
from ai_operator.queue.ingest import ingest
from ai_operator.reports.morning import generate_morning_report

log = get_logger(__name__)
MAX_BACKOFF_SECONDS = 300


def backoff(base: int, attempt: int) -> float:
    """Exponential backoff with jitter: base, 2*base, 4*base ... capped."""
    capped: float = min(MAX_BACKOFF_SECONDS, base * 2 ** (attempt - 1))
    return capped + random.uniform(0, 1)  # noqa: S311


def _app(ctx: dict[str, Any]) -> AppContext:
    app: AppContext = ctx["app"]
    return app


async def _retry_or_bury(
    ctx: dict[str, Any],
    job: str,
    ref_id: int,
    exc: Exception,
    bury: Callable[[str, int], Awaitable[None]],
) -> None:
    app = _app(ctx)
    attempt = int(ctx.get("job_try", 1))
    error = f"{type(exc).__name__}: {exc}"
    if attempt >= app.settings.max_attempts:
        log.error("job.dead", job=job, ref_id=ref_id, attempts=attempt, error=error)
        await bury(error, attempt)
        return
    delay = backoff(app.settings.retry_base_seconds, attempt)
    log.warning("job.retry", job=job, ref_id=ref_id, attempt=attempt, retry_in=round(delay, 1))
    raise Retry(defer=delay) from exc


async def _bury(app: AppContext, job: str, ref_id: int, error: str, attempts: int) -> None:
    """Write the dead letter and tell the owner."""
    async with session_scope(app.sessions) as session:
        session.add(DeadLetter(job=job, ref_id=ref_id, error=error[:2000], attempts=attempts))
        lang = app.business.owner_language
        notification = await notify_owner(
            session,
            kind="dead_letter",
            dedupe_key=f"dead:{job}:{ref_id}",
            text=t(lang, "notify.dead", job=job, attempts=attempts, ref=ref_id),
        )
        audit(session, "job.dead", level="error", job=job, ref_id=ref_id, error=error[:500])
    if notification is not None and job != jobs.DELIVER_NOTIFICATION:
        await jobs.safe_enqueue(app.queue, jobs.DELIVER_NOTIFICATION, notification)


# --------------------------------------------------------------------------- jobs


async def process_message(ctx: dict[str, Any], message_id: int) -> str:
    app = _app(ctx)
    pipeline = Pipeline(app)
    try:
        status = await pipeline.process(message_id)
    except Exception as exc:
        attempt = int(ctx.get("job_try", 1))
        error = f"{type(exc).__name__}: {exc}"
        if attempt >= app.settings.max_attempts:
            log.error("job.dead", job=jobs.PROCESS_MESSAGE, ref_id=message_id, error=error)
            await pipeline.mark_dead(message_id, error, attempt)
            return "dead"
        await pipeline.record_failure(message_id, error)
        delay = backoff(app.settings.retry_base_seconds, attempt)
        log.warning("job.retry", job=jobs.PROCESS_MESSAGE, ref_id=message_id, attempt=attempt)
        raise Retry(defer=delay) from exc
    return status.value if status else "skipped"


async def _load_outbound(app: AppContext, outbound_id: int) -> OutboundMessage | None:
    async with app.sessions() as session:
        row = await session.get(Outbound, outbound_id)
        if row is None or row.status != DeliveryStatus.PENDING.value:
            return None
        attachments: list[Attachment] = []
        for item in row.attachments or []:
            if item.get("kind") != "invoice":
                continue
            invoice = await session.get(Invoice, item["invoice_id"])
            lead = await session.get(Lead, invoice.lead_id) if invoice else None
            if invoice is None or lead is None:
                continue
            path = await ensure_invoice_pdf(
                invoice, lead, app.business, app.settings.invoices_dir, demo=app.settings.demo_mode
            )
            attachments.append(Attachment(filename=f"{invoice.number}.pdf", path=str(path)))
        return OutboundMessage(
            channel=row.channel,
            recipient=row.recipient,
            subject=row.subject,
            text=row.text,
            attachments=attachments,
            meta=dict(row.meta or {}),
        )


async def send_outbound(ctx: dict[str, Any], outbound_id: int) -> str:
    """Deliver one outbox row to the customer (at-least-once)."""
    app = _app(ctx)
    try:
        message = await _load_outbound(app, outbound_id)
        if message is None:
            return "skipped"
        await app.channels.get(message.channel).send(message)
    except Exception as exc:

        async def bury(error: str, attempts: int) -> None:
            async with session_scope(app.sessions) as session:
                row = await session.get(Outbound, outbound_id)
                if row is not None:
                    row.status = DeliveryStatus.FAILED.value
                    row.error = error[:2000]
                    row.attempts = attempts
            await _bury(app, jobs.SEND_OUTBOUND, outbound_id, error, attempts)

        await _retry_or_bury(ctx, jobs.SEND_OUTBOUND, outbound_id, exc, bury)
        return "dead"

    async with session_scope(app.sessions) as session:
        row = await session.get(Outbound, outbound_id)
        if row is not None:
            row.status = DeliveryStatus.SENT.value
            row.sent_at = datetime.now(UTC)
            row.attempts = int(ctx.get("job_try", 1))
            row.error = None
            audit(session, "outbound.sent", message_id=row.message_id, channel=row.channel)
    return "sent"


async def deliver_notification(ctx: dict[str, Any], notification_id: int) -> str:
    """Send a notification to the owner's Telegram; without Telegram it lives in the panel."""
    app = _app(ctx)
    async with app.sessions() as session:
        row = await session.get(Notification, notification_id)
        if row is None or row.status != DeliveryStatus.PENDING.value:
            return "skipped"
        text = row.text

    via = "panel"
    owner_chat = app.settings.telegram_owner_chat_id
    try:
        if owner_chat and "telegram" in app.channels:
            await app.channels.get("telegram").send(
                OutboundMessage(channel="telegram", recipient=owner_chat, text=text)
            )
            via = "telegram"
    except Exception as exc:

        async def bury(error: str, attempts: int) -> None:
            async with session_scope(app.sessions) as session:
                failed = await session.get(Notification, notification_id)
                if failed is not None:
                    failed.status = DeliveryStatus.FAILED.value
                    failed.error = error[:2000]
            await _bury(app, jobs.DELIVER_NOTIFICATION, notification_id, error, attempts)

        await _retry_or_bury(ctx, jobs.DELIVER_NOTIFICATION, notification_id, exc, bury)
        return "dead"

    async with session_scope(app.sessions) as session:
        row = await session.get(Notification, notification_id)
        if row is not None:
            row.status = DeliveryStatus.SENT.value
            row.delivered_via = via
            row.sent_at = datetime.now(UTC)
    return via


async def sync_lead(ctx: dict[str, Any], lead_id: int) -> str:
    """Copy a booking into the owner's spreadsheet (upsert by lead id)."""
    app = _app(ctx)
    async with app.sessions() as session:
        lead = await session.get(Lead, lead_id)
        if lead is None or lead.status == LeadStatus.INCOMPLETE.value:
            return "skipped"
        row = lead_to_row(lead, app.business)
    try:
        await app.lead_sink.upsert(row)
    except Exception as exc:

        async def bury(error: str, attempts: int) -> None:
            async with session_scope(app.sessions) as session:
                failed = await session.get(Lead, lead_id)
                if failed is not None:
                    failed.sheet_error = error[:2000]
            await _bury(app, jobs.SYNC_LEAD, lead_id, error, attempts)

        await _retry_or_bury(ctx, jobs.SYNC_LEAD, lead_id, exc, bury)
        return "dead"

    async with session_scope(app.sessions) as session:
        lead = await session.get(Lead, lead_id)
        if lead is not None:
            lead.sheet_synced_at = datetime.now(UTC)
            lead.sheet_error = None
    return app.lead_sink.name


async def poll_email(ctx: dict[str, Any]) -> int:
    """Pull new mail into the pipeline. Mail is acknowledged only once it is stored."""
    app = _app(ctx)
    inbox = app.extras.get("inbox")
    if inbox is None:
        return 0
    stored: list[str] = []
    created_count = 0
    for ref, inbound in await inbox.fetch():
        if inbound is not None:
            _, created = await ingest(app, inbound)
            created_count += int(created)
        stored.append(ref)
    await inbox.ack(stored)
    return created_count


async def scrape_prices(ctx: dict[str, Any]) -> int | None:
    return await run_price_check(_app(ctx))


async def morning_report(ctx: dict[str, Any]) -> int:
    return await generate_morning_report(_app(ctx))


async def sweep(ctx: dict[str, Any]) -> dict[str, int]:
    """Safety net: find rows whose job was lost and enqueue them again.

    Covers the gap between "committed to PostgreSQL" and "enqueued in Redis"
    (Redis down, process killed) and workers that died mid-job.
    """
    app = _app(ctx)
    cutoff = datetime.now(UTC) - timedelta(seconds=app.settings.stuck_after_seconds)
    found: dict[str, list[int]] = {}
    async with app.sessions() as session:
        found[jobs.PROCESS_MESSAGE] = list(
            await session.scalars(
                select(Message.id).where(
                    (
                        (Message.status == MessageStatus.RECEIVED.value)
                        & (Message.received_at < cutoff)
                    )
                    | (
                        (Message.status == MessageStatus.PROCESSING.value)
                        & (Message.processing_started_at < cutoff)
                    )
                )
            )
        )
        found[jobs.SEND_OUTBOUND] = list(
            await session.scalars(
                select(Outbound.id).where(
                    Outbound.status == DeliveryStatus.PENDING.value, Outbound.created_at < cutoff
                )
            )
        )
        found[jobs.DELIVER_NOTIFICATION] = list(
            await session.scalars(
                select(Notification.id).where(
                    Notification.status == DeliveryStatus.PENDING.value,
                    Notification.created_at < cutoff,
                )
            )
        )
        found[jobs.SYNC_LEAD] = list(
            await session.scalars(
                select(Lead.id).where(
                    Lead.status != LeadStatus.INCOMPLETE.value,
                    Lead.sheet_synced_at.is_(None),
                    Lead.sheet_error.is_(None),
                    Lead.updated_at < cutoff,
                )
            )
        )
    requeued = dict.fromkeys(found, 0)
    for job, ids in found.items():
        for ref_id in ids:
            if await app.queue.enqueue(job, ref_id):
                requeued[job] += 1
    if any(requeued.values()):
        log.warning("sweep.requeued", **requeued)
    return requeued
