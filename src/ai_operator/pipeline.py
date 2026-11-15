"""The brain of the conveyor: one inbound message in, one decision out.

    claim -> understand (AI) -> decide (plain code) -> apply (one DB transaction)

Design rules that make this reliable:

* **The AI only describes, code decides.** The model returns validated data
  (category, confidence, extracted fields, grounded answer). Every business
  rule - when to hand off to a human, when a booking is complete - is ordinary,
  testable Python below.
* **No network calls inside a DB transaction.** Slow AI calls happen between
  two short transactions, so no row is locked while we wait for a vendor.
* **Everything is idempotent.** A crash at any point means the job runs again;
  re-running never duplicates a reply, a booking or a notification.
* **Side effects go through the outbox.** ``apply`` only writes rows (reply,
  notification, sheet sync) in the same transaction as the decision; separate
  jobs deliver them and retry on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from ai_operator.actions.audit import audit, notify_owner
from ai_operator.actions.invoice import ensure_invoice
from ai_operator.ai.guard import detect_injection, looks_leaky
from ai_operator.ai.rag import KnowledgeBase
from ai_operator.ai.schemas import (
    Analysis,
    AnalyzeRequest,
    Extraction,
    HistoryTurn,
    RetrievedChunk,
    ServiceOption,
    normalize_phone,
)
from ai_operator.context import AppContext
from ai_operator.core.formatting import format_datetime, format_money
from ai_operator.core.i18n import normalize_lang, t
from ai_operator.core.language import detect_language
from ai_operator.core.logging import get_logger
from ai_operator.core.message import (
    Category,
    HandoffReason,
    LeadStatus,
    MessageStatus,
)
from ai_operator.db.models import DeadLetter, Handoff, Lead, Message, Outbound, Setting
from ai_operator.db.session import session_scope
from ai_operator.queue import jobs

log = get_logger(__name__)

HISTORY_TURNS = 6
PENDING_LEAD_TTL = timedelta(hours=48)
AI_REPLY = "ai_reply"


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Everything the pipeline needs to know about a message, read in one go."""

    id: int
    channel: str
    sender_id: str
    sender_name: str | None
    subject: str | None
    text: str
    meta: dict[str, Any]
    received_at: datetime
    history: tuple[HistoryTurn, ...]
    pending_lead_id: int | None
    pending: Extraction | None
    preferred_language: str | None
    confidence_threshold: float


@dataclass(frozen=True, slots=True)
class LeadDraft:
    name: str | None
    phone: str | None
    email: str | None
    service_code: str | None
    service_text: str | None
    scheduled_at: datetime | None
    amount: Decimal | None
    missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing


@dataclass(frozen=True, slots=True)
class Decision:
    status: MessageStatus
    reply: str | None = None
    handoff: HandoffReason | None = None
    lead: LeadDraft | None = None
    sources: tuple[RetrievedChunk, ...] = field(default=())


def preference_key(channel: str, sender_id: str) -> str:
    return f"lang:{channel}:{sender_id}"[:64]


class Pipeline:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------ run

    async def process(self, message_id: int) -> MessageStatus | None:
        """Process one message. Safe to call any number of times."""
        snapshot = await self._claim(message_id)
        if snapshot is None:
            return None
        patterns = detect_injection(f"{snapshot.subject or ''}\n{snapshot.text}")
        analysis = await self.ctx.provider.analyze(self._request(snapshot))
        language = self._language(snapshot, analysis)
        decision = await self._decide(snapshot, analysis, language, patterns)
        queued = await self._apply(snapshot, analysis, language, decision, patterns)
        for job, ref_id in queued:
            await jobs.safe_enqueue(self.ctx.queue, job, ref_id)
        log.info(
            "pipeline.done",
            message_id=message_id,
            category=analysis.category.value,
            confidence=analysis.confidence,
            status=decision.status.value,
            handoff=decision.handoff.value if decision.handoff else None,
        )
        return decision.status

    # ---------------------------------------------------------------- claim

    async def _claim(self, message_id: int) -> Snapshot | None:
        async with session_scope(self.ctx.sessions) as session:
            message = await session.get(Message, message_id, with_for_update=True)
            if message is None or MessageStatus(message.status).is_final:
                return None
            message.status = MessageStatus.PROCESSING.value
            message.attempts += 1
            message.processing_started_at = datetime.now(UTC)

            earlier = (
                await session.scalars(
                    select(Message)
                    .where(
                        Message.channel == message.channel,
                        Message.sender_id == message.sender_id,
                        Message.id < message.id,
                    )
                    .order_by(Message.id.desc())
                    .limit(HISTORY_TURNS)
                )
            ).all()
            history: list[HistoryTurn] = []
            for old in reversed(earlier):
                history.append(HistoryTurn("customer", old.text))
                reply = await session.scalar(
                    select(Outbound.text).where(Outbound.message_id == old.id).limit(1)
                )
                if reply:
                    history.append(HistoryTurn("business", reply))

            lead = await session.scalar(
                select(Lead)
                .where(
                    Lead.channel == message.channel,
                    Lead.sender_id == message.sender_id,
                    Lead.status == LeadStatus.INCOMPLETE.value,
                    Lead.updated_at > datetime.now(UTC) - PENDING_LEAD_TTL,
                )
                .order_by(Lead.id.desc())
                .limit(1)
            )
            pending = None
            if lead is not None:
                local = (
                    lead.scheduled_at.astimezone(self.ctx.business.tz)
                    if lead.scheduled_at
                    else None
                )
                pending = Extraction(
                    name=lead.name,
                    phone=lead.phone,
                    service_code=lead.service_code,
                    service_text=lead.service_text,
                    scheduled_at=f"{local:%Y-%m-%dT%H:%M}" if local else None,
                )

            preference = await session.get(
                Setting, preference_key(message.channel, message.sender_id)
            )
            thresholds = await session.get(Setting, "thresholds")
            threshold = self.ctx.settings.confidence_threshold
            if thresholds is not None:
                threshold = float(thresholds.value.get("confidence_threshold", threshold))

            return Snapshot(
                id=message.id,
                channel=message.channel,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                subject=message.subject,
                text=message.text,
                meta=dict(message.meta or {}),
                received_at=message.received_at,
                history=tuple(history),
                pending_lead_id=lead.id if lead is not None else None,
                pending=pending,
                preferred_language=preference.value.get("lang") if preference else None,
                confidence_threshold=threshold,
            )

    # ----------------------------------------------------------- understand

    def _request(self, snapshot: Snapshot) -> AnalyzeRequest:
        business = self.ctx.business
        return AnalyzeRequest(
            text=snapshot.text,
            subject=snapshot.subject,
            # Relative dates ("tomorrow") are resolved against the moment the customer
            # wrote, not the moment we got round to processing. It also keeps the
            # prompt identical across retries, so the response cache can hit.
            now=snapshot.received_at.astimezone(business.tz),
            services=tuple(
                ServiceOption(
                    code=s.code,
                    names=(s.name.en, s.name.ru, s.name.hy),
                    keywords=tuple(s.keywords),
                )
                for s in business.services
            ),
            history=snapshot.history,
            pending=snapshot.pending,
        )

    def _language(self, snapshot: Snapshot, analysis: Analysis) -> str:
        if snapshot.preferred_language:
            return normalize_lang(snapshot.preferred_language)
        by_script = detect_language(snapshot.text, default="en")
        # Scripts are unambiguous for Armenian and Russian. For Latin text we trust
        # the model: it may be English, or Armenian/Russian typed in Latin letters.
        return by_script if by_script != "en" else normalize_lang(analysis.language)

    # --------------------------------------------------------------- decide

    async def _decide(
        self, snapshot: Snapshot, analysis: Analysis, language: str, patterns: list[str]
    ) -> Decision:
        if patterns or analysis.injection_suspected:
            return self._handoff(HandoffReason.INJECTION, language)
        if analysis.category is Category.COMPLAINT:
            return self._handoff(HandoffReason.COMPLAINT, language)
        if analysis.confidence < snapshot.confidence_threshold:
            return self._handoff(HandoffReason.LOW_CONFIDENCE, language)
        if analysis.category is Category.SPAM:
            return Decision(status=MessageStatus.IGNORED)
        if analysis.category is Category.QUESTION:
            return await self._answer(snapshot, language)
        if analysis.category is Category.ORDER:
            return self._book(snapshot, analysis, language)
        return Decision(status=MessageStatus.ANSWERED, reply=t(language, "reply.other"))

    @staticmethod
    def _handoff(reason: HandoffReason, language: str) -> Decision:
        key = {
            HandoffReason.COMPLAINT: "reply.handoff.complaint",
            HandoffReason.NO_ANSWER: "reply.handoff.no_answer",
        }.get(reason, "reply.handoff.generic")
        return Decision(status=MessageStatus.HANDOFF, handoff=reason, reply=t(language, key))

    async def _answer(self, snapshot: Snapshot, language: str) -> Decision:
        async with self.ctx.sessions() as session:
            chunks = await KnowledgeBase(session, self.ctx.provider).search(
                snapshot.text, language=language, top_k=self.ctx.settings.rag_top_k
            )
        if not chunks:
            return self._handoff(HandoffReason.NO_ANSWER, language)

        answer = await self.ctx.provider.answer(snapshot.text, language, chunks)
        cited = set(answer.source_ids)
        # The model may only cite what we retrieved; anything else is discarded.
        sources = tuple(c for c in chunks if c.id in cited)
        text = answer.answer.strip()
        if not answer.found or not text or not sources or looks_leaky(text):
            return self._handoff(HandoffReason.NO_ANSWER, language)

        titles = ", ".join(dict.fromkeys(c.title for c in sources))
        reply = f"{text}\n\n{t(language, 'reply.source', titles=titles)}"
        return Decision(status=MessageStatus.ANSWERED, reply=reply, sources=sources)

    def _book(self, snapshot: Snapshot, analysis: Analysis, language: str) -> Decision:
        business = self.ctx.business
        data = analysis.extraction.merged_over(snapshot.pending)
        contact = str(snapshot.meta.get("contact") or "")
        phone = normalize_phone(data.phone) or normalize_phone(None if "@" in contact else contact)
        email = snapshot.sender_id if snapshot.channel == "email" else None
        if email is None and "@" in contact:
            email = contact.strip()
        service = business.service(data.service_code)
        when = data.parsed_datetime()
        if when is not None and when.tzinfo is None:
            when = when.replace(tzinfo=business.tz)

        missing: list[str] = []
        if service is None:
            missing.append("service")
        if when is None:
            missing.append("datetime")
        if phone is None and email is None:
            missing.append("phone")

        # The price always comes from our catalogue. A sum written by the customer
        # is kept in the analysis for the owner to see, but never becomes an invoice.
        amount = service.price if service is not None else None
        draft = LeadDraft(
            name=data.name or snapshot.sender_name,
            phone=phone,
            email=email,
            service_code=service.code if service else None,
            service_text=data.service_text,
            scheduled_at=when,
            amount=amount,
            missing=tuple(missing),
        )
        if missing:
            fields = ", ".join(t(language, f"field.{name}") for name in missing)
            reply = t(language, "reply.order.missing", fields=fields)
            return Decision(status=MessageStatus.ANSWERED, reply=reply, lead=draft)

        assert service is not None
        lines = [
            t(language, "reply.order.confirmed", name=f", {draft.name}" if draft.name else ""),
            t(language, "reply.order.service", service=service.name.get(language)),
            t(language, "reply.order.date", date=format_datetime(when, language, business.tz)),
        ]
        if amount is not None:
            key = "reply.order.price_from" if service.price_from else "reply.order.price"
            lines.append(t(language, key, price=format_money(amount), currency=business.currency))
        if phone:
            lines.append(t(language, "reply.order.callback", phone=phone))
        if amount is not None:
            lines.append(t(language, "reply.order.invoice"))
        return Decision(status=MessageStatus.ANSWERED, reply="\n".join(lines), lead=draft)

    # ---------------------------------------------------------------- apply

    async def _apply(
        self,
        snapshot: Snapshot,
        analysis: Analysis,
        language: str,
        decision: Decision,
        patterns: list[str],
    ) -> list[tuple[str, int]]:
        """Persist the whole outcome atomically; return the jobs to enqueue after commit."""
        business = self.ctx.business
        owner_lang = business.owner_language
        queued: list[tuple[str, int]] = []

        async with session_scope(self.ctx.sessions) as session:
            message = await session.get(Message, snapshot.id, with_for_update=True)
            if message is None or MessageStatus(message.status).is_final:
                return []  # another worker finished it first

            message.status = decision.status.value
            message.language = language
            message.category = analysis.category.value
            message.confidence = analysis.confidence
            message.injection_suspected = bool(patterns) or analysis.injection_suspected
            message.analysis = {
                **analysis.model_dump(mode="json"),
                "injection_patterns": patterns,
                "sources": [
                    {
                        "chunk_id": c.id,
                        "document_id": c.document_id,
                        "title": c.title,
                        "score": c.score,
                    }
                    for c in decision.sources
                ],
            }
            message.error = None
            message.processed_at = datetime.now(UTC)

            if decision.handoff is not None:
                session.add(
                    Handoff(
                        message_id=message.id,
                        reason=decision.handoff.value,
                        summary=analysis.summary,
                    )
                )
                notification = await notify_owner(
                    session,
                    kind="handoff",
                    dedupe_key=f"handoff:{message.id}",
                    text=t(
                        owner_lang,
                        "notify.handoff",
                        reason=t(owner_lang, f"reason.{decision.handoff.value}"),
                        id=message.id,
                        channel=message.channel,
                        summary=analysis.summary,
                        url=f"{self.ctx.settings.public_base_url}/panel/inbox/{message.id}",
                    ),
                )
                if notification is not None:
                    queued.append((jobs.DELIVER_NOTIFICATION, notification))

            attachments: list[dict[str, Any]] = []
            if decision.lead is not None:
                lead = await self._save_lead(session, snapshot, decision.lead, language)
                if decision.lead.complete:
                    invoice = await ensure_invoice(session, lead)
                    if invoice is not None:
                        attachments.append({"kind": "invoice", "invoice_id": invoice.id})
                    queued.append((jobs.SYNC_LEAD, lead.id))
                    service = business.service(lead.service_code)
                    notification = await notify_owner(
                        session,
                        kind="lead",
                        dedupe_key=f"lead:{lead.id}",
                        text=t(
                            owner_lang,
                            "notify.lead",
                            id=lead.id,
                            channel=lead.channel,
                            service=service.name.get(owner_lang) if service else "-",
                            date=format_datetime(lead.scheduled_at, owner_lang, business.tz),
                            amount=f"{format_money(lead.amount)} {lead.currency}",
                            name=lead.name or t(owner_lang, "common.not_set"),
                            phone=lead.phone or lead.email or t(owner_lang, "common.not_set"),
                        ),
                    )
                    if notification is not None:
                        queued.append((jobs.DELIVER_NOTIFICATION, notification))

            if decision.reply is not None:
                subject = None
                if message.channel == "email":
                    subject = (
                        f"Re: {message.subject}"
                        if message.subject
                        else t(language, "reply.email.subject", company=business.name.get(language))
                    )
                outbound_id = await session.scalar(
                    insert(Outbound)
                    .values(
                        message_id=message.id,
                        dedupe_key=f"msg:{message.id}:{AI_REPLY}",
                        channel=message.channel,
                        recipient=message.sender_id,
                        subject=subject,
                        text=decision.reply,
                        attachments=attachments,
                        meta={"in_reply_to": snapshot.meta.get("message_id")},
                        kind=AI_REPLY,
                    )
                    .on_conflict_do_nothing(index_elements=[Outbound.dedupe_key])
                    .returning(Outbound.id)
                )
                if outbound_id is not None:
                    queued.append((jobs.SEND_OUTBOUND, outbound_id))

            audit(
                session,
                "pipeline.decision",
                message_id=message.id,
                category=analysis.category.value,
                confidence=analysis.confidence,
                language=language,
                status=decision.status.value,
                handoff=decision.handoff.value if decision.handoff else None,
                provider=self.ctx.provider.name,
                sources=[c.title for c in decision.sources],
                injection_patterns=patterns,
            )
        return queued

    @staticmethod
    async def _save_lead(session: Any, snapshot: Snapshot, draft: LeadDraft, language: str) -> Lead:
        lead = None
        if snapshot.pending_lead_id is not None:
            lead = await session.get(Lead, snapshot.pending_lead_id, with_for_update=True)
        if lead is None:
            lead = Lead(
                channel=snapshot.channel, sender_id=snapshot.sender_id, message_id=snapshot.id
            )
            session.add(lead)
        lead.name = draft.name
        lead.phone = draft.phone
        lead.email = draft.email
        lead.service_code = draft.service_code
        lead.service_text = draft.service_text
        lead.scheduled_at = draft.scheduled_at
        lead.amount = draft.amount
        lead.language = language
        lead.status = (LeadStatus.NEW if draft.complete else LeadStatus.INCOMPLETE).value
        await session.flush()
        return lead

    # ------------------------------------------------------------- failures

    async def record_failure(self, message_id: int, error: str) -> None:
        """A retry will follow: release the message so it is not stuck in 'processing'."""
        async with session_scope(self.ctx.sessions) as session:
            message = await session.get(Message, message_id, with_for_update=True)
            if message is None or MessageStatus(message.status).is_final:
                return
            message.status = MessageStatus.RECEIVED.value
            message.error = error[:2000]
            audit(
                session, "pipeline.retry", message_id=message_id, level="warning", error=error[:500]
            )

    async def mark_dead(self, message_id: int, error: str, attempts: int) -> None:
        """All retries are spent. The customer must not be forgotten: a human takes over."""
        queued: list[tuple[str, int]] = []
        async with session_scope(self.ctx.sessions) as session:
            message = await session.get(Message, message_id, with_for_update=True)
            if message is None or MessageStatus(message.status).is_final:
                return
            message.status = MessageStatus.DEAD.value
            message.error = error[:2000]
            message.processed_at = datetime.now(UTC)
            session.add(
                Handoff(
                    message_id=message_id, reason=HandoffReason.FAILED.value, summary=error[:500]
                )
            )
            session.add(
                DeadLetter(
                    job=jobs.PROCESS_MESSAGE,
                    ref_id=message_id,
                    error=error[:2000],
                    attempts=attempts,
                )
            )
            owner_lang = self.ctx.business.owner_language
            notification = await notify_owner(
                session,
                kind="dead_letter",
                dedupe_key=f"dead:{jobs.PROCESS_MESSAGE}:{message_id}",
                text=t(
                    owner_lang,
                    "notify.dead",
                    job=jobs.PROCESS_MESSAGE,
                    attempts=attempts,
                    ref=message_id,
                ),
            )
            if notification is not None:
                queued.append((jobs.DELIVER_NOTIFICATION, notification))
            audit(session, "pipeline.dead", message_id=message_id, level="error", error=error[:500])
        for job, ref_id in queued:
            await jobs.safe_enqueue(self.ctx.queue, job, ref_id)
