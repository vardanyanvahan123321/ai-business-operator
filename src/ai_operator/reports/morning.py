"""The owner's morning report: yesterday in numbers, and what needs attention today."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ai_operator.actions.audit import notify_owner
from ai_operator.config import BusinessConfig
from ai_operator.context import AppContext
from ai_operator.core.formatting import format_money
from ai_operator.core.i18n import t
from ai_operator.core.message import LeadStatus, MessageStatus
from ai_operator.db.models import DeadLetter, Handoff, Lead, Message, PriceReport, Report
from ai_operator.db.session import session_scope
from ai_operator.queue import jobs

MORNING = "morning"


async def collect(session: AsyncSession, business: BusinessConfig, today: date) -> dict[str, Any]:
    """Gather the report's numbers. "Yesterday" and "today" are in the business timezone."""
    tz = business.tz
    today_start = datetime.combine(today, time.min, tzinfo=tz)
    yesterday_start = today_start - timedelta(days=1)
    tomorrow_start = today_start + timedelta(days=1)

    by_status = dict(
        (
            await session.execute(
                select(Message.status, func.count())
                .where(Message.received_at >= yesterday_start, Message.received_at < today_start)
                .group_by(Message.status)
            )
        ).all()
    )
    lead_count, lead_sum = (
        await session.execute(
            select(func.count(), func.coalesce(func.sum(Lead.amount), 0)).where(
                Lead.created_at >= yesterday_start,
                Lead.created_at < today_start,
                Lead.status != LeadStatus.INCOMPLETE.value,
            )
        )
    ).one()
    visits_today = await session.scalar(
        select(func.count())
        .select_from(Lead)
        .where(
            Lead.scheduled_at >= today_start,
            Lead.scheduled_at < tomorrow_start,
            Lead.status.in_([LeadStatus.NEW.value, LeadStatus.CONFIRMED.value]),
        )
    )
    open_handoffs = await session.scalar(
        select(func.count()).select_from(Handoff).where(Handoff.status == "open")
    )
    incomplete = await session.scalar(
        select(func.count()).select_from(Lead).where(Lead.status == LeadStatus.INCOMPLETE.value)
    )
    dead = await session.scalar(
        select(func.count()).select_from(DeadLetter).where(DeadLetter.resolved_at.is_(None))
    )
    price_summary = await session.scalar(
        select(PriceReport.summary)
        .where(PriceReport.created_at >= yesterday_start)
        .order_by(PriceReport.id.desc())
        .limit(1)
    )
    return {
        "messages_total": sum(by_status.values()),
        "answered": by_status.get(MessageStatus.ANSWERED.value, 0),
        "handoff": by_status.get(MessageStatus.HANDOFF.value, 0)
        + by_status.get(MessageStatus.DEAD.value, 0),
        "spam": by_status.get(MessageStatus.IGNORED.value, 0),
        "leads": int(lead_count),
        "leads_amount": str(Decimal(lead_sum or 0)),
        "visits_today": int(visits_today or 0),
        "open_handoffs": int(open_handoffs or 0),
        "incomplete_leads": int(incomplete or 0),
        "dead_letters": int(dead or 0),
        "price_summary": price_summary,
    }


def render(data: dict[str, Any], business: BusinessConfig, today: date, lang: str) -> str:
    lines = [
        t(lang, "report.title", date=f"{today:%d.%m.%Y}"),
        "",
        t(
            lang,
            "report.messages",
            total=data["messages_total"],
            answered=data["answered"],
            handoff=data["handoff"],
            spam=data["spam"],
        ),
        t(
            lang,
            "report.leads",
            count=data["leads"],
            amount=format_money(Decimal(data["leads_amount"])),
            currency=business.currency,
        ),
        t(lang, "report.today", count=data["visits_today"]),
        "",
        t(lang, "report.attention"),
    ]
    attention = [
        ("report.attention.handoffs", data["open_handoffs"]),
        ("report.attention.incomplete", data["incomplete_leads"]),
        ("report.attention.dead", data["dead_letters"]),
    ]
    flagged = [t(lang, key, count=count) for key, count in attention if count]
    lines.extend(flagged or [t(lang, "report.attention.none")])
    lines.extend(
        ["", t(lang, "report.prices"), data["price_summary"] or t(lang, "report.prices.none")]
    )
    return "\n".join(lines)


async def generate_morning_report(ctx: AppContext, today: date | None = None) -> int:
    """Build, store and send today's report. Re-running the same day updates it in place."""
    business = ctx.business
    today = today or datetime.now(business.tz).date()
    lang = business.owner_language
    async with session_scope(ctx.sessions) as session:
        data = await collect(session, business, today)
        content = render(data, business, today, lang)
        report_id = await session.scalar(
            insert(Report)
            .values(kind=MORNING, report_date=today, language=lang, content=content, data=data)
            .on_conflict_do_update(
                constraint="uq_reports_kind_date", set_={"content": content, "data": data}
            )
            .returning(Report.id)
        )
        notification = await notify_owner(
            session, kind="report", dedupe_key=f"morning:{today.isoformat()}", text=content
        )
    if notification is not None:
        await jobs.safe_enqueue(ctx.queue, jobs.DELIVER_NOTIFICATION, notification)
    return int(report_id or 0)
