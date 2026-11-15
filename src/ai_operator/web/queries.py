"""Read-only queries behind the dashboard and the list pages."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_operator.config import BusinessConfig
from ai_operator.core.message import LeadStatus, MessageStatus
from ai_operator.db.models import DeadLetter, Handoff, Lead, Message, PriceSnapshot

OUTCOMES = (MessageStatus.ANSWERED.value, MessageStatus.HANDOFF.value, MessageStatus.IGNORED.value)


def _local_day(column: Any, business: BusinessConfig) -> Any:
    return func.date(func.timezone(business.timezone, column))


def _days(end: date, count: int) -> list[date]:
    return [end - timedelta(days=offset) for offset in range(count - 1, -1, -1)]


async def kpis(session: AsyncSession, business: BusinessConfig, today: date) -> dict[str, Any]:
    tz = business.tz
    today_start = datetime.combine(today, time.min, tzinfo=tz)
    week_start = today_start - timedelta(days=6)

    by_status = dict(
        (
            await session.execute(
                select(Message.status, func.count())
                .where(Message.received_at >= week_start)
                .group_by(Message.status)
            )
        ).all()
    )
    answered = by_status.get(MessageStatus.ANSWERED.value, 0)
    escalated = by_status.get(MessageStatus.HANDOFF.value, 0) + by_status.get(
        MessageStatus.DEAD.value, 0
    )
    handled = answered + escalated
    today_count = await session.scalar(
        select(func.count()).select_from(Message).where(Message.received_at >= today_start)
    )
    lead_count, lead_sum = (
        await session.execute(
            select(func.count(), func.coalesce(func.sum(Lead.amount), 0)).where(
                Lead.created_at >= week_start, Lead.status != LeadStatus.INCOMPLETE.value
            )
        )
    ).one()
    open_handoffs = await session.scalar(
        select(func.count()).select_from(Handoff).where(Handoff.status == "open")
    )
    dead = await session.scalar(
        select(func.count()).select_from(DeadLetter).where(DeadLetter.resolved_at.is_(None))
    )
    seconds = await session.scalar(
        select(
            func.percentile_cont(0.5).within_group(
                func.extract("epoch", Message.processed_at - Message.received_at)
            )
        ).where(Message.received_at >= week_start, Message.processed_at.is_not(None))
    )
    return {
        "messages_today": int(today_count or 0),
        "messages_week": int(sum(by_status.values())),
        "auto_rate": round(100 * answered / handled) if handled else None,
        "open_handoffs": int(open_handoffs or 0),
        "leads_week": int(lead_count),
        "leads_amount_week": Decimal(lead_sum or 0),
        "dead_letters": int(dead or 0),
        "median_seconds": float(seconds) if seconds is not None else None,
    }


async def messages_per_day(
    session: AsyncSession, business: BusinessConfig, today: date, days: int = 14
) -> tuple[list[date], dict[str, list[int]]]:
    """Per-day counts for each outcome (answered / handoff / ignored)."""
    span = _days(today, days)
    day = _local_day(Message.received_at, business)
    rows = await session.execute(
        select(day, Message.status, func.count())
        .where(day >= span[0])
        .group_by(day, Message.status)
    )
    table: dict[str, dict[date, int]] = {outcome: {} for outcome in OUTCOMES}
    for row_day, status, count in rows:
        # Messages that failed every retry also ended up with a human.
        outcome = MessageStatus.HANDOFF.value if status == MessageStatus.DEAD.value else status
        if outcome in table:
            table[outcome][row_day] = table[outcome].get(row_day, 0) + int(count)
    return span, {outcome: [table[outcome].get(d, 0) for d in span] for outcome in OUTCOMES}


async def categories(
    session: AsyncSession, business: BusinessConfig, today: date, days: int = 30
) -> list[tuple[str, int]]:
    start = datetime.combine(today - timedelta(days=days - 1), time.min, tzinfo=business.tz)
    rows = await session.execute(
        select(Message.category, func.count())
        .where(Message.received_at >= start, Message.category.is_not(None))
        .group_by(Message.category)
        .order_by(func.count().desc())
    )
    return [(str(category), int(count)) for category, count in rows]


async def channels(
    session: AsyncSession, business: BusinessConfig, today: date, days: int = 30
) -> list[tuple[str, int]]:
    start = datetime.combine(today - timedelta(days=days - 1), time.min, tzinfo=business.tz)
    rows = await session.execute(
        select(Message.channel, func.count())
        .where(Message.received_at >= start)
        .group_by(Message.channel)
        .order_by(func.count().desc())
    )
    return [(str(channel), int(count)) for channel, count in rows]


async def revenue_per_day(
    session: AsyncSession, business: BusinessConfig, today: date, days: int = 14
) -> tuple[list[date], list[float]]:
    span = _days(today, days)
    day = _local_day(Lead.created_at, business)
    rows = await session.execute(
        select(day, func.coalesce(func.sum(Lead.amount), 0))
        .where(day >= span[0], Lead.status != LeadStatus.INCOMPLETE.value)
        .group_by(day)
    )
    by_day = {row_day: float(amount) for row_day, amount in rows}
    return span, [by_day.get(d, 0.0) for d in span]


async def price_history(
    session: AsyncSession, business: BusinessConfig, service_code: str, today: date, days: int = 14
) -> tuple[list[date], dict[str, list[float]]]:
    """Last known price per competitor per day for one of our services."""
    span = _days(today, days)
    day = _local_day(PriceSnapshot.scraped_at, business)
    rows = await session.execute(
        select(PriceSnapshot.competitor, day, PriceSnapshot.price)
        .where(PriceSnapshot.service_code == service_code, day >= span[0])
        .order_by(PriceSnapshot.id)
    )
    table: dict[str, dict[date, float]] = {}
    for competitor, row_day, price in rows:
        table.setdefault(str(competitor), {})[row_day] = float(price)
    result: dict[str, list[float]] = {}
    for competitor, by_day in sorted(table.items()):
        # Carry the last known price forward; before the first reading, use that reading.
        last = by_day[min(by_day)]
        values: list[float] = []
        for d in span:
            last = by_day.get(d, last)
            values.append(last)
        result[competitor] = values
    return span, result


async def price_comparison(
    session: AsyncSession, business: BusinessConfig
) -> tuple[list[str], list[dict[str, Any]]]:
    """Our price next to each competitor's latest price, per service."""
    latest = (
        select(
            PriceSnapshot.competitor,
            PriceSnapshot.service_code,
            func.max(PriceSnapshot.id).label("id"),
        )
        .where(PriceSnapshot.service_code.is_not(None))
        .group_by(PriceSnapshot.competitor, PriceSnapshot.service_code)
        .subquery()
    )
    rows = await session.execute(
        select(PriceSnapshot.competitor, PriceSnapshot.service_code, PriceSnapshot.price).join(
            latest, PriceSnapshot.id == latest.c.id
        )
    )
    prices: dict[str, dict[str, Decimal]] = {}
    for competitor, service_code, price in rows:
        prices.setdefault(str(service_code), {})[str(competitor)] = Decimal(price)
    competitors = sorted({name for by_competitor in prices.values() for name in by_competitor})
    table = []
    for service in business.services:
        theirs = prices.get(service.code, {})
        cheapest = min(theirs.values(), default=None)
        table.append(
            {
                "service": service,
                "ours": service.price,
                "theirs": [theirs.get(name) for name in competitors],
                # We are "undercut" when someone is cheaper than our price.
                "undercut": bool(
                    cheapest is not None and service.price is not None and cheapest < service.price
                ),
            }
        )
    return competitors, table
