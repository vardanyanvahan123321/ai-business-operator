"""Competitor price monitoring: scrape, store history, detect and explain changes."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_operator.actions.audit import audit, notify_owner
from ai_operator.ai.schemas import PriceChange
from ai_operator.context import AppContext
from ai_operator.core.i18n import t
from ai_operator.core.logging import get_logger
from ai_operator.db.models import PriceReport, PriceSnapshot
from ai_operator.db.session import session_scope
from ai_operator.pricing.scraper import PoliteFetcher, ScrapeError, parse_prices
from ai_operator.queue import jobs

log = get_logger(__name__)


async def latest_prices(session: AsyncSession, competitor: str) -> dict[str, Decimal]:
    """Most recent stored price per item for one competitor."""
    latest = (
        select(PriceSnapshot.item_name, func.max(PriceSnapshot.id).label("id"))
        .where(PriceSnapshot.competitor == competitor)
        .group_by(PriceSnapshot.item_name)
        .subquery()
    )
    rows = await session.execute(
        select(PriceSnapshot.item_name, PriceSnapshot.price).join(
            latest, PriceSnapshot.id == latest.c.id
        )
    )
    return {str(name): Decimal(price) for name, price in rows.all()}


async def run_price_check(ctx: AppContext, fetcher: PoliteFetcher | None = None) -> int | None:
    """Scrape every configured competitor. Returns the id of the change report, if any."""
    business = ctx.business
    own_fetcher = fetcher is None
    fetcher = fetcher or PoliteFetcher(business.scrape)
    changes: list[PriceChange] = []
    scraped = 0
    try:
        async with session_scope(ctx.sessions) as session:
            for competitor in business.competitors:
                try:
                    items = parse_prices(await fetcher.get(competitor.url), competitor)
                except ScrapeError as exc:
                    # One broken site must not stop the others.
                    log.warning("prices.scrape_failed", competitor=competitor.name, error=str(exc))
                    audit(
                        session,
                        "prices.scrape_failed",
                        level="warning",
                        competitor=competitor.name,
                        error=str(exc),
                    )
                    continue
                if not items:
                    audit(session, "prices.empty", level="warning", competitor=competitor.name)
                    continue
                previous = await latest_prices(session, competitor.name)
                for item in items:
                    scraped += 1
                    service_code = competitor.service_map.get(item.name)
                    session.add(
                        PriceSnapshot(
                            competitor=competitor.name,
                            item_name=item.name,
                            service_code=service_code,
                            price=item.price,
                            currency=business.currency,
                        )
                    )
                    old = previous.get(item.name)
                    # A brand-new item on first sight is a baseline, not a "change".
                    if old is not None and old != item.price:
                        service = business.service(service_code)
                        changes.append(
                            PriceChange(
                                competitor=competitor.name,
                                item_name=item.name,
                                service_code=service_code,
                                old_price=old,
                                new_price=item.price,
                                our_price=service.price if service else None,
                                currency=business.currency,
                            )
                        )
            audit(session, "prices.checked", items=scraped, changes=len(changes))
    finally:
        if own_fetcher:
            await fetcher.close()

    if not changes:
        return None

    # The AI call happens outside the transaction; the facts above are already saved.
    summary = await ctx.provider.summarize_prices(changes, business.owner_language)
    async with session_scope(ctx.sessions) as session:
        report = PriceReport(
            summary=summary,
            changes=[
                {
                    "competitor": c.competitor,
                    "item_name": c.item_name,
                    "service_code": c.service_code,
                    "old_price": str(c.old_price),
                    "new_price": str(c.new_price),
                    "our_price": str(c.our_price) if c.our_price is not None else None,
                    "delta_percent": round(c.delta_percent or 0.0, 1),
                }
                for c in changes
            ],
        )
        session.add(report)
        await session.flush()
        notification = await notify_owner(
            session,
            kind="prices",
            dedupe_key=f"prices:{report.id}",
            text=t(business.owner_language, "notify.prices", summary=summary),
        )
        report_id = report.id
    if notification is not None:
        await jobs.safe_enqueue(ctx.queue, jobs.DELIVER_NOTIFICATION, notification)
    return report_id
