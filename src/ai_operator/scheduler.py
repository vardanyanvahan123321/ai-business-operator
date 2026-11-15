"""The clock: APScheduler decides *when*, the worker does the work.

Scheduled jobs only drop a message onto the queue. That way a report or a
scrape gets the same retries, backoff and dead-letter handling as any other
job, and a slow run never blocks the clock.

The Telegram long-polling loop also lives in this process, because it is the
other "always listening" component.
"""

from __future__ import annotations

import asyncio
import signal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ai_operator.channels.telegram import TelegramChannel
from ai_operator.channels.telegram_bot import run_polling
from ai_operator.context import AppContext
from ai_operator.core.logging import get_logger
from ai_operator.queue import jobs
from ai_operator.runtime import build_context

log = get_logger(__name__)
EMAIL_POLL_SECONDS = 20


def build_scheduler(ctx: AppContext) -> AsyncIOScheduler:
    business = ctx.business
    scheduler = AsyncIOScheduler(timezone=business.tz)

    async def enqueue(job: str) -> None:
        await jobs.safe_enqueue(ctx.queue, job)

    scheduler.add_job(
        enqueue,
        CronTrigger(hour=business.morning_report.hour, minute=business.morning_report.minute),
        args=[jobs.MORNING_REPORT],
        id=jobs.MORNING_REPORT,
        misfire_grace_time=3600,
        coalesce=True,
    )
    if business.competitors:
        scheduler.add_job(
            enqueue,
            CronTrigger(hour=business.scrape.hour, minute=business.scrape.minute),
            args=[jobs.SCRAPE_PRICES],
            id=jobs.SCRAPE_PRICES,
            misfire_grace_time=3600,
            coalesce=True,
        )
    if "inbox" in ctx.extras:
        scheduler.add_job(
            enqueue,
            IntervalTrigger(seconds=EMAIL_POLL_SECONDS),
            args=[jobs.POLL_EMAIL],
            id=jobs.POLL_EMAIL,
            coalesce=True,
            max_instances=1,
        )
    return scheduler


async def main() -> None:  # pragma: no cover - process entry point
    ctx = await build_context()
    scheduler = build_scheduler(ctx)
    scheduler.start()
    log.info("scheduler.started", jobs=[job.id for job in scheduler.get_jobs()])

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    polling: asyncio.Task[None] | None = None
    channel = ctx.channels.get("telegram") if "telegram" in ctx.channels else None
    if isinstance(channel, TelegramChannel):  # not the simulated demo channel
        polling = asyncio.create_task(run_polling(ctx, channel))

    await stop.wait()
    if polling is not None:
        polling.cancel()
        await asyncio.gather(polling, return_exceptions=True)
    scheduler.shutdown(wait=False)
    await ctx.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
