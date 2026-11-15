"""arq worker entry point: ``arq ai_operator.queue.worker.WorkerSettings``."""

from __future__ import annotations

from typing import Any, ClassVar

from arq import cron
from arq.connections import RedisSettings

from ai_operator.config import get_settings
from ai_operator.core.logging import get_logger
from ai_operator.queue import tasks
from ai_operator.runtime import build_context

log = get_logger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    ctx["app"] = await build_context()
    app = ctx["app"]
    log.info("worker.started", provider=app.provider.name, channels=app.channels.names())


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["app"].close()


class WorkerSettings:
    functions: ClassVar[list[Any]] = [
        tasks.process_message,
        tasks.send_outbound,
        tasks.deliver_notification,
        tasks.sync_lead,
        tasks.poll_email,
        tasks.scrape_prices,
        tasks.morning_report,
    ]
    # The sweeper lives inside the worker on purpose: it must run even if the
    # scheduler process is down.
    cron_jobs: ClassVar[list[Any]] = [cron(tasks.sweep, second=0)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    # Our own code decides when to give up (and writes the dead letter); arq's
    # limit is only a backstop above it.
    max_tries = get_settings().max_attempts + 1
    max_jobs = 4
    job_timeout = 180
    # Results are not needed, and keeping them would block re-enqueueing the same job id.
    keep_result = 0
