"""Job queue abstraction over arq/Redis."""

from __future__ import annotations

from typing import Any, Protocol

from ai_operator.core.logging import get_logger

log = get_logger(__name__)

# Job names (functions registered in the worker).
PROCESS_MESSAGE = "process_message"
SEND_OUTBOUND = "send_outbound"
DELIVER_NOTIFICATION = "deliver_notification"
SYNC_LEAD = "sync_lead"
POLL_EMAIL = "poll_email"
SCRAPE_PRICES = "scrape_prices"
MORNING_REPORT = "morning_report"


class JobQueue(Protocol):
    async def enqueue(self, job: str, ref_id: int | None = None, *, defer: float = 0) -> bool:
        """Schedule a job. Returns False if the same job is already queued."""


class ArqQueue:
    """Redis-backed queue.

    The job id is derived from the job name and the row it works on, so asking
    twice for "process message 42" yields a single queued job.
    """

    def __init__(self, redis: Any) -> None:
        self._redis = redis

    async def enqueue(self, job: str, ref_id: int | None = None, *, defer: float = 0) -> bool:
        job_id = f"{job}:{ref_id}" if ref_id is not None else job
        args = (ref_id,) if ref_id is not None else ()
        result = await self._redis.enqueue_job(
            job, *args, _job_id=job_id, _defer_by=defer if defer > 0 else None
        )
        return result is not None


class RecordingQueue:
    """In-memory queue for tests and one-off scripts: remembers what was enqueued."""

    def __init__(self) -> None:
        self.jobs: list[tuple[str, int | None]] = []

    async def enqueue(self, job: str, ref_id: int | None = None, *, defer: float = 0) -> bool:
        if (job, ref_id) in self.jobs:
            return False
        self.jobs.append((job, ref_id))
        return True

    def pop_all(self) -> list[tuple[str, int | None]]:
        jobs, self.jobs = self.jobs, []
        return jobs


async def safe_enqueue(queue: JobQueue, job: str, ref_id: int | None = None) -> None:
    """Enqueue, but never fail the caller.

    The row is already committed with a ``pending`` status; if Redis is down
    right now, the periodic sweeper will find the row and enqueue it later.
    """
    try:
        await queue.enqueue(job, ref_id)
    except Exception as exc:
        log.warning("queue.enqueue_failed", job=job, ref_id=ref_id, error=repr(exc))
