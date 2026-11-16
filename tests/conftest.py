"""Shared fixtures.

Tests run against a real PostgreSQL (with pgvector): the schema is created by
the actual Alembic migrations, so the migrations are tested on every run.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from datetime import time
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://operator:operator@localhost:55432/operator"
)
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:56379/15")

# Must be set before any application module reads its settings.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["REDIS_URL"] = TEST_REDIS_URL
os.environ["AI_PROVIDER"] = "mock"
os.environ["LOG_LEVEL"] = "WARNING"
for _name in ("TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "SMTP_HOST", "GOOGLE_SHEETS_ID"):
    os.environ.pop(_name, None)

from ai_operator.actions.sheets import CsvSink  # noqa: E402
from ai_operator.ai.providers.mock import MockProvider  # noqa: E402
from ai_operator.channels.base import Channel, ChannelRegistry  # noqa: E402
from ai_operator.channels.webform import WebFormChannel  # noqa: E402
from ai_operator.config import (  # noqa: E402
    WEEKDAYS,
    BusinessConfig,
    Schedule,
    Settings,
    load_business_config,
)
from ai_operator.context import AppContext  # noqa: E402
from ai_operator.core.message import InboundMessage, OutboundMessage  # noqa: E402
from ai_operator.db.session import make_engine, make_session_factory  # noqa: E402
from ai_operator.queue import jobs, tasks  # noqa: E402
from ai_operator.queue.jobs import RecordingQueue  # noqa: E402
from ai_operator.seed import load_knowledge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = (
    "events, dead_letters, notifications, outbound_messages, handoffs, invoices, leads, "
    "messages, kb_chunks, kb_documents, price_snapshots, price_reports, reports, settings"
)


class FakeChannel(Channel):
    """Records what would have been sent; can be told to fail."""

    def __init__(self, name: str) -> None:
        self.name = name  # type: ignore[misc]
        self.sent: list[OutboundMessage] = []
        self.fail_times = 0

    async def send(self, message: OutboundMessage) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            from ai_operator.channels.base import ChannelError

            raise ChannelError("simulated delivery failure")
        self.sent.append(message)


@pytest.fixture(scope="session")
def _migrated() -> Iterator[None]:
    """Fresh schema built by the real migrations, once per test session."""
    import asyncio

    from sqlalchemy.ext.asyncio import create_async_engine

    async def reset() -> None:
        engine = create_async_engine(TEST_DATABASE_URL)
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
        await engine.dispose()

    asyncio.run(reset())
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    command.upgrade(config, "head")
    yield


@pytest.fixture(scope="session")
def real_business() -> BusinessConfig:
    """The demo config exactly as shipped (Sunday is a day off)."""
    return load_business_config(ROOT / "config" / "business.yaml")


@pytest.fixture(scope="session")
def business(real_business: BusinessConfig) -> BusinessConfig:
    """Open seven days a week, so "tomorrow at 10" works whatever day the tests run.

    Tests about working hours use fixed dates and ``real_business`` instead.
    """
    always_open = Schedule(hours={day: (time(9), time(19)) for day in WEEKDAYS})
    return real_business.model_copy(update={"schedule": always_open})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=TEST_DATABASE_URL,
        redis_url=TEST_REDIS_URL,
        ai_provider="mock",
        data_dir=tmp_path / "var",
        business_config=ROOT / "config" / "business.yaml",
        admin_password="test-password",
        session_secret="test-secret-test-secret-test-secret",
        retry_base_seconds=0,
        demo_mode=True,
    )


@pytest.fixture
async def ctx(
    _migrated: None, settings: Settings, business: BusinessConfig
) -> AsyncIterator[AppContext]:
    engine = make_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    channels = ChannelRegistry()
    channels.register(WebFormChannel())
    channels.register(FakeChannel("telegram"))
    channels.register(FakeChannel("email"))
    context = AppContext(
        settings=settings,
        business=business,
        engine=engine,
        sessions=make_session_factory(engine),
        provider=MockProvider(),
        channels=channels,
        queue=RecordingQueue(),
        lead_sink=CsvSink(settings.leads_csv),
    )
    yield context
    await engine.dispose()


@pytest.fixture
async def kb(ctx: AppContext) -> int:
    """The demo knowledge base, indexed."""
    return await load_knowledge(ctx, ROOT / "data" / "knowledge")


_TASKS = {
    jobs.PROCESS_MESSAGE: tasks.process_message,
    jobs.SEND_OUTBOUND: tasks.send_outbound,
    jobs.DELIVER_NOTIFICATION: tasks.deliver_notification,
    jobs.SYNC_LEAD: tasks.sync_lead,
}


async def drain(ctx: AppContext, max_rounds: int = 20) -> list[tuple[str, Any]]:
    """Run queued jobs in-process until the queue is empty (the worker, minus Redis)."""
    from arq import Retry

    queue = ctx.queue
    assert isinstance(queue, RecordingQueue)
    executed: list[tuple[str, Any]] = []
    attempts: dict[tuple[str, int | None], int] = {}
    for _ in range(max_rounds):
        batch = queue.pop_all()
        if not batch:
            break
        for job, ref_id in batch:
            attempts[(job, ref_id)] = attempts.get((job, ref_id), 0) + 1
            try:
                result = await _TASKS[job]({"app": ctx, "job_try": attempts[(job, ref_id)]}, ref_id)
            except Retry:
                await queue.enqueue(job, ref_id)
                result = "retry"
            executed.append((job, result))
    return executed


def telegram_message(
    text: str, *, chat: str = "1001", msg_id: str = "1", name: str | None = "Aram"
) -> InboundMessage:
    return InboundMessage(
        channel="telegram",
        external_id=f"{chat}:{msg_id}",
        sender_id=chat,
        sender_name=name,
        text=text,
    )
