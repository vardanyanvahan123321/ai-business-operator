"""Price monitoring, morning report, spreadsheet sinks, charts, sweeper, scheduler, demo seed."""

from __future__ import annotations

import csv
import importlib.util
import sys
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select, update

from ai_operator import demo, ops
from ai_operator.actions import sheets
from ai_operator.actions.sheets import CsvSink, GoogleSheetsSink, build_lead_sink, lead_to_row
from ai_operator.config import BusinessConfig, ScrapeConfig, Settings
from ai_operator.context import AppContext
from ai_operator.db.models import (
    DeadLetter,
    Handoff,
    Lead,
    Message,
    Notification,
    Outbound,
    PriceReport,
    PriceSnapshot,
    Report,
)
from ai_operator.db.session import session_scope
from ai_operator.pricing.monitor import run_price_check
from ai_operator.pricing.scraper import (
    PoliteFetcher,
    ScrapeDisallowed,
    ScrapeError,
    parse_price,
    parse_prices,
)
from ai_operator.queue import jobs, tasks
from ai_operator.queue.ingest import ingest
from ai_operator.queue.jobs import ArqQueue, RecordingQueue
from ai_operator.reports.morning import generate_morning_report
from ai_operator.scheduler import build_scheduler
from ai_operator.web import charts
from tests.conftest import ROOT, drain, telegram_message

_spec = importlib.util.spec_from_file_location(
    "competitor_site", ROOT / "demo/competitor_site/app.py"
)
assert _spec is not None
assert _spec.loader is not None
site = importlib.util.module_from_spec(_spec)
sys.modules["competitor_site"] = site
_spec.loader.exec_module(site)


class FakeSite:
    """The demo competitor site, served through an httpx mock transport."""

    def __init__(self) -> None:
        self.robots: httpx.Response | Exception = httpx.Response(200, text=site.ROBOTS)
        self.requests: list[str] = []
        self.overrides: dict[str, httpx.Response] = {}
        self.day = date(2026, 11, 16)

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append(path)
        assert "AIBusinessOperatorBot" in request.headers["user-agent"]
        if path == "/robots.txt":
            if isinstance(self.robots, Exception):
                raise self.robots
            return self.robots
        if path in self.overrides:
            return self.overrides[path]
        if path == "/autofix/prices":
            return httpx.Response(200, text=site.autofix_page(self.day))
        if path == "/garagepro/services":
            return httpx.Response(200, text=site.garagepro_page(self.day))
        return httpx.Response(404)

    def fetcher(self, min_delay: float = 0.0) -> PoliteFetcher:
        config = ScrapeConfig(min_delay_seconds=min_delay)
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.handler), headers={"User-Agent": config.user_agent}
        )
        return PoliteFetcher(config, client)


class TestScraper:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("12 000 AMD", "12000"),
            ("from 11500 dram", "11500"),
            ("12,500.50 $", "12500.50"),
            ("12.500,5 €", "12500.5"),
            ("9 000 ֏", "9000"),
            ("1,200", "1200"),
            ("call us", None),
        ],
    )
    def test_parse_price(self, text: str, expected: str | None) -> None:
        assert parse_price(text) == (Decimal(expected) if expected else None)

    def test_both_page_layouts_are_parsed_by_configured_selectors(
        self, business: BusinessConfig
    ) -> None:
        autofix, garage = business.competitors
        day = date(2026, 11, 16)
        table = parse_prices(site.autofix_page(day), autofix)
        cards = parse_prices(site.garagepro_page(day), garage)
        assert [item.name for item in table] == list(site.AUTOFIX)
        assert [item.name for item in cards] == list(site.GARAGE_PRO)
        assert all(abs(item.price - site.AUTOFIX[item.name]) <= 500 for item in table)
        assert (
            parse_prices(
                "<html><table class='prices'><tr class='item'><td>x</td></tr></table>", autofix
            )
            == []
        )

    async def test_robots_txt_is_obeyed(self) -> None:
        fake = FakeSite()
        fetcher = fake.fetcher()
        assert "AutoFix" in await fetcher.get("http://competitor.test/autofix/prices")
        with pytest.raises(ScrapeDisallowed):
            await fetcher.get("http://competitor.test/admin/prices")
        with pytest.raises(ScrapeDisallowed):
            await fetcher.get("http://competitor.test/garagepro/internal/costs")
        assert "/admin/prices" not in fake.requests  # the forbidden URL was never requested
        assert fake.requests.count("/robots.txt") == 1  # robots.txt is cached per host
        await fetcher.close()

    async def test_unreadable_robots_means_stay_away_and_missing_means_allowed(self) -> None:
        broken = FakeSite()
        broken.robots = httpx.Response(503)
        with pytest.raises(ScrapeDisallowed):
            await broken.fetcher().get("http://competitor.test/autofix/prices")
        down = FakeSite()
        down.robots = httpx.ConnectError("refused")
        with pytest.raises(ScrapeDisallowed):
            await down.fetcher().get("http://competitor.test/autofix/prices")
        absent = FakeSite()
        absent.robots = httpx.Response(404)
        assert await absent.fetcher().get("http://competitor.test/autofix/prices")

    async def test_requests_to_one_host_are_spaced_by_the_crawl_delay(self) -> None:
        import time

        fake = FakeSite()
        fake.robots = httpx.Response(200, text="User-agent: *\nCrawl-delay: 1\n")
        fetcher = fake.fetcher(min_delay=0.05)  # the site asks for more than our minimum
        started = time.monotonic()
        await fetcher.get("http://competitor.test/autofix/prices")
        await fetcher.get("http://competitor.test/garagepro/services")
        assert time.monotonic() - started >= 1.0

    async def test_http_errors_are_scrape_errors(self) -> None:
        fake = FakeSite()
        fake.overrides["/autofix/prices"] = httpx.Response(500)
        with pytest.raises(ScrapeError, match="cannot fetch"):
            await fake.fetcher().get("http://competitor.test/autofix/prices")


@pytest.fixture
def competitors_on_test_host(ctx: AppContext) -> AppContext:
    business = ctx.business.model_copy(deep=True)
    for competitor in business.competitors:
        competitor.url = competitor.url.replace("http://localhost:8090", "http://competitor.test")
    ctx.business = business
    return ctx


class TestPriceMonitor:
    async def test_first_run_is_a_baseline_second_run_reports_changes(
        self, competitors_on_test_host: AppContext
    ) -> None:
        ctx = competitors_on_test_host
        fake = FakeSite()

        assert await run_price_check(ctx, fake.fetcher()) is None  # nothing to compare with yet
        async with ctx.sessions() as session:
            snapshots = (await session.scalars(select(PriceSnapshot))).all()
        assert len(snapshots) == len(site.AUTOFIX) + len(site.GARAGE_PRO)
        assert {s.service_code for s in snapshots if s.item_name == "Brake pad change"} == {
            "brake_pads"
        }

        cheap = site.autofix_page(fake.day).replace(
            f"{site.drift('Oil change', 11500, fake.day):,}".replace(",", " "), "9 900"
        )
        fake.overrides["/autofix/prices"] = httpx.Response(200, text=cheap)
        report_id = await run_price_check(ctx, fake.fetcher())
        assert report_id is not None
        async with ctx.sessions() as session:
            report = await session.get(PriceReport, report_id)
            notification = await session.scalar(
                select(Notification).where(Notification.kind == "prices")
            )
        assert report is not None
        [change] = report.changes
        assert (change["item_name"], change["new_price"], change["our_price"]) == (
            "Oil change",
            "9900",
            "12000",
        )
        assert "AutoFix Center" in report.summary
        assert "дешевле нас" in report.summary  # the owner reads Russian
        assert notification is not None
        queue = ctx.queue
        assert isinstance(queue, RecordingQueue)
        assert (jobs.DELIVER_NOTIFICATION, notification.id) in queue.jobs

    async def test_one_broken_site_does_not_stop_the_others(
        self, competitors_on_test_host: AppContext
    ) -> None:
        ctx = competitors_on_test_host
        fake = FakeSite()
        fake.overrides["/autofix/prices"] = httpx.Response(500)
        fake.overrides["/garagepro/services"] = httpx.Response(200, text="<html>redesigned</html>")
        assert await run_price_check(ctx, fake.fetcher()) is None
        async with ctx.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(PriceSnapshot)) == 0

        del fake.overrides["/garagepro/services"]
        await run_price_check(ctx, fake.fetcher())
        async with ctx.sessions() as session:
            names = set(await session.scalars(select(PriceSnapshot.competitor)))
        assert names == {"Garage Pro"}


class TestMorningReport:
    async def test_report_counts_yesterday_and_flags_what_needs_attention(
        self, ctx: AppContext, kb: int
    ) -> None:
        today = datetime.now(ctx.business.tz).date()
        yesterday = datetime.now(UTC) - timedelta(days=1)
        texts = [
            "Сколько стоит замена масла?",
            "Ужасный сервис, верните деньги!",
            "Earn money with crypto casino https://x.example",
            "Запишите на диагностику сегодня в 23:59, телефон 091123456",
            "Хочу записаться на шиномонтаж",
        ]
        for index, text in enumerate(texts):
            inbound = telegram_message(text, chat=str(index), msg_id=str(index))
            inbound.received_at = yesterday
            await ingest(ctx, inbound)
        await drain(ctx)
        async with session_scope(ctx.sessions) as session:
            await session.execute(update(Lead).values(created_at=yesterday))
            session.add(DeadLetter(job="send_outbound", ref_id=1, error="boom", attempts=5))
            session.add(PriceReport(summary="Garage Pro lowered prices.", changes=[]))

        report_id = await generate_morning_report(ctx, today)
        async with ctx.sessions() as session:
            report = await session.get(Report, report_id)
        assert report is not None
        assert report.data["messages_total"] == 5
        assert report.data["answered"] == 3
        assert report.data["handoff"] == 1
        assert report.data["spam"] == 1
        assert report.data["leads"] == 1
        assert report.data["leads_amount"] == "8000.00"
        assert (report.data["open_handoffs"], report.data["incomplete_leads"]) == (1, 1)
        assert report.data["dead_letters"] == 1
        assert f"Утренний отчёт - {today:%d.%m.%Y}" in report.content
        assert "Новых заявок: 1, на сумму 8 000 AMD" in report.content
        assert "невыполненные задачи: 1" in report.content
        assert "Garage Pro lowered prices." in report.content

        # Re-running the same day updates the report instead of duplicating it.
        assert await tasks.morning_report({"app": ctx}) == report_id
        async with ctx.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Report)) == 1
            sent = await session.scalar(
                select(func.count()).select_from(Notification).where(Notification.kind == "report")
            )
        assert sent == 1

    async def test_quiet_day(self, ctx: AppContext) -> None:
        report_id = await generate_morning_report(ctx, date(2026, 11, 16))
        async with ctx.sessions() as session:
            report = await session.get(Report, report_id)
        assert report is not None
        assert "ничего, всё в порядке" in report.content
        assert "данных пока нет" in report.content


class FakeWorksheet:
    def __init__(self) -> None:
        self.rows: list[list[str]] = []

    def row_values(self, index: int) -> list[str]:
        return self.rows[index - 1] if len(self.rows) >= index else []

    def update(self, range_name: str, values: list[list[str]], **_kwargs: Any) -> None:
        index = int(range_name[1:]) - 1
        while len(self.rows) <= index:
            self.rows.append([])
        self.rows[index] = values[0]

    def find(self, value: str, in_column: int) -> Any:
        for index, row in enumerate(self.rows[1:], start=2):
            if row and row[in_column - 1] == value:
                return type("Cell", (), {"row": index})()
        return None

    def append_row(self, row: list[str], **kwargs: Any) -> None:
        assert kwargs["value_input_option"] == "RAW"
        self.rows.append(row)


class TestSheets:
    def _lead(self, **overrides: Any) -> Lead:
        values: dict[str, Any] = {
            "id": 7,
            "created_at": datetime(2026, 11, 16, 6, 0, tzinfo=UTC),
            "status": "new",
            "channel": "telegram",
            "sender_id": "42",
            "name": "Anna",
            "phone": "+37491123456",
            "service_code": "oil_change",
            "scheduled_at": datetime(2026, 11, 17, 6, 0, tzinfo=UTC),
            "amount": Decimal("12000"),
            "currency": "AMD",
            "language": "en",
        }
        values.update(overrides)
        return Lead(**values)

    def test_row_uses_business_timezone_and_neutralises_formulas(
        self, business: BusinessConfig
    ) -> None:
        row = lead_to_row(self._lead(name='=HYPERLINK("http://evil")'), business)
        assert row[1] == "2026-11-16 10:00"
        assert row[4].startswith("'=")  # cannot run as a spreadsheet formula
        assert row[7:10] == ["Oil change", "2026-11-17 10:00", "12000"]
        loose = lead_to_row(
            self._lead(service_code=None, service_text="+bumper", scheduled_at=None, amount=None),
            business,
        )
        assert loose[7:10] == ["'+bumper", "", ""]

    async def test_csv_sink_upserts_by_id(self, tmp_path: Path, business: BusinessConfig) -> None:
        sink = CsvSink(tmp_path / "out" / "leads.csv")
        await sink.upsert(lead_to_row(self._lead(), business))
        await sink.upsert(lead_to_row(self._lead(id=8, name="Bob"), business))
        await sink.upsert(lead_to_row(self._lead(status="confirmed"), business))
        with (tmp_path / "out" / "leads.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [(r["id"], r["status"], r["name"]) for r in rows] == [
            ("7", "confirmed", "Anna"),
            ("8", "new", "Bob"),
        ]

    async def test_google_sheets_sink_writes_header_then_upserts(
        self, business: BusinessConfig
    ) -> None:
        sink = GoogleSheetsSink("sheet-id", "service-account.json")
        worksheet = FakeWorksheet()
        sink._worksheet = worksheet
        await sink.upsert(lead_to_row(self._lead(), business))
        await sink.upsert(lead_to_row(self._lead(status="done"), business))
        assert worksheet.rows[0] == sheets.COLUMNS
        assert len(worksheet.rows) == 2
        assert worksheet.rows[1][2] == "done"

    def test_sink_selection(self, tmp_path: Path) -> None:
        plain = Settings(_env_file=None, data_dir=tmp_path)  # type: ignore[call-arg]
        assert isinstance(build_lead_sink(plain), CsvSink)
        google = Settings(
            _env_file=None, google_sheets_id="x", google_service_account_file="sa.json"
        )  # type: ignore[call-arg]
        assert isinstance(build_lead_sink(google), GoogleSheetsSink)

    async def test_sync_failure_is_retried_then_buried(self, ctx: AppContext, kb: int) -> None:
        class BrokenSink:
            name = "broken"

            async def upsert(self, row: list[str]) -> None:
                raise ConnectionError("sheets API is down")

        ctx.lead_sink = BrokenSink()
        await ingest(
            ctx, telegram_message("Book me for an oil change tomorrow at 9:00, phone +37491123456")
        )
        await drain(ctx, max_rounds=30)
        async with ctx.sessions() as session:
            lead = await session.scalar(select(Lead))
            dead = await session.scalar(select(DeadLetter).where(DeadLetter.job == "sync_lead"))
        assert lead is not None
        assert lead.sheet_synced_at is None
        assert "sheets API is down" in (lead.sheet_error or "")
        assert dead is not None

        ctx.lead_sink = CsvSink(ctx.settings.leads_csv)
        assert await ops.retry_dead_letter(ctx, dead.id)
        await drain(ctx)
        async with ctx.sessions() as session:
            lead = await session.scalar(select(Lead))
        assert lead is not None
        assert lead.sheet_synced_at is not None


class TestCharts:
    def test_axis_helpers(self) -> None:
        assert [charts.nice_max(v) for v in (0, 7, 14, 37, 80, 137000)] == [
            1,
            10,
            20,
            50,
            100,
            200000,
        ]
        assert [charts.compact(v) for v in (950, 12500, 3_000_000)] == ["950", "12.5k", "3M"]

    def test_stacked_bars_have_tooltips_and_escape_labels(self) -> None:
        svg = charts.stacked_bars(
            ["01.11", "02.11"],
            [charts.Series("Answered <b>", [3, 0], 1), charts.Series("Handoff", [1, 2], 2)],
            "Outcomes",
        )
        assert svg.count("<rect") == 3  # zero-height segments are not drawn
        assert "<title>01.11 - Answered &lt;b&gt;: 3</title>" in svg
        assert 'aria-label="Outcomes"' in svg
        assert "var(--series-2)" in svg

    def test_line_chart_labels_series_ends(self) -> None:
        svg = charts.line_chart(
            ["01.11", "02.11", "03.11"],
            [charts.Series("Garage Pro", [11500, 11000, 11000], 3)],
            "History",
        )
        assert svg.count("<circle") == 3
        assert '<text class="end-label"' in svg
        assert charts.line_chart([], [charts.Series("x", [], 1)], "Empty").startswith("<svg")

    def test_horizontal_bars(self) -> None:
        svg = charts.horizontal_bars([("Question", 38), ("Spam", 0)], "Categories")
        assert svg.count("<rect") == 2
        assert ">38</text>" in svg
        assert charts.horizontal_bars([], "Empty").startswith("<svg")


class TestSweeperAndQueue:
    async def test_sweeper_requeues_everything_whose_job_was_lost(
        self, ctx: AppContext, kb: int
    ) -> None:
        queue = ctx.queue
        assert isinstance(queue, RecordingQueue)
        long_ago = datetime.now(UTC) - timedelta(hours=1)

        # 1. received but never enqueued (Redis was down at ingest time)
        lost_id, _ = await ingest(ctx, telegram_message("Сколько стоит диагностика?", msg_id="1"))
        # 2. a worker died in the middle of processing
        stuck_id, _ = await ingest(ctx, telegram_message("Где вы находитесь?", msg_id="2"))
        # 3. a fresh message that is simply waiting its turn must be left alone
        fresh_id, _ = await ingest(ctx, telegram_message("Какой адрес?", msg_id="3"))
        # 4. processed, but its reply / notification / sheet sync jobs were lost
        await ingest(
            ctx,
            telegram_message(
                "Book an oil change tomorrow at 9:00, phone +37491123456", chat="9", msg_id="4"
            ),
        )
        queue.pop_all()
        from ai_operator.pipeline import Pipeline

        await Pipeline(ctx).process(4)
        queue.pop_all()
        async with session_scope(ctx.sessions) as session:
            await session.execute(
                update(Message).where(Message.id == lost_id).values(received_at=long_ago)
            )
            await session.execute(
                update(Message)
                .where(Message.id == stuck_id)
                .values(status="processing", processing_started_at=long_ago)
            )
            await session.execute(update(Outbound).values(created_at=long_ago))
            await session.execute(update(Notification).values(created_at=long_ago))
            await session.execute(update(Lead).values(updated_at=long_ago))

        requeued = await tasks.sweep({"app": ctx})
        assert requeued == {
            "process_message": 2,
            "send_outbound": 1,
            "deliver_notification": 1,
            "sync_lead": 1,
        }
        assert (jobs.PROCESS_MESSAGE, fresh_id) not in queue.jobs
        await drain(ctx)
        async with ctx.sessions() as session:
            statuses = dict((await session.execute(select(Message.id, Message.status))).all())
            pending = await session.scalar(
                select(func.count()).select_from(Outbound).where(Outbound.status == "pending")
            )
        assert statuses[lost_id] == "answered"
        assert statuses[stuck_id] == "answered"
        assert statuses[fresh_id] == "received"  # untouched: its own job is still on its way
        assert pending == 0
        assert await tasks.sweep({"app": ctx}) == dict.fromkeys(requeued, 0)

    async def test_real_redis_queue_runs_the_whole_conveyor(self, ctx: AppContext, kb: int) -> None:
        """The same scenario as the e2e tests, but through a real arq worker and Redis."""
        from arq.connections import RedisSettings, create_pool
        from arq.worker import Worker

        from tests.conftest import TEST_REDIS_URL, FakeChannel

        redis = await create_pool(RedisSettings.from_dsn(TEST_REDIS_URL))
        await redis.flushdb()
        ctx.queue = ArqQueue(redis)
        message_id, _ = await ingest(ctx, telegram_message("Сколько стоит замена масла?"))
        assert await ctx.queue.enqueue(jobs.PROCESS_MESSAGE, message_id) is False  # deduplicated

        worker = Worker(
            functions=[
                tasks.process_message,
                tasks.send_outbound,
                tasks.deliver_notification,
                tasks.sync_lead,
            ],
            redis_pool=redis,
            burst=True,
            ctx={"app": ctx},
            keep_result=0,
            handle_signals=False,
            poll_delay=0.05,
        )
        await worker.main()
        await worker.close()

        channel = ctx.channels.get("telegram")
        assert isinstance(channel, FakeChannel)
        assert "12 000" in channel.sent[0].text
        # keep_result=0: the finished job id is free again (needed by retries and the sweeper)
        assert await ctx.queue.enqueue(jobs.PROCESS_MESSAGE, message_id) is True
        await redis.flushdb()

    async def test_enqueue_failure_does_not_break_the_caller(self, ctx: AppContext) -> None:
        class DownQueue:
            async def enqueue(
                self, job: str, ref_id: int | None = None, *, defer: float = 0
            ) -> bool:
                raise ConnectionError("redis is down")

        ctx.queue = DownQueue()
        message_id, created = await ingest(ctx, telegram_message("Hello"))
        assert created  # stored even though the queue is unavailable
        async with ctx.sessions() as session:
            stored = await session.get(Message, message_id)
        assert stored is not None
        assert stored.status == "received"

    def test_backoff_grows_and_is_capped(self) -> None:
        assert 5 <= tasks.backoff(5, 1) < 6
        assert 20 <= tasks.backoff(5, 3) < 21
        assert 300 <= tasks.backoff(5, 20) < 301


class TestSchedulerAndDemo:
    async def test_scheduler_registers_business_schedules(self, ctx: AppContext) -> None:
        scheduler = build_scheduler(ctx)
        by_id = {job.id: job for job in scheduler.get_jobs()}
        assert set(by_id) == {"morning_report", "scrape_prices"}
        assert "hour='8'" in str(by_id["morning_report"].trigger)
        assert str(scheduler.timezone) == "Asia/Yerevan"

        ctx.extras["inbox"] = object()
        assert "poll_email" in {job.id for job in build_scheduler(ctx).get_jobs()}
        await by_id["morning_report"].func(*by_id["morning_report"].args)
        queue = ctx.queue
        assert isinstance(queue, RecordingQueue)
        assert queue.jobs == [("morning_report", None)]

    async def test_demo_seed_fills_every_section_and_is_idempotent(self, ctx: AppContext) -> None:
        assert await demo.seed(ctx) is True
        async with ctx.sessions() as session:
            messages = await session.scalar(select(func.count()).select_from(Message))
            stuck = await session.scalar(
                select(func.count())
                .select_from(Message)
                .where(Message.status.in_(["received", "processing"]))
            )
            pending = await session.scalar(
                select(func.count()).select_from(Outbound).where(Outbound.status != "sent")
            )
            leads = await session.scalar(
                select(func.count()).select_from(Lead).where(Lead.status == "new")
            )
            open_handoffs = await session.scalar(
                select(func.count()).select_from(Handoff).where(Handoff.status == "open")
            )
            price_reports = await session.scalar(select(func.count()).select_from(PriceReport))
            report = await session.scalar(select(Report))
            oldest = await session.scalar(select(func.min(Message.received_at)))
        assert messages is not None
        assert messages > 50
        assert (stuck, pending) == (0, 0)
        assert leads
        assert price_reports == 4
        assert open_handoffs is not None
        assert report is not None
        assert oldest is not None
        assert oldest < datetime.now(UTC) - timedelta(days=12)
        assert ctx.settings.leads_csv.exists()

        assert await demo.seed(ctx) is False  # second run changes nothing
        async with ctx.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Message)) == messages

    async def test_demo_traffic_sends_customer_emails(
        self, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[Any] = []

        async def fake_send(message: Any, **kwargs: Any) -> None:
            sent.append((message["To"], kwargs["hostname"]))

        monkeypatch.setattr(demo.aiosmtplib, "send", fake_send)
        ctx.settings.smtp_host = "mailpit"
        assert await demo.traffic(ctx) == 3
        assert sent[0] == (ctx.settings.email_address, "mailpit")

    def test_competitor_site_prices_drift_within_bounds(self) -> None:
        values = {site.drift("Oil change", 11500, date(2026, 11, day)) for day in range(1, 29)}
        assert values <= {11000, 11500, 12000}
        assert len(values) > 1
