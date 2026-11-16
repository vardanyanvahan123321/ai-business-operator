"""Booking rules found by testing the live Telegram bot.

1. A date in the past is never booked; a date without a year means its next
   occurrence, and the customer confirms it.
2. A slot is booked once: working hours, service duration and existing
   bookings are checked, also when two customers ask at the same moment.
3. A request that matches no service unambiguously is clarified, not guessed,
   and no price is quoted for it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, time, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from ai_operator.ai import prompts, rules
from ai_operator.ai.schemas import Analysis, AnalyzeRequest, Extraction
from ai_operator.config import BusinessConfig, Schedule
from ai_operator.context import AppContext
from ai_operator.core.message import Category, InboundMessage
from ai_operator.db.models import Invoice, Lead, Message
from ai_operator.pipeline import Pipeline
from ai_operator.queue.ingest import ingest
from ai_operator.scheduling import (
    Interval,
    SlotProblem,
    check_slot,
    nearest_free_slots,
    nearest_future,
)
from tests.conftest import FakeChannel, drain

# Monday 16 November 2026, 12:00 in Yerevan: the moment every customer below writes.
NOW_UTC = datetime(2026, 11, 16, 8, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)


@pytest.fixture
async def shop(ctx: AppContext, real_business: BusinessConfig) -> AsyncIterator[AppContext]:
    """The pipeline with the real demo schedule: Mon-Sat 09:00-19:00, Sunday closed."""
    yield replace(ctx, business=real_business)


def message(text: str, chat: str = "1", msg_id: str = "1") -> InboundMessage:
    return InboundMessage(
        channel="telegram",
        external_id=f"{chat}:{msg_id}",
        sender_id=chat,
        sender_name="Aram",
        text=text,
        received_at=NOW_UTC,
    )


async def say(ctx: AppContext, text: str, chat: str = "1", msg_id: str = "1") -> str:
    """Send one customer message through the whole conveyor; return the bot's reply."""
    fake = ctx.channels.get("telegram")
    assert isinstance(fake, FakeChannel)
    before = len(fake.sent)
    await ingest(ctx, message(text, chat, msg_id))
    await drain(ctx)
    replies = [m.text for m in fake.sent[before:] if m.recipient == chat]
    assert len(replies) == 1
    return replies[0]


async def leads(ctx: AppContext, status: str | None = None) -> list[Lead]:
    async with ctx.sessions() as session:
        query = select(Lead).order_by(Lead.id)
        if status:
            query = query.where(Lead.status == status)
        return list(await session.scalars(query))


async def invoices(ctx: AppContext) -> int:
    async with ctx.sessions() as session:
        return int(await session.scalar(select(func.count()).select_from(Invoice)) or 0)


def local(ctx: AppContext, lead: Lead) -> str:
    assert lead.scheduled_at is not None
    return lead.scheduled_at.astimezone(ctx.business.tz).strftime("%Y-%m-%d %H:%M")


def llm_says(ctx: AppContext, monkeypatch: pytest.MonkeyPatch, **extraction: object) -> None:
    """Make the provider return this extraction, as a real LLM might."""

    async def analyze(request: AnalyzeRequest) -> Analysis:
        return Analysis(
            category=Category.ORDER,
            confidence=0.95,
            language="ru",
            summary="booking request",
            extraction=Extraction.model_validate(extraction),
        )

    monkeypatch.setattr(ctx.provider, "analyze", analyze)


# --------------------------------------------------------------------------- 1. dates


class TestPastDates:
    async def test_explicit_past_date_is_refused_and_free_slots_are_offered(
        self, shop: AppContext
    ) -> None:
        reply = await say(shop, "Запишите на диагностику 20.10.2026 в 16:00, телефон 091123456")

        assert "20.10.2026 16:00 уже прошло" in reply
        assert "Ближайшее свободное время: 16.11.2026 12:30, 16.11.2026 13:00" in reply
        assert "AMD" not in reply
        [lead] = await leads(shop)
        assert lead.status == "incomplete"
        assert lead.scheduled_at is None  # the refused date is not kept
        assert await invoices(shop) == 0

        reply = await say(shop, "Тогда 16.11.2026 в 13:00", msg_id="2")
        [lead] = await leads(shop)
        assert lead.status == "new"
        assert local(shop, lead) == "2026-11-16 13:00"
        assert "Ваша заявка принята" in reply

    async def test_relative_date_in_the_past_is_refused(
        self, shop: AppContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "Yesterday at 16:00": the model resolves it, code refuses it.
        llm_says(
            shop,
            monkeypatch,
            service_code="diagnostics",
            service_certain=True,
            phone="091123456",
            scheduled_at="2026-11-15T16:00",
        )
        reply = await say(shop, "Запишите на диагностику вчера в 16:00, телефон 091123456")
        assert "уже прошло" in reply
        assert await leads(shop, "new") == []

    async def test_earlier_today_is_in_the_past_too(self, shop: AppContext) -> None:
        reply = await say(shop, "Запишите на диагностику сегодня в 10:00, телефон 091123456")
        assert "16.11.2026 10:00 уже прошло" in reply
        assert await leads(shop, "new") == []

    async def test_date_without_year_means_next_occurrence_and_is_confirmed(
        self, shop: AppContext
    ) -> None:
        reply = await say(shop, "Запишите на диагностику 20.10 в 16:00, телефон 091123456")

        assert "Записать вас на 20.10.2027 16:00?" in reply
        [lead] = await leads(shop)
        assert lead.status == "incomplete"
        assert lead.needs_confirmation
        assert await invoices(shop) == 0  # nothing is booked until the customer agrees

        reply = await say(shop, "Да", msg_id="2")
        [lead] = await leads(shop)
        assert lead.status == "new"
        assert not lead.needs_confirmation
        assert local(shop, lead) == "2027-10-20 16:00"
        assert "20.10.2027 16:00" in reply

    async def test_llm_that_keeps_the_current_year_is_corrected_by_code(
        self, shop: AppContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The exact live failure: "20.10" on 16.11.2026 came back as 2026-10-20.
        llm_says(
            shop,
            monkeypatch,
            service_code="brake_pads",
            service_certain=True,
            phone="091123456",
            scheduled_at="2026-10-20T16:00",
            year_stated=False,
        )
        reply = await say(shop, "Запишите на замену колодок 20.10 в 16:00, телефон 091123456")
        assert "Записать вас на 20.10.2027 16:00?" in reply
        assert await leads(shop, "new") == []

    async def test_customer_can_decline_the_proposed_year_and_pick_another_date(
        self, shop: AppContext
    ) -> None:
        await say(shop, "Запишите на диагностику 20.10 в 16:00, телефон 091123456")
        reply = await say(shop, "Нет, давайте завтра в 15:00", msg_id="2")
        [lead] = await leads(shop)
        assert lead.status == "new"
        assert local(shop, lead) == "2026-11-17 15:00"
        assert "Ваша заявка принята" in reply

    async def test_unrelated_reply_does_not_count_as_confirmation(self, shop: AppContext) -> None:
        await say(shop, "Запишите на диагностику 20.10 в 16:00, телефон 091123456")
        reply = await say(shop, "Мой телефон 091 555 444", msg_id="2")
        assert "Записать вас на 20.10.2027 16:00?" in reply
        assert await leads(shop, "new") == []

    async def test_upcoming_date_without_year_needs_no_confirmation(self, shop: AppContext) -> None:
        reply = await say(shop, "Запишите на диагностику 25.11 в 16:00, телефон 091123456")
        [lead] = await leads(shop)
        assert lead.status == "new"
        assert local(shop, lead) == "2026-11-25 16:00"
        assert "Ваша заявка принята" in reply

    def test_nearest_future_rules(self, real_business: BusinessConfig) -> None:
        tz = real_business.tz
        now = NOW_UTC.astimezone(tz)
        passed = datetime(2026, 10, 20, 16, 0, tzinfo=tz)
        assert nearest_future(passed, now, False) == (passed.replace(year=2027), True)
        assert nearest_future(passed, now, True) == (passed, False)  # explicit year: as written
        assert nearest_future(passed, now, None) == (passed, False)  # relative date: as resolved
        ahead = datetime(2026, 12, 1, 10, 0, tzinfo=tz)
        assert nearest_future(ahead.replace(year=2027), now, False) == (ahead, False)
        today_no_time = datetime(2026, 11, 16, 0, 0, tzinfo=tz)
        assert nearest_future(today_no_time, now, False) == (today_no_time, False)
        leap = datetime(2028, 2, 29, 10, 0, tzinfo=tz)
        assert nearest_future(leap, now, False) == (leap, False)

    def test_mock_rules_report_whether_the_year_was_written(self) -> None:
        assert rules.year_stated("20.10 в 16:00") is False
        assert rules.year_stated("20.10.2026 в 16:00") is True
        assert rules.year_stated("5 марта в 10") is False
        assert rules.year_stated("завтра в 10:00") is None
        assert rules.year_stated("просто текст") is None
        assert rules.find_confirmation("Да, подходит") is True
        assert rules.find_confirmation("ok") is True
        assert rules.find_confirmation("Нет, не надо") is False
        assert rules.find_confirmation("Данные верны?") is None


# --------------------------------------------------------------------------- 2. slots


class TestSlots:
    async def test_same_slot_is_not_booked_twice(self, shop: AppContext) -> None:
        first = await say(shop, "Запишите на замену колодок завтра в 16:00, тел 091123456", "1")
        second = await say(shop, "Запишите на диагностику завтра в 16:00, тел 091555444", "2")

        assert "Ваша заявка принята" in first
        assert "17.11.2026 16:00 уже занято" in second
        assert "Ближайшее свободное время:" in second
        assert "17.11.2026 16:00," not in second.split("Ближайшее")[1]
        assert "AMD" not in second
        booked = await leads(shop, "new")
        assert [lead.sender_id for lead in booked] == ["1"]
        assert await invoices(shop) == 1

        # The second customer takes one of the offered slots.
        third = await say(shop, "Тогда завтра в 17:00", "2", "2")
        assert "Ваша заявка принята" in third
        assert [local(shop, lead) for lead in await leads(shop, "new")] == [
            "2026-11-17 16:00",
            "2026-11-17 17:00",
        ]

    async def test_overlap_is_checked_with_the_service_duration(self, shop: AppContext) -> None:
        # Brake pads take 60 minutes: 16:00-17:00.
        await say(shop, "Запишите на замену колодок завтра в 16:00, тел 091123456", "1")
        inside = await say(shop, "Запишите на диагностику завтра в 16:30, тел 091555444", "2")
        assert "уже занято" in inside
        # Diagnostics take 30 minutes: 15:45-16:15 would run into the booked hour.
        before = await say(shop, "Запишите на диагностику завтра в 15:45, тел 091555445", "3")
        assert "уже занято" in before
        after = await say(shop, "Запишите на диагностику завтра в 17:00, тел 091555446", "4")
        assert "Ваша заявка принята" in after
        assert len(await leads(shop, "new")) == 2

    async def test_two_customers_at_the_same_moment_get_one_slot(self, shop: AppContext) -> None:
        text = "Запишите на замену масла завтра в 11:00, тел 09112345{}"
        ids = [(await ingest(shop, message(text.format(i), str(i))))[0] for i in range(1, 5)]
        pipeline = Pipeline(shop)

        await asyncio.gather(*(pipeline.process(message_id) for message_id in ids))

        booked = await leads(shop, "new")
        assert len(booked) == 1
        assert local(shop, booked[0]) == "2026-11-17 11:00"
        assert len(await leads(shop, "incomplete")) == 3
        async with shop.sessions() as session:
            statuses = list(await session.scalars(select(Message.status)))
        assert statuses == ["answered"] * 4

    async def test_customer_can_move_their_own_unfinished_booking(self, shop: AppContext) -> None:
        # The same customer repeating their own slot is not a conflict with themselves.
        await say(shop, "Запишите на диагностику завтра в 16:00", "1")
        reply = await say(shop, "завтра в 16:00, телефон 091123456", "1", "2")
        assert "Ваша заявка принята" in reply
        assert len(await leads(shop)) == 1

    async def test_day_off_is_refused(self, shop: AppContext) -> None:
        reply = await say(shop, "Запишите на диагностику 22.11.2026 в 12:00, тел 091123456")
        assert "вне рабочего времени" in reply
        assert "Пн-Сб 09:00-19:00" in reply
        assert "21.11.2026" in reply  # nearest free slots around the closed Sunday
        assert await leads(shop, "new") == []

    async def test_visit_must_finish_before_closing(self, shop: AppContext) -> None:
        late = await say(shop, "Запишите на замену колодок завтра в 18:30, тел 091123456", "1")
        assert "вне рабочего времени" in late
        assert "17.11.2026 18:00" in late  # the last start that still fits a 60-minute job
        # Engine repair is an 8-hour job: at 12:00 it cannot be finished by 19:00.
        long = await say(shop, "Запишите на ремонт двигателя завтра в 12:00, тел 091555444", "2")
        assert "вне рабочего времени" in long
        assert "AMD" not in long
        fits = await say(shop, "Запишите на замену колодок завтра в 18:00, тел 091555445", "3")
        assert "Ваша заявка принята" in fits

    async def test_date_without_time_asks_for_a_time(self, shop: AppContext) -> None:
        reply = await say(shop, "Запишите на диагностику завтра, тел 091123456")
        assert "На какое время 17.11.2026 вас записать?" in reply
        assert "Свободное время в этот день: 09:00, 12:00, 15:30, 18:30." in reply
        assert await leads(shop, "new") == []

        reply = await say(shop, "в 15:00", msg_id="2")
        [lead] = await leads(shop)
        assert lead.status == "new"
        assert local(shop, lead) == "2026-11-17 15:00"

    def test_slot_checks(self, real_business: BusinessConfig) -> None:
        schedule, tz = real_business.schedule, real_business.tz
        now = NOW_UTC.astimezone(tz)
        four = datetime(2026, 11, 17, 16, 0, tzinfo=tz)
        busy = [Interval(four, four + HOUR)]

        assert check_slot(schedule, four, HOUR, now, []) is None
        assert check_slot(schedule, four, HOUR, now, busy) is SlotProblem.BUSY
        assert check_slot(schedule, four + HOUR, HOUR, now, busy) is None  # back to back
        assert check_slot(schedule, now - HOUR, HOUR, now, []) is SlotProblem.PAST
        assert check_slot(schedule, four.replace(day=22), HOUR, now, []) is SlotProblem.CLOSED
        assert check_slot(schedule, four.replace(hour=8), HOUR, now, []) is SlotProblem.CLOSED
        assert check_slot(schedule, four.replace(hour=18, minute=30), HOUR, now, []) is (
            SlotProblem.CLOSED
        )
        assert check_slot(schedule, four.replace(hour=0), HOUR, now, []) is SlotProblem.TIME_NEEDED

        assert nearest_free_slots(schedule, four, HOUR, now, busy) == [
            four.replace(hour=14, minute=30),
            four.replace(hour=15),
            four.replace(hour=17),
        ]
        # Nothing in the past is ever offered.
        assert all(s > now for s in nearest_free_slots(schedule, now - 5 * HOUR, HOUR, now, []))

    def test_two_parallel_bookings_when_the_config_allows_it(
        self, real_business: BusinessConfig
    ) -> None:
        tz = real_business.tz
        schedule = real_business.schedule.model_copy(update={"parallel_bookings": 2})
        four = datetime(2026, 11, 17, 16, 0, tzinfo=tz)
        one = [Interval(four, four + HOUR)]
        now = NOW_UTC.astimezone(tz)
        assert check_slot(schedule, four, HOUR, now, one) is None
        assert check_slot(schedule, four, HOUR, now, one * 2) is SlotProblem.BUSY

    def test_schedule_is_validated(self) -> None:
        assert Schedule().hours_on(datetime(2026, 11, 22).date()) is None  # Sunday by default
        assert Schedule().hours_on(datetime(2026, 11, 16).date()) == (time(9), time(19))
        with pytest.raises(ValidationError, match="unknown weekday"):
            Schedule(hours={"funday": (time(9), time(18))})
        with pytest.raises(ValidationError, match="before closing"):
            Schedule(hours={"mon": (time(19), time(9))})


# --------------------------------------------------------------------------- 3. services


class TestUnclearService:
    async def test_uncertain_match_is_clarified_without_a_price(
        self, shop: AppContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The exact live failure: the model mapped an accident repair to "engine repair".
        llm_says(
            shop,
            monkeypatch,
            service_code="engine_repair",
            service_certain=False,
            service_text="починить машину после аварии",
            phone="091123456",
            scheduled_at="2026-11-17T16:00",
        )
        reply = await say(shop, "Хочу починить машину после аварии завтра в 16:00, 091123456")

        assert "не хочу угадывать" in reply
        assert "Замена масла" in reply and "Ремонт двигателя" in reply  # the list to choose from
        assert "можно начать с услуги «Компьютерная диагностика»" in reply
        assert "AMD" not in reply and "60 000" not in reply
        [lead] = await leads(shop)
        assert lead.status == "incomplete"
        assert lead.service_code is None
        assert lead.amount is None
        assert lead.service_text == "починить машину после аварии"
        assert await invoices(shop) == 0

    async def test_customer_then_chooses_a_service_and_is_booked(self, shop: AppContext) -> None:
        first = await say(shop, "Хочу записаться, починить машину после аварии, 091123456")
        assert "Мы выполняем:" in first
        assert "AMD" not in first

        reply = await say(shop, "Давайте диагностику завтра в 16:00", msg_id="2")
        [lead] = await leads(shop)
        assert lead.status == "new"
        assert lead.service_code == "diagnostics"
        assert "8 000 AMD" in reply  # a price appears only for the service actually chosen

    async def test_ambiguous_keywords_are_not_resolved_by_the_rules(self, shop: AppContext) -> None:
        # Two services match equally well: the rule-based provider must not pick one.
        reply = await say(shop, "Запишите на диагностику или кондиционер завтра в 16:00, 091123456")
        assert "Мы выполняем:" in reply
        assert "AMD" not in reply
        assert await leads(shop, "new") == []

    def test_keyword_must_start_a_word(self) -> None:
        keywords = {"tire_fitting": ("шин", "tire"), "oil_change": ("масл", "oil")}
        assert rules.match_service("починить машину после аварии", keywords) == (None, False)
        assert rules.match_service("the entire car needs a look", keywords) == (None, False)
        assert rules.match_service("поменять шины", keywords) == ("tire_fitting", True)
        assert rules.match_service("шины и масло", keywords) == ("tire_fitting", False)  # a tie

    def test_untrusted_service_is_dropped(self) -> None:
        guess = Extraction(service_code="engine_repair", service_certain=False, name="Aram")
        assert guess.trusted().service_code is None
        assert guess.trusted().name == "Aram"
        sure = Extraction(service_code="engine_repair", service_certain=True)
        assert sure.trusted() is sure
        # A service chosen earlier survives a later message that names none.
        merged = Extraction(phone="091123456").trusted().merged_over(sure)
        assert merged.service_code == "engine_repair" and merged.service_certain

    def test_prompt_tells_the_model_not_to_guess(self) -> None:
        system = prompts.analyze_system("Demo", "Asia/Yerevan")
        assert "Do not pick the closest-sounding one" in system
        assert "service_certain" in system
        assert "year_stated" in system
        assert "do not move it" in system

    def test_fallback_service_must_exist(self, real_business: BusinessConfig) -> None:
        assert real_business.fallback_service == "diagnostics"
        data = real_business.model_dump()
        data["fallback_service"] = "teleportation"
        with pytest.raises(ValidationError, match="not in services"):
            BusinessConfig.model_validate(data)
