"""Booking calendar rules: plain, testable functions with no I/O.

The AI only reports which date and time the customer asked for. Whether that
slot can actually be booked - not in the past, inside working hours, not
already taken - is decided here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from ai_operator.config import Schedule

SEARCH_DAYS = 14


class SlotProblem(StrEnum):
    PAST = "past"  # the requested moment has already passed
    CLOSED = "closed"  # a day off, or the visit would not fit into working hours
    BUSY = "busy"  # another booking occupies that time
    TIME_NEEDED = "time_needed"  # a date without a time: nothing to reserve yet


@dataclass(frozen=True, slots=True)
class Interval:
    start: datetime
    end: datetime

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return self.start < end and start < self.end


def has_time(when: datetime) -> bool:
    """Midnight is how "a date without a time" is represented."""
    return (when.hour, when.minute) != (0, 0)


def is_past(when: datetime, now: datetime) -> bool:
    if not has_time(when):
        return when.date() < now.date()
    return when <= now


def nearest_future(
    when: datetime, now: datetime, year_stated: bool | None
) -> tuple[datetime, bool]:
    """Resolve a calendar date written without a year ("20.10") to its next occurrence.

    Returns the date and whether it had to move to a later year - in that case
    the customer is asked to confirm instead of being booked a year ahead silently.
    Dates with an explicit year and relative dates ("tomorrow") are left alone.
    """
    if year_stated is not False:
        return when, False
    try:
        this_year = when.replace(year=now.year)
    except ValueError:  # 29 February
        return when, False
    if not is_past(this_year, now):
        return this_year, False
    try:
        return this_year.replace(year=now.year + 1), True
    except ValueError:
        return when, False


def fits_hours(schedule: Schedule, start: datetime, duration: timedelta) -> bool:
    hours = schedule.hours_on(start.date())
    if hours is None:
        return False
    end = start + duration
    return end.date() == start.date() and hours[0] <= start.time() and end.time() <= hours[1]


def is_free(schedule: Schedule, start: datetime, duration: timedelta, busy: list[Interval]) -> bool:
    end = start + duration
    return sum(1 for b in busy if b.overlaps(start, end)) < schedule.parallel_bookings


def check_slot(
    schedule: Schedule,
    start: datetime,
    duration: timedelta,
    now: datetime,
    busy: list[Interval],
) -> SlotProblem | None:
    if is_past(start, now):
        return SlotProblem.PAST
    if schedule.hours_on(start.date()) is None:
        return SlotProblem.CLOSED
    if not has_time(start):
        return SlotProblem.TIME_NEEDED
    if not fits_hours(schedule, start, duration):
        return SlotProblem.CLOSED
    if not is_free(schedule, start, duration, busy):
        return SlotProblem.BUSY
    return None


def free_slots_on(
    schedule: Schedule,
    day: datetime,
    duration: timedelta,
    now: datetime,
    busy: list[Interval],
    count: int = 4,
) -> list[datetime]:
    """Up to ``count`` free start times on one day, spread from opening to closing."""
    hours = schedule.hours_on(day.date())
    if hours is None:
        return []
    free: list[datetime] = []
    start = day.replace(hour=hours[0].hour, minute=hours[0].minute, second=0, microsecond=0)
    while fits_hours(schedule, start, duration):
        if start > now and is_free(schedule, start, duration, busy):
            free.append(start)
        start += timedelta(minutes=schedule.slot_minutes)
    if len(free) <= count:
        return free
    last = len(free) - 1
    return [free[round(i * last / (count - 1))] for i in range(count)]


def nearest_free_slots(
    schedule: Schedule,
    target: datetime,
    duration: timedelta,
    now: datetime,
    busy: list[Interval],
    count: int = 3,
) -> list[datetime]:
    """Up to ``count`` free start times closest to ``target``, in chronological order."""
    step = timedelta(minutes=schedule.slot_minutes)
    target = max(target, now)
    found: list[datetime] = []
    day = target.date() - timedelta(days=1)
    for _ in range(SEARCH_DAYS + 1):
        hours = schedule.hours_on(day)
        if hours is not None:
            start = target.replace(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=hours[0].hour,
                minute=hours[0].minute,
                second=0,
                microsecond=0,
            )
            while fits_hours(schedule, start, duration):
                if start > now and is_free(schedule, start, duration, busy):
                    found.append(start)
                start += step
        day += timedelta(days=1)
        # Later days can only be further away than what we already have.
        if day > target.date() and len(found) >= count:
            later = [s for s in found if s >= target]
            if len(later) >= count or day > target.date() + timedelta(days=2):
                break
    found.sort(key=lambda s: (abs((s - target).total_seconds()), s))
    return sorted(found[:count])
