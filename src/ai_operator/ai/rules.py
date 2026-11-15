"""Deterministic text rules in English, Russian and Armenian.

These power the ``mock`` provider: a keyword classifier, a small date/time
parser and regex extractors. They are a transparent baseline - the evals show
exactly how far rules alone get you compared to a real LLM.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from ai_operator.core.message import Category

# ---------------------------------------------------------------------------
# Classification lexicons: lower-case substrings ("stems") per category.
# ---------------------------------------------------------------------------

SPAM = (
    "casino", "crypto", "bitcoin", "seo ", "earn money", "earn $", "click here", "limited offer",
    "followers", "loan approved", "lottery", "you have won", "you won", "viagra", "backlinks",
    "promote your", "boost your", "investment opportunity", "passive income", "unsubscribe",
    "казино", "крипт", "биткоин", "заработ", "раскрутк", "продвижени", "подписчик", "кредит без",
    "вы выиграли", "лотере", "перейдите по ссылке", "пассивный доход", "инвестиц", "рассылк",
    "կազինո", "կրիպտո", "բիթքոին", "վաստակ", "շահել եք", "վիճակախաղ", "սեղմեք հղման",
    "առաջխաղացում", "պասիվ եկամուտ", "ներդրում",
)  # fmt: skip

COMPLAINT = (
    "complain", "terrible", "awful", "horrible", "worst", "refund", "money back", "scam",
    "rude", "ruined", "damaged", "scratched", "disappointed", "unacceptable", "still broken",
    "not fixed", "overcharged", "never again", "poor service", "angry", "furious", "outrageous",
    "ripped off", "still leaking", "still making", "worse than", "waited for",
    "жалоб", "ужасн", "отвратительн", "кошмар", "верните деньги", "возврат денег", "вернуть деньги",
    "обман", "хамств", "хамит", "нагруб", "испортил", "поцарап", "сломали", "недовол", "разочаров",
    "безобрази", "не починили", "не исправили", "так и не", "стало хуже", "больше не приду",
    "развод", "некачествен", "прождал", "возмущ", "претензи",
    "բողոք", "սարսափելի", "ամենավատ", "դժգոհ", "հիասթափ", "փչացր", "քերծ", "վնասել", "խաբ",
    "կոպիտ", "գումարը վերադարձ", "վերադարձրեք", "անընդունելի", "չեն վերանորոգ", "չվերանորոգ",
    "այլևս չեմ գա", "անորակ", "վրդով", "ավելի վատ", "խայտառակ",
)  # fmt: skip

ORDER = (
    "book", "appointment", "schedule me", "sign me up", "reserve", "come in", "bring my car",
    "bring the car", "register me", "a slot", "i'd like to get", "i would like to get",
    "i want to get", "i need to get", "put me down",
    "запис", "запиш", "заброниров", "бронь", "приеду", "подъеду", "хочу приехать", "хочу поменять",
    "хочу сделать", "хочу заменить", "нужно поменять", "надо поменять", "нужно заменить",
    "оформить заказ", "заказать", "привезу", "хотел бы", "хотела бы",
    "գրանց", "ամրագր", "կգամ", "ուզում եմ գալ", "կբերեմ", "ուզում եմ փոխել", "պետք է փոխեմ",
    "պատվիր", "ժամ նշանակ", "հերթագր", "կցանկանայի",
)  # fmt: skip

QUESTION = (
    "how much", "price", "cost", "what time", "when ", "where ", "do you", "can you", "is it",
    "are you", "what ", "which ", "how long", "address", "open", "warranty", "guarantee",
    "payment", "pay by", "how do", "is there",
    "сколько", "цена", "цены", "стоит", "стоимость", "когда", "где ", "как ", "можно ли",
    "есть ли", "какой", "какая", "какие", "работаете", "адрес", "гаранти", "оплат", "подскаж",
    "скажите", "почём", "почем",
    "որքան", "ինչքան", "ինչ արժե", "արժե", "գին", "երբ ", "որտեղ", "ինչպես", "հասցե",
    "աշխատում եք", "երաշխիք", "վճար", "կարելի", "ունեք", "ասեք", "ինչ ",
)  # fmt: skip

SMALL_TALK = (
    "hello", "hi ", "good morning", "good afternoon", "thanks", "thank you", "ok", "bye",
    "привет", "здравствуйте", "добрый день", "спасибо", "благодарю", "хорошо", "понятно",
    "բարև", "ողջույն", "շնորհակալ", "լավ", "հասկացա", "ցտեսություն",
)  # fmt: skip

_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_PHONE = re.compile(r"(?<![\w.])\+?\d(?:[\s\-()]{0,2}\d){7,14}(?![\w])")


def _hits(text: str, lexicon: tuple[str, ...]) -> int:
    return sum(1 for stem in lexicon if stem in text)


def classify(
    text: str, *, has_pending_booking: bool, has_booking_data: bool
) -> tuple[Category, float]:
    """Keyword scoring. Returns the category and a confidence in 0..1."""
    low = f" {text.lower()} "
    scores = {
        Category.SPAM: 2.5 * _hits(low, SPAM) + (1.5 if _URL.search(low) else 0.0),
        Category.COMPLAINT: 2.5 * _hits(low, COMPLAINT),
        Category.ORDER: 2.5 * _hits(low, ORDER),
        Category.QUESTION: min(2.5, 1.0 * _hits(low, QUESTION))
        + (1.5 if "?" in low or "՞" in low else 0.0),
    }
    if has_booking_data:
        scores[Category.ORDER] += 1.5
    if has_pending_booking and has_booking_data:
        # The customer is answering our "please send your phone" request.
        scores[Category.ORDER] += 3.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top, top_score), (_, second_score) = ranked[0], ranked[1]
    if top_score == 0.0:
        confident = _hits(low, SMALL_TALK) > 0
        return Category.OTHER, 0.9 if confident else 0.5
    margin = (top_score - second_score) / top_score
    confidence = 0.55 + 0.4 * margin
    if top_score < 2.0:
        confidence = min(confidence, 0.6)
    return top, round(min(confidence, 0.97), 2)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_NAME_PATTERNS = (
    re.compile(
        r"(?i:my name is|my name's|i am|i'm|this is|name:)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"
    ),
    re.compile(
        r"(?i:меня зовут|мо[её] имя|имя:|зовут)\s+[-—]?\s*([А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)?)"
    ),
    re.compile(r"(?:[Աա]նունս|[Իի]մ անունը|[Աա]նունը)՝?\s+([Ա-Ֆ][ա-ֆև]+)"),
    re.compile(r"(?:^|[\s,.])[Եե]ս\s+([Ա-Ֆ][ա-ֆև]+?)ն?\s+եմ"),
)
_NOT_NAMES = frozenset({"Interested", "Looking", "Writing", "Calling", "Here", "Not", "Very", "So"})

_AMOUNT = re.compile(r"(\d[\d\s.,]*\d|\d+)\s*(?:amd|dram|драм|դրամ|֏)", re.IGNORECASE)


def find_phone(text: str) -> str | None:
    match = _PHONE.search(text)
    return match.group(0).strip() if match else None


def find_name(text: str) -> str | None:
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match and match.group(1).split()[0] not in _NOT_NAMES:
            return match.group(1)
    return None


def find_amount(text: str) -> float | None:
    match = _AMOUNT.search(text)
    if not match:
        return None
    digits = re.sub(r"[^\d]", "", match.group(1))
    return float(digits) if digits else None


def find_service(text: str, keywords_by_code: dict[str, tuple[str, ...]]) -> str | None:
    """The service whose keywords occur most often in the text."""
    low = text.lower()
    best_code, best_hits = None, 0
    for code, keywords in keywords_by_code.items():
        hits = sum(1 for kw in keywords if kw.lower() in low)
        if hits > best_hits:
            best_code, best_hits = code, hits
    return best_code


# ---------------------------------------------------------------------------
# Dates and times
# ---------------------------------------------------------------------------

_WEEKDAYS: tuple[tuple[str, ...], ...] = (
    ("monday", "понедельник", "երկուշաբթի"),
    ("tuesday", "вторник", "երեքշաբթի"),
    ("wednesday", "сред", "չորեքշաբթի"),
    ("thursday", "четверг", "հինգշաբթի"),
    ("friday", "пятниц", "ուրբաթ"),
    ("saturday", "суббот", "շաբաթ օր"),
    ("sunday", "воскресень", "կիրակի"),
)
_MONTHS: tuple[tuple[str, ...], ...] = (
    ("january", "январ", "հունվար"),
    ("february", "феврал", "փետրվար"),
    ("march", "март", "մարտ"),
    ("april", "апрел", "ապրիլ"),
    ("may", "мая", "մայիս"),
    ("june", "июн", "հունիս"),
    ("july", "июл", "հուլիս"),
    ("august", "август", "օգոստոս"),
    ("september", "сентябр", "սեպտեմբեր"),
    ("october", "октябр", "հոկտեմբեր"),
    ("november", "ноябр", "նոյեմբեր"),
    ("december", "декабр", "դեկտեմբեր"),
)
_DAY_AFTER_TOMORROW = ("day after tomorrow", "послезавтра", "վաղը չէ մյուս օրը", "մյուս օրը")
_TOMORROW = ("tomorrow", "завтра", "վաղը")
_TODAY = ("today", "сегодня", "այսօր")

_CLOCK = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_HOUR_WORD = re.compile(
    r"(?:\bat|\bв|\bк|ժամը)\s*(\d{1,2})(?!\s*[.:/]\d)\s*(am|pm|утра|дня|вечера|часов|часа|час)?"
    r"|(\d{1,2})\s*(am|pm)\b|(\d{1,2})-ին",
    re.IGNORECASE,
)
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})(?:[./](\d{4}|\d{2}))?(?!\d)")
_DAY_NUMBER = re.compile(r"(?<!\d)(\d{1,2})(?!\d)")
_EVENING = ("pm", "дня", "вечера")


def _find_time(low: str) -> tuple[int, int] | None:
    clock = _CLOCK.search(low)
    if clock:
        return int(clock.group(1)), int(clock.group(2))
    word = _HOUR_WORD.search(low)
    if not word:
        return None
    hour = int(word.group(1) or word.group(3) or word.group(5))
    suffix = (word.group(2) or word.group(4) or "").lower()
    if suffix in _EVENING and hour < 12:
        hour += 12
    return (hour, 0) if 0 <= hour <= 23 else None


def _find_date(low: str, now: datetime) -> datetime | None:
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if any(w in low for w in _DAY_AFTER_TOMORROW):
        return today + timedelta(days=2)
    if any(w in low for w in _TOMORROW):
        return today + timedelta(days=1)
    if any(w in low for w in _TODAY):
        return today

    numeric = _NUMERIC_DATE.search(low)
    if numeric:
        day, month = int(numeric.group(1)), int(numeric.group(2))
        year = int(numeric.group(3)) if numeric.group(3) else today.year
        year += 2000 if year < 100 else 0
        return _safe_date(today, year, month, day, explicit_year=bool(numeric.group(3)))

    for month_index, names in enumerate(_MONTHS, start=1):
        for name in names:
            position = low.find(name)
            if position == -1:
                continue
            window = low[max(0, position - 6) : position + len(name) + 12]
            day_match = _DAY_NUMBER.search(window)
            if day_match:
                return _safe_date(today, today.year, month_index, int(day_match.group(1)), False)

    for weekday_index, names in enumerate(_WEEKDAYS):
        if any(name in low for name in names):
            ahead = (weekday_index - today.weekday()) % 7 or 7
            return today + timedelta(days=ahead)
    return None


def _safe_date(
    today: datetime, year: int, month: int, day: int, explicit_year: bool
) -> datetime | None:
    try:
        candidate = today.replace(year=year, month=month, day=day)
    except ValueError:
        return None
    if candidate < today and not explicit_year:
        candidate = candidate.replace(year=year + 1)
    return candidate


def parse_datetime(text: str, now: datetime) -> datetime | None:
    """Find a requested date/time in the text, relative to ``now`` (naive local time).

    A date without a time comes back at 00:00, which downstream means "time not set".
    """
    low = text.lower()
    date = _find_date(low, now)
    time = _find_time(low)
    if date is None and time is None:
        return None
    if date is None:
        assert time is not None
        candidate = now.replace(hour=time[0], minute=time[1], second=0, microsecond=0)
        return candidate if candidate > now else candidate + timedelta(days=1)
    if time is None:
        return date
    return date.replace(hour=time[0], minute=time[1])
