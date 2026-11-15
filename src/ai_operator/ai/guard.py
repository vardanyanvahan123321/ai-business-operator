"""Prompt-injection defence.

A customer message is *data*. Three independent layers make sure it stays data:

1. ``detect_injection`` - cheap pattern check before any model call. A hit sends
   the message to a human instead of letting the model answer freely.
2. ``wrap_untrusted`` - the text is sanitised and put inside explicit delimiters;
   the system prompt tells the model that nothing inside them is an instruction.
3. Output is never trusted either: the model only fills validated schemas, the
   answer may cite only sources we retrieved ourselves, and ``looks_leaky``
   blocks replies that echo internal instructions.
"""

from __future__ import annotations

import re
import unicodedata

MAX_INPUT_CHARS = 4000

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE | re.UNICODE))
    for name, pattern in (
        # --- English ---------------------------------------------------------
        (
            "override_instructions",
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(instruction|prompt|rule|guideline|above|previous|prior)s?\b",
        ),
        (
            "reveal_prompt",
            r"\b(reveal|show|print|repeat|output|leak|tell me)\b[^.\n]{0,40}"
            r"\b(system prompt|your prompt|your instructions|initial instructions|hidden rules)\b",
        ),
        (
            "role_hijack",
            r"\b(you are now|from now on you|act as|pretend (to be|you are)|"
            r"developer mode|jailbreak|do anything now|\bDAN\b)",
        ),
        (
            "fake_role_tag",
            r"(^|\n)\s*(system|assistant|developer)\s*:|</?\s*(system|assistant)\s*>",
        ),
        ("fake_delimiter", r"(#{3,}|-{5,}|={5,})\s*(system|new instructions|end of)"),
        # --- Russian ---------------------------------------------------------
        (
            "override_instructions_ru",
            r"(игнорируй|проигнорируй|забудь|отмени|не обращай внимания на)[^.\n]{0,40}"
            r"(инструкци|правил|промпт|указани|предыдущ|выше)",
        ),
        (
            "reveal_prompt_ru",
            r"(покажи|выведи|раскрой|напиши|повтори|расскажи)[^.\n]{0,40}"
            r"(системн\w+ (промпт|сообщени|инструкци)|сво[ий] (промпт|инструкци)|"
            r"скрыт\w+ (правил|инструкци))",
        ),
        ("role_hijack_ru", r"(ты теперь|теперь ты|с этого момента ты|притворись|веди себя как)"),
        # --- Armenian --------------------------------------------------------
        (
            "override_instructions_hy",
            r"(անտեսիր|անտեսեք|մոռացիր|մոռացեք|չեղարկիր)[^.\n]{0,40}"
            r"(հրահանգ|կանոն|ցուցում|նախորդ|պրոմպտ)",
        ),
        (
            "reveal_prompt_hy",
            r"(ցույց տուր|գրիր|բացահայտիր|կրկնիր)[^.\n]{0,40}"
            r"(համակարգային|քո հրահանգ|թաքնված կանոն|պրոմպտ)",
        ),
        ("role_hijack_hy", r"(դու հիմա|այսուհետ դու|ձևացիր)"),
    )
)

_LEAK_MARKERS = re.compile(
    r"(system prompt|<customer_message|untrusted data|my instructions are|"
    r"системный промпт|мои инструкции|համակարգային հրահանգ)",
    re.IGNORECASE,
)

OPEN_TAG = "<customer_message>"
CLOSE_TAG = "</customer_message>"


def detect_injection(text: str) -> list[str]:
    """Return the names of all injection patterns found in the text."""
    normalized = unicodedata.normalize("NFKC", text)
    return [name for name, pattern in _PATTERNS if pattern.search(normalized)]


def sanitize(text: str, limit: int = MAX_INPUT_CHARS) -> str:
    """Normalise, drop control characters, defuse our own delimiters, cap the length."""
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = "".join(ch for ch in normalized if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    cleaned = re.sub(r"</?\s*customer_message\s*>", "[tag removed]", cleaned, flags=re.IGNORECASE)
    if len(cleaned) > limit:
        cleaned = cleaned[:limit] + " [truncated]"
    return cleaned.strip()


def wrap_untrusted(text: str) -> str:
    return f"{OPEN_TAG}\n{sanitize(text)}\n{CLOSE_TAG}"


def looks_leaky(reply: str) -> bool:
    """True if a generated reply seems to expose internal instructions."""
    return bool(_LEAK_MARKERS.search(reply))
