"""Cheap, deterministic language detection for the three supported languages.

Armenian and Russian have their own scripts, so counting letters per script is
both faster and more reliable here than a statistical model. Anything else
falls back to English.
"""

from __future__ import annotations

from ai_operator.config import Lang


def detect_language(text: str, default: Lang = "en") -> Lang:
    armenian = cyrillic = latin = 0
    for ch in text:
        code = ord(ch)
        if 0x0530 <= code <= 0x058F or 0xFB13 <= code <= 0xFB17:
            armenian += 1
        elif 0x0400 <= code <= 0x04FF:
            cyrillic += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    best = max(armenian, cyrillic, latin)
    if best == 0:
        return default
    if armenian == best:
        return "hy"
    if cyrillic == best:
        return "ru"
    return "en"
