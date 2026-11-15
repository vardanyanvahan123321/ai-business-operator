"""Tiny translation layer for the three supported languages.

Catalogues are flat JSON files in ``ai_operator/locales``. A test guarantees
that all three contain exactly the same keys and placeholders.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai_operator.config import LANGS

LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"
FALLBACK = "en"


@lru_cache
def load_catalog(lang: str) -> dict[str, str]:
    with (LOCALES_DIR / f"{lang}.json").open(encoding="utf-8") as fh:
        data: dict[str, str] = json.load(fh)
    return data


def normalize_lang(lang: str | None) -> str:
    return lang if lang in LANGS else FALLBACK


def t(lang: str | None, key: str, **params: Any) -> str:
    """Translate ``key`` into ``lang`` and fill in ``{placeholders}``."""
    catalog = load_catalog(normalize_lang(lang))
    template = catalog.get(key)
    if template is None:
        template = load_catalog(FALLBACK).get(key, key)
    return template.format(**params) if params else template
