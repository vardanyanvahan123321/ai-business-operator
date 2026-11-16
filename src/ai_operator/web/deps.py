"""Shared web plumbing: templates, language, authentication, CSRF."""

from __future__ import annotations

import secrets
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from ai_operator import __version__
from ai_operator.config import LANGS
from ai_operator.context import AppContext
from ai_operator.core.formatting import format_datetime, format_money
from ai_operator.core.i18n import normalize_lang, t

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
LANG_COOKIE = "abo_lang"


class LoginRequired(Exception):
    """Raised by ``require_admin``; turned into a redirect to the login page."""


def get_ctx(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


def current_lang(request: Request, default: str | None = None) -> str:
    """Language: explicit ``?lang=`` wins, then the cookie, then the browser, then default."""
    explicit = request.query_params.get("lang")
    if explicit in LANGS:
        return explicit
    cookie = request.cookies.get(LANG_COOKIE)
    if cookie in LANGS:
        return cookie
    header = request.headers.get("accept-language", "")
    for part in header.split(","):
        code = part.split(";")[0].strip().lower()[:2]
        if code in LANGS:
            return code
    return normalize_lang(default)


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(24)
        request.session["csrf"] = token
    return str(token)


async def verify_csrf(request: Request) -> None:
    """Reject state-changing requests that do not carry the session's CSRF token."""
    expected = request.session.get("csrf")
    supplied = request.headers.get("x-csrf-token")
    if supplied is None:
        form = await request.form()
        value = form.get("csrf_token")
        supplied = value if isinstance(value, str) else None
    if not expected or not supplied or not secrets.compare_digest(str(expected), supplied):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")


def require_admin(request: Request) -> None:
    if not request.session.get("admin"):
        raise LoginRequired


def render(
    request: Request, name: str, context: dict[str, Any] | None = None, status_code: int = 200
) -> Response:
    ctx = get_ctx(request)
    lang = current_lang(request, ctx.business.default_language)
    base: dict[str, Any] = {
        "lang": lang,
        "langs": LANGS,
        "t": partial(t, lang),
        "business": ctx.business,
        "company": ctx.business.name.get(lang),
        "demo_mode": ctx.settings.demo_mode,
        "version": __version__,
        "csrf_token": csrf_token(request),
        "is_admin": bool(request.session.get("admin")),
        "path": request.url.path,
        "fmt_dt": lambda value: format_datetime(value, lang, ctx.business.tz),
        "fmt_money": format_money,
        "fmt_ts": lambda value: (
            f"{value.astimezone(ctx.business.tz):%d.%m %H:%M}" if value else "-"
        ),
    }
    base.update(context or {})
    response = templates.TemplateResponse(request, name, base, status_code=status_code)
    if request.query_params.get("lang") in LANGS:
        response.set_cookie(LANG_COOKIE, lang, max_age=365 * 24 * 3600, samesite="lax")
    return response


class Throttle:
    """At most N recorded events per time window per client address.

    Used for failed sign-ins (slows down password guessing) and for the public
    form (one address cannot flood the queue and burn the AI quota). State is in
    memory: it is per process and resets on restart, which is enough for a
    single web container.
    """

    def __init__(self, max_events: int = 5, window_seconds: int = 300) -> None:
        self._max = max_events
        self._window = window_seconds
        self._events: dict[str, list[float]] = defaultdict(list)

    def _recent(self, client: str) -> list[float]:
        cutoff = time.monotonic() - self._window
        recent = [stamp for stamp in self._events[client] if stamp > cutoff]
        if recent:
            self._events[client] = recent
        else:
            self._events.pop(client, None)  # do not keep an entry per visitor forever
        return recent

    def blocked(self, client: str) -> bool:
        return self._max > 0 and len(self._recent(client)) >= self._max

    def record(self, client: str) -> None:
        self._events[client].append(time.monotonic())

    def reset(self, client: str) -> None:
        self._events.pop(client, None)


def client_address(request: Request) -> str:
    return request.client.host if request.client else "unknown"
