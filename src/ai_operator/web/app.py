"""FastAPI application: public request form, webhooks, health check and the admin panel."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

from ai_operator import __version__
from ai_operator.config import get_settings
from ai_operator.context import AppContext
from ai_operator.core.logging import get_logger
from ai_operator.runtime import build_context
from ai_operator.web import panel, public
from ai_operator.web.deps import WEB_DIR, LoginRequired, Throttle

log = get_logger(__name__)


def create_app(ctx: AppContext | None = None) -> FastAPI:
    """Build the app. Tests pass a ready context; production builds one on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = ctx is None
        app.state.ctx = ctx or await build_context()
        log.info("web.started", provider=app.state.ctx.provider.name)
        yield
        if owned:
            await app.state.ctx.close()

    app = FastAPI(
        title="AI Business Operator", version=__version__, lifespan=lifespan, docs_url=None
    )
    if ctx is not None:
        app.state.ctx = ctx
    settings = ctx.settings if ctx is not None else get_settings()
    app.state.login_throttle = Throttle()
    app.state.form_throttle = Throttle(
        max_events=settings.webform_max_messages, window_seconds=settings.webform_window_seconds
    )

    app.add_middleware(
        SessionMiddleware,
        # Without a configured secret, sessions simply do not survive a restart.
        secret_key=settings.session_secret or secrets.token_urlsafe(32),
        session_cookie="abo_session",
        same_site="lax",
        # Behind HTTPS the session cookie is never sent over plain HTTP.
        https_only=settings.public_base_url.lower().startswith("https://"),
        max_age=12 * 3600,
    )
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
    app.include_router(public.router)
    app.include_router(panel.router)

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, _exc: LoginRequired) -> Response:
        if request.headers.get("hx-request"):
            # HTMX swaps fragments; tell it to do a full-page redirect instead.
            return Response(status_code=401, headers={"HX-Redirect": "/panel/login"})
        return RedirectResponse("/panel/login", status_code=303)

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    @app.get("/health", include_in_schema=False)
    async def health(request: Request) -> JSONResponse:
        """Liveness + dependency check, used by Docker and uptime monitors."""
        context: AppContext = request.app.state.ctx
        checks: dict[str, str] = {}
        try:
            async with context.sessions() as session:
                await session.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {type(exc).__name__}"
        if context.redis is not None:
            try:
                await context.redis.ping()
                checks["redis"] = "ok"
            except Exception as exc:
                checks["redis"] = f"error: {type(exc).__name__}"
        healthy = all(value == "ok" for value in checks.values())
        return JSONResponse(
            {
                "status": "ok" if healthy else "degraded",
                "version": __version__,
                "ai_provider": context.provider.name,
                "channels": context.channels.names(),
                "checks": checks,
            },
            status_code=200 if healthy else 503,
        )

    return app


def app_factory() -> FastAPI:  # pragma: no cover - uvicorn entry point
    return create_app()
