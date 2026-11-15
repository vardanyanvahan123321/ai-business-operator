"""Public endpoints: the customer request form and the Telegram webhook."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import select
from starlette.responses import Response

from ai_operator.actions.invoice import ensure_invoice_pdf
from ai_operator.channels.telegram import TelegramChannel
from ai_operator.channels.telegram_bot import handle_update
from ai_operator.channels.webform import WebFormChannel, form_to_inbound, new_conversation_token
from ai_operator.core.i18n import t
from ai_operator.db.models import Invoice, Lead, Message, Outbound
from ai_operator.queue.ingest import ingest
from ai_operator.web.deps import current_lang, get_ctx, render

router = APIRouter()
MAX_FORM_TEXT = 4000


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/request", status_code=307)


@router.get("/request")
async def request_form(request: Request) -> Response:
    return render(request, "public/form.html", {"token": None, "error": None})


@router.post("/request")
async def submit_request(
    request: Request,
    text: Annotated[str, Form()] = "",
    name: Annotated[str, Form()] = "",
    contact: Annotated[str, Form()] = "",
    token: Annotated[str, Form()] = "",
    website: Annotated[str, Form()] = "",
) -> Response:
    ctx = get_ctx(request)
    lang = current_lang(request, ctx.business.default_language)
    if website:
        # Honeypot: humans never see this field, simple bots fill every field.
        return RedirectResponse("/request", status_code=303)
    cleaned = text.strip()
    if len(cleaned) < 2:
        return render(
            request,
            "public/form.html",
            {"token": token or None, "error": t(lang, "webform.invalid")},
            status_code=422,
        )
    conversation = token if _valid_token(token) else new_conversation_token()
    await ingest(
        ctx,
        form_to_inbound(
            token=conversation,
            name=name[:120],
            contact=contact[:120],
            text=cleaned[:MAX_FORM_TEXT],
        ),
    )
    return RedirectResponse(f"/request/{conversation}", status_code=303)


def _valid_token(token: str) -> bool:
    return 16 <= len(token) <= 64 and token.replace("-", "").replace("_", "").isalnum()


async def _thread(request: Request, token: str) -> list[dict[str, object]]:
    ctx = get_ctx(request)
    if not _valid_token(token):
        raise HTTPException(status_code=404)
    async with ctx.sessions() as session:
        messages = (
            await session.scalars(
                select(Message)
                .where(Message.channel == WebFormChannel.name, Message.sender_id == token)
                .order_by(Message.id)
            )
        ).all()
        if not messages:
            raise HTTPException(status_code=404)
        thread: list[dict[str, object]] = []
        for message in messages:
            replies = (
                await session.scalars(
                    select(Outbound).where(Outbound.message_id == message.id).order_by(Outbound.id)
                )
            ).all()
            thread.append(
                {
                    "message": message,
                    "replies": replies,
                    # Spam gets no reply at all; do not keep the page spinning for it.
                    "waiting": not replies and message.status in {"received", "processing"},
                }
            )
        return thread


@router.get("/request/{token}")
async def conversation(request: Request, token: str) -> Response:
    thread = await _thread(request, token)
    return render(request, "public/thread.html", {"token": token, "thread": thread, "error": None})


@router.get("/request/{token}/thread")
async def conversation_fragment(request: Request, token: str) -> Response:
    """HTMX polls this fragment until the assistant's reply appears."""
    thread = await _thread(request, token)
    return render(request, "public/_thread.html", {"token": token, "thread": thread})


@router.get("/request/{token}/invoice/{invoice_id}")
async def conversation_invoice(request: Request, token: str, invoice_id: int) -> Response:
    """The customer can download only an invoice that belongs to their own conversation."""
    ctx = get_ctx(request)
    async with ctx.sessions() as session:
        invoice = await session.get(Invoice, invoice_id)
        lead = await session.get(Lead, invoice.lead_id) if invoice else None
        if invoice is None or lead is None or lead.channel != "web" or lead.sender_id != token:
            raise HTTPException(status_code=404)
        path = await ensure_invoice_pdf(
            invoice, lead, ctx.business, ctx.settings.invoices_dir, demo=ctx.settings.demo_mode
        )
    return FileResponse(path, media_type="application/pdf", filename=f"{invoice.number}.pdf")


@router.post("/webhooks/telegram/{secret}", include_in_schema=False)
async def telegram_webhook(request: Request, secret: str) -> dict[str, bool]:
    """Alternative to long polling. The secret path segment authenticates Telegram."""
    ctx = get_ctx(request)
    expected = ctx.settings.telegram_webhook_secret
    if not expected or not secrets.compare_digest(secret, expected):
        raise HTTPException(status_code=404)
    channel = ctx.channels.get("telegram") if "telegram" in ctx.channels else None
    if not isinstance(channel, TelegramChannel):
        raise HTTPException(status_code=503, detail="telegram channel is not configured")
    await handle_update(ctx, channel, await request.json())
    return {"ok": True}
