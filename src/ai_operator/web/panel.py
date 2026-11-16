"""Admin panel (password-protected): inbox, leads, handoffs, knowledge base, prices, logs."""

from __future__ import annotations

import secrets
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import selectinload
from starlette.responses import Response

from ai_operator import ops
from ai_operator.actions.audit import audit
from ai_operator.actions.invoice import ensure_invoice_pdf
from ai_operator.ai.rag import KnowledgeBase
from ai_operator.config import LANGS
from ai_operator.core.i18n import t
from ai_operator.core.message import Category, LeadStatus, MessageStatus
from ai_operator.db.models import (
    DeadLetter,
    Event,
    Handoff,
    Invoice,
    KBChunk,
    KBDocument,
    Lead,
    Message,
    Notification,
    Outbound,
    PriceReport,
    Report,
    Setting,
)
from ai_operator.db.session import session_scope
from ai_operator.queue import jobs
from ai_operator.web import charts, queries
from ai_operator.web.deps import (
    LANG_COOKIE,
    client_address,
    current_lang,
    get_ctx,
    render,
    require_admin,
    verify_csrf,
)

router = APIRouter(prefix="/panel")
admin = [Depends(require_admin)]
admin_post = [Depends(require_admin), Depends(verify_csrf)]
PAGE_SIZE = 30
MAX_UPLOAD_BYTES = 512 * 1024


def _back(request: Request, fallback: str) -> RedirectResponse:
    return RedirectResponse(fallback, status_code=303)


def _today(request: Request) -> Any:
    return datetime.now(get_ctx(request).business.tz).date()


# ------------------------------------------------------------------- auth


@router.get("/login")
async def login_form(request: Request) -> Response:
    if request.session.get("admin"):
        return RedirectResponse("/panel", status_code=303)
    return render(request, "login.html", {"error": None})


@router.post("/login", dependencies=[Depends(verify_csrf)])
async def login(request: Request, password: Annotated[str, Form()] = "") -> Response:
    ctx = get_ctx(request)
    lang = current_lang(request, ctx.business.default_language)
    throttle = request.app.state.login_throttle
    client = client_address(request)
    if throttle.blocked(client):
        return render(request, "login.html", {"error": t(lang, "ui.login.throttled")}, 429)
    expected = ctx.settings.admin_password
    if not expected:
        # Secure by default: no password configured means nobody can log in.
        return render(request, "login.html", {"error": t(lang, "ui.login.disabled")}, 503)
    if not secrets.compare_digest(password.encode(), expected.encode()):
        throttle.record(client)
        return render(request, "login.html", {"error": t(lang, "ui.login.wrong")}, 401)
    throttle.reset(client)
    request.session.clear()  # new session id material after login
    request.session["admin"] = True
    return RedirectResponse("/panel", status_code=303)


@router.post("/logout", dependencies=admin_post)
async def logout(request: Request) -> Response:
    request.session.clear()
    return RedirectResponse("/panel/login", status_code=303)


@router.get("/lang/{code}")
async def set_language(request: Request, code: str) -> Response:
    target = request.headers.get("referer") or "/panel"
    response = RedirectResponse(target, status_code=303)
    if code in LANGS:
        response.set_cookie(LANG_COOKIE, code, max_age=365 * 24 * 3600, samesite="lax")
    return response


# -------------------------------------------------------------- dashboard


@router.get("", dependencies=admin)
async def dashboard(request: Request) -> Response:
    ctx = get_ctx(request)
    lang = current_lang(request, ctx.business.default_language)
    today = _today(request)
    async with ctx.sessions() as session:
        kpis = await queries.kpis(session, ctx.business, today)
        days, outcomes = await queries.messages_per_day(session, ctx.business, today)
        category_rows = await queries.categories(session, ctx.business, today)
        channel_rows = await queries.channels(session, ctx.business, today)
        revenue_days, revenue = await queries.revenue_per_day(session, ctx.business, today)
        recent = (await session.scalars(select(Message).order_by(Message.id.desc()).limit(8))).all()
        report = await session.scalar(
            select(Report).where(Report.kind == "morning").order_by(Report.id.desc()).limit(1)
        )
    labels = [f"{d:%d.%m}" for d in days]
    outcome_series = [
        charts.Series(t(lang, "status.answered"), outcomes[MessageStatus.ANSWERED.value], 1),
        charts.Series(t(lang, "status.handoff"), outcomes[MessageStatus.HANDOFF.value], 2),
        charts.Series(t(lang, "status.ignored"), outcomes[MessageStatus.IGNORED.value], 3),
    ]
    return render(
        request,
        "dashboard.html",
        {
            "kpis": kpis,
            "outcome_chart": charts.stacked_bars(
                labels, outcome_series, t(lang, "ui.chart.outcomes")
            ),
            "outcome_series": outcome_series,
            "outcome_labels": labels,
            "category_chart": charts.horizontal_bars(
                [(t(lang, f"category.{name}"), count) for name, count in category_rows],
                t(lang, "ui.chart.categories"),
            ),
            "channel_chart": charts.horizontal_bars(
                [(name, count) for name, count in channel_rows], t(lang, "ui.chart.channels")
            ),
            "revenue_chart": charts.stacked_bars(
                [f"{d:%d.%m}" for d in revenue_days],
                [charts.Series(t(lang, "ui.chart.revenue"), revenue, 1)],
                t(lang, "ui.chart.revenue"),
            ),
            "has_data": kpis["messages_week"] > 0,
            "recent": recent,
            "report": report,
        },
    )


# ------------------------------------------------------------------ inbox


async def _inbox_page(request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    params = request.query_params
    status, category, channel = params.get("status"), params.get("category"), params.get("channel")
    search = (params.get("q") or "").strip()
    page = max(1, int(params.get("page", "1")) if params.get("page", "1").isdigit() else 1)

    conditions = []
    if status in MessageStatus._value2member_map_:
        conditions.append(Message.status == status)
    if category in Category._value2member_map_:
        conditions.append(Message.category == category)
    if channel:
        conditions.append(Message.channel == channel)
    if search:
        like = f"%{search}%"
        conditions.append(or_(Message.text.ilike(like), Message.sender_name.ilike(like)))

    async with ctx.sessions() as session:
        total = await session.scalar(select(func.count()).select_from(Message).where(*conditions))
        rows = (
            await session.scalars(
                select(Message)
                .where(*conditions)
                .order_by(Message.id.desc())
                .limit(PAGE_SIZE)
                .offset((page - 1) * PAGE_SIZE)
            )
        ).all()
    return {
        "messages": rows,
        "total": int(total or 0),
        "page": page,
        "pages": max(1, -(-int(total or 0) // PAGE_SIZE)),
        "filters": {"status": status, "category": category, "channel": channel, "q": search},
        "statuses": [s.value for s in MessageStatus],
        "categories": [c.value for c in Category],
        "channels": ctx.channels.names(),
        "query": request.url.query,
    }


@router.get("/inbox", dependencies=admin)
async def inbox(request: Request) -> Response:
    context = await _inbox_page(request)
    template = "_inbox_rows.html" if request.headers.get("hx-request") else "inbox.html"
    return render(request, template, context)


@router.get("/inbox/{message_id}", dependencies=admin)
async def message_detail(request: Request, message_id: int) -> Response:
    ctx = get_ctx(request)
    async with ctx.sessions() as session:
        message = await session.scalar(
            select(Message)
            .where(Message.id == message_id)
            .options(selectinload(Message.outbound), selectinload(Message.handoff))
        )
        if message is None:
            raise HTTPException(status_code=404)
        events = (
            await session.scalars(
                select(Event).where(Event.message_id == message_id).order_by(Event.id)
            )
        ).all()
        lead = await session.scalar(
            select(Lead)
            .where(Lead.channel == message.channel, Lead.sender_id == message.sender_id)
            .order_by(Lead.id.desc())
            .limit(1)
        )
    return render(request, "message.html", {"message": message, "events": events, "lead": lead})


# --------------------------------------------------------------- handoffs


@router.get("/handoffs", dependencies=admin)
async def handoffs(request: Request) -> Response:
    ctx = get_ctx(request)
    show = request.query_params.get("show", "open")
    async with ctx.sessions() as session:
        query = select(Handoff).options(selectinload(Handoff.message)).order_by(Handoff.id.desc())
        if show != "all":
            query = query.where(Handoff.status == "open")
        rows = (await session.scalars(query.limit(100))).all()
    return render(request, "handoffs.html", {"handoffs": rows, "show": show})


@router.post("/handoffs/{handoff_id}/resolve", dependencies=admin_post)
async def resolve_handoff(
    request: Request, handoff_id: int, reply: Annotated[str, Form()] = ""
) -> Response:
    if not await ops.resolve_handoff(get_ctx(request), handoff_id, reply):
        raise HTTPException(status_code=404)
    return _back(request, "/panel/handoffs")


# ------------------------------------------------------------------ leads


@router.get("/leads", dependencies=admin)
async def leads(request: Request) -> Response:
    ctx = get_ctx(request)
    status = request.query_params.get("status")
    async with ctx.sessions() as session:
        query = select(Lead).options(selectinload(Lead.invoice)).order_by(Lead.id.desc())
        if status in LeadStatus._value2member_map_:
            query = query.where(Lead.status == status)
        rows = (await session.scalars(query.limit(200))).all()
    return render(
        request,
        "leads.html",
        {
            "leads": rows,
            "status": status,
            "statuses": [s.value for s in LeadStatus],
            "sink": ctx.lead_sink.name,
        },
    )


@router.post("/leads/{lead_id}/status", dependencies=admin_post)
async def set_lead_status(
    request: Request, lead_id: int, status: Annotated[str, Form()]
) -> Response:
    ctx = get_ctx(request)
    if status not in LeadStatus._value2member_map_ or status == LeadStatus.INCOMPLETE.value:
        raise HTTPException(status_code=422)
    async with session_scope(ctx.sessions) as session:
        lead = await session.get(Lead, lead_id)
        if lead is None:
            raise HTTPException(status_code=404)
        lead.status = status
        lead.sheet_synced_at = None  # the sheet copy is now stale
        audit(session, "lead.status_changed", lead_id=lead_id, status=status)
    await jobs.safe_enqueue(ctx.queue, jobs.SYNC_LEAD, lead_id)
    return _back(request, "/panel/leads")


@router.get("/leads/{lead_id}/invoice", dependencies=admin)
async def lead_invoice(request: Request, lead_id: int) -> Response:
    ctx = get_ctx(request)
    async with ctx.sessions() as session:
        lead = await session.get(Lead, lead_id)
        invoice = await session.scalar(select(Invoice).where(Invoice.lead_id == lead_id))
        if lead is None or invoice is None:
            raise HTTPException(status_code=404)
        path = await ensure_invoice_pdf(
            invoice, lead, ctx.business, ctx.settings.invoices_dir, demo=ctx.settings.demo_mode
        )
    return FileResponse(path, media_type="application/pdf", filename=f"{invoice.number}.pdf")


# --------------------------------------------------------- knowledge base


@router.get("/kb", dependencies=admin)
async def kb_list(request: Request) -> Response:
    ctx = get_ctx(request)
    async with ctx.sessions() as session:
        chunk_count = (
            select(KBChunk.document_id, func.count().label("chunks"))
            .group_by(KBChunk.document_id)
            .subquery()
        )
        rows = (
            await session.execute(
                select(KBDocument, func.coalesce(chunk_count.c.chunks, 0))
                .outerjoin(chunk_count, chunk_count.c.document_id == KBDocument.id)
                .order_by(KBDocument.language, KBDocument.title)
            )
        ).all()
        stale = await KnowledgeBase(session, ctx.provider).stale_chunks()
    return render(
        request,
        "kb.html",
        {"documents": rows, "stale": stale, "embedding_model": ctx.provider.embedding_model},
    )


@router.get("/kb/search", dependencies=admin)
async def kb_search(request: Request) -> Response:
    """Retrieval playground: see exactly which chunks the AI would be given."""
    ctx = get_ctx(request)
    query = (request.query_params.get("q") or "").strip()
    results = []
    if query:
        from ai_operator.core.language import detect_language

        async with ctx.sessions() as session:
            results = await KnowledgeBase(session, ctx.provider).search(
                query, language=detect_language(query), top_k=ctx.settings.rag_top_k
            )
    return render(request, "_kb_search.html", {"results": results, "query": query})


@router.get("/kb/new", dependencies=admin)
async def kb_new(request: Request) -> Response:
    return render(request, "kb_edit.html", {"document": None, "error": None})


@router.get("/kb/{document_id}", dependencies=admin)
async def kb_edit(request: Request, document_id: int) -> Response:
    ctx = get_ctx(request)
    async with ctx.sessions() as session:
        document = await session.get(KBDocument, document_id)
    if document is None:
        raise HTTPException(status_code=404)
    return render(request, "kb_edit.html", {"document": document, "error": None})


@router.post("/kb", dependencies=admin_post)
async def kb_save(
    request: Request,
    title: Annotated[str, Form()] = "",
    language: Annotated[str, Form()] = "en",
    content: Annotated[str, Form()] = "",
    document_id: Annotated[str, Form()] = "",
    upload: UploadFile | None = None,
) -> Response:
    ctx = get_ctx(request)
    lang = current_lang(request, ctx.business.default_language)
    if upload is not None and upload.filename:
        raw = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return render(
                request,
                "kb_edit.html",
                {"document": None, "error": t(lang, "ui.kb.too_large")},
                413,
            )
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return render(
                request, "kb_edit.html", {"document": None, "error": t(lang, "ui.kb.not_text")}, 422
            )
        title = title.strip() or upload.filename.rsplit(".", 1)[0]
    title, content = title.strip(), content.strip()
    if not title or not content or language not in LANGS:
        return render(
            request, "kb_edit.html", {"document": None, "error": t(lang, "ui.kb.required")}, 422
        )
    async with session_scope(ctx.sessions) as session:
        kb = KnowledgeBase(session, ctx.provider)
        existing = (
            await session.get(KBDocument, int(document_id)) if document_id.isdigit() else None
        )
        source = existing.source if existing else f"panel/{secrets.token_hex(6)}"
        document = await kb.upsert_document(
            source=source, title=title[:255], content=content, language=language
        )
        audit(session, "kb.saved", document_id=document.id, title=title[:80])
    return _back(request, "/panel/kb")


@router.post("/kb/{document_id}/delete", dependencies=admin_post)
async def kb_delete(request: Request, document_id: int) -> Response:
    ctx = get_ctx(request)
    async with session_scope(ctx.sessions) as session:
        await KnowledgeBase(session, ctx.provider).delete_document(document_id)
        audit(session, "kb.deleted", document_id=document_id)
    return _back(request, "/panel/kb")


@router.post("/kb/reindex", dependencies=admin_post)
async def kb_reindex(request: Request) -> Response:
    ctx = get_ctx(request)
    async with session_scope(ctx.sessions) as session:
        chunks = await KnowledgeBase(session, ctx.provider).reindex_all()
        audit(session, "kb.reindexed", chunks=chunks, model=ctx.provider.embedding_model)
    return _back(request, "/panel/kb")


# ----------------------------------------------------------------- prices


@router.get("/prices", dependencies=admin)
async def prices(request: Request) -> Response:
    ctx = get_ctx(request)
    lang = current_lang(request, ctx.business.default_language)
    today = _today(request)
    codes = [s.code for s in ctx.business.services]
    selected = request.query_params.get("service")
    selected = selected if selected in codes else (codes[0] if codes else "")
    async with ctx.sessions() as session:
        competitors, table = await queries.price_comparison(session, ctx.business)
        days, history = await queries.price_history(session, ctx.business, selected, today)
        reports = (
            await session.scalars(select(PriceReport).order_by(PriceReport.id.desc()).limit(5))
        ).all()
    service = ctx.business.service(selected)
    labels = [f"{d:%d.%m}" for d in days]
    series = [
        charts.Series(name, values, slot + 2) for slot, (name, values) in enumerate(history.items())
    ]
    if service is not None and service.price is not None and series:
        series.insert(
            0, charts.Series(t(lang, "ui.prices.ours"), [float(service.price)] * len(days), 1)
        )
    return render(
        request,
        "prices.html",
        {
            "competitors": competitors,
            "table": table,
            "reports": reports,
            "selected": selected,
            "history_chart": charts.line_chart(labels, series, t(lang, "ui.prices.history"))
            if series
            else None,
            "history_series": series,
            "history_labels": labels,
        },
    )


@router.post("/prices/run", dependencies=admin_post)
async def prices_run(request: Request) -> Response:
    await jobs.safe_enqueue(get_ctx(request).queue, jobs.SCRAPE_PRICES)
    return _back(request, "/panel/prices")


# ---------------------------------------------------------------- reports


@router.get("/reports", dependencies=admin)
async def reports(request: Request) -> Response:
    ctx = get_ctx(request)
    async with ctx.sessions() as session:
        rows = (
            await session.scalars(select(Report).order_by(Report.report_date.desc()).limit(30))
        ).all()
        notifications = (
            await session.scalars(select(Notification).order_by(Notification.id.desc()).limit(30))
        ).all()
    return render(request, "reports.html", {"reports": rows, "notifications": notifications})


@router.post("/reports/run", dependencies=admin_post)
async def reports_run(request: Request) -> Response:
    await jobs.safe_enqueue(get_ctx(request).queue, jobs.MORNING_REPORT)
    return _back(request, "/panel/reports")


# ------------------------------------------------------------------- logs


@router.get("/logs", dependencies=admin)
async def logs(request: Request) -> Response:
    ctx = get_ctx(request)
    level = request.query_params.get("level")
    async with ctx.sessions() as session:
        query = select(Event).order_by(Event.id.desc())
        if level in {"info", "warning", "error"}:
            query = query.where(Event.level == level)
        events = (await session.scalars(query.limit(150))).all()
        dead = (
            await session.scalars(
                select(DeadLetter)
                .where(DeadLetter.resolved_at.is_(None))
                .order_by(DeadLetter.id.desc())
            )
        ).all()
        pending = await session.scalar(
            select(func.count()).select_from(Outbound).where(Outbound.status == "pending")
        )
    return render(
        request,
        "logs.html",
        {"events": events, "dead_letters": dead, "level": level, "pending_outbound": pending or 0},
    )


@router.post("/logs/dead/{dead_letter_id}/retry", dependencies=admin_post)
async def retry_dead(request: Request, dead_letter_id: int) -> Response:
    if not await ops.retry_dead_letter(get_ctx(request), dead_letter_id):
        raise HTTPException(status_code=404)
    return _back(request, "/panel/logs")


# --------------------------------------------------------------- settings


@router.get("/settings", dependencies=admin)
async def settings_page(request: Request) -> Response:
    ctx = get_ctx(request)
    async with ctx.sessions() as session:
        thresholds = await session.get(Setting, "thresholds")
    threshold = ctx.settings.confidence_threshold
    if thresholds is not None:
        threshold = float(thresholds.value.get("confidence_threshold", threshold))
    return render(
        request,
        "settings.html",
        {
            "threshold": threshold,
            "provider": ctx.provider,
            "channels": ctx.channels.names(),
            "settings": ctx.settings,
            "sink": ctx.lead_sink.name,
            "saved": request.query_params.get("saved") == "1",
        },
    )


@router.post("/settings", dependencies=admin_post)
async def settings_save(
    request: Request, confidence_threshold: Annotated[float, Form()]
) -> Response:
    ctx = get_ctx(request)
    value = {"confidence_threshold": min(1.0, max(0.0, confidence_threshold))}
    async with session_scope(ctx.sessions) as session:
        await session.execute(
            insert(Setting)
            .values(key="thresholds", value=value)
            .on_conflict_do_update(index_elements=[Setting.key], set_={"value": value})
        )
        audit(session, "settings.changed", threshold=value["confidence_threshold"])
    return _back(request, "/panel/settings?saved=1")
