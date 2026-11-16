"""Web layer: public form, auth, CSRF, panel pages, operator actions."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import select

from ai_operator.channels.telegram import TelegramChannel
from ai_operator.context import AppContext
from ai_operator.db.models import DeadLetter, Handoff, KBDocument, Lead, Message, Outbound, Setting
from ai_operator.queue.ingest import ingest
from ai_operator.queue.jobs import RecordingQueue
from ai_operator.web.app import create_app
from tests.conftest import FakeChannel, drain, telegram_message

CSRF = re.compile(r'name="csrf_token" value="([^"]+)"')


@pytest.fixture
async def client(ctx: AppContext) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(ctx))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


async def login(client: httpx.AsyncClient) -> str:
    page = await client.get("/panel/login")
    token = CSRF.search(page.text).group(1)  # type: ignore[union-attr]
    response = await client.post(
        "/panel/login", data={"password": "test-password", "csrf_token": token}
    )
    assert response.status_code == 303
    page = await client.get("/panel/settings")
    return CSRF.search(page.text).group(1)  # type: ignore[union-attr]


async def test_health_reports_dependencies(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"
    assert body["ai_provider"] == "mock"


async def test_public_form_end_to_end_with_invoice_download(
    client: httpx.AsyncClient, ctx: AppContext, kb: int
) -> None:
    page = await client.get("/request?lang=ru")
    assert "Оставьте заявку" in page.text

    response = await client.post(
        "/request",
        data={
            "name": "Мария",
            "contact": "+374 77 112233",
            "text": "Запишите меня на шиномонтаж завтра в 11:00",
        },
    )
    assert response.status_code == 303
    thread_url = response.headers["location"]

    waiting = await client.get(thread_url)
    assert "hx-get" in waiting.text  # the page polls until the reply is ready
    await drain(ctx)

    fragment = await client.get(f"{thread_url}/thread")
    assert "Шиномонтаж" in fragment.text
    assert "hx-get" not in fragment.text  # polling stops once answered
    invoice_link = re.search(r'href="(/request/[^"]+/invoice/\d+)"', fragment.text)
    assert invoice_link is not None
    pdf = await client.get(invoice_link.group(1))
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF-")

    # Somebody else's token must not open this invoice.
    other = await client.get("/request/" + "x" * 24 + "/invoice/1")
    assert other.status_code == 404


async def test_public_form_rejects_empty_and_honeypot(
    client: httpx.AsyncClient, ctx: AppContext
) -> None:
    empty = await client.post("/request", data={"text": " "})
    assert empty.status_code == 422
    bot = await client.post("/request", data={"text": "Buy followers", "website": "http://x"})
    assert bot.status_code == 303
    async with ctx.sessions() as session:
        assert (await session.scalars(select(Message))).all() == []


async def test_public_form_limits_messages_per_address(ctx: AppContext) -> None:
    limited = ctx.settings.model_copy(update={"webform_max_messages": 2})
    ctx.settings = limited
    transport = httpx.ASGITransport(app=create_app(ctx))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        for _ in range(2):
            accepted = await http.post("/request", data={"text": "How much is an oil change?"})
            assert accepted.status_code == 303
        flood = await http.post("/request", data={"text": "How much is an oil change?"})
        assert flood.status_code == 429
        # Rejected input does not use up the allowance, and nothing extra was stored.
        async with ctx.sessions() as session:
            assert len((await session.scalars(select(Message))).all()) == 2


async def test_session_cookie_is_secure_only_behind_https(ctx: AppContext) -> None:
    async def session_cookie(base_url: str) -> str:
        ctx.settings = ctx.settings.model_copy(update={"public_base_url": base_url})
        transport = httpx.ASGITransport(app=create_app(ctx))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            return (await http.get("/panel/login")).headers["set-cookie"].lower()

    assert "secure" in await session_cookie("https://operator.example.com")
    assert "secure" not in await session_cookie("http://localhost:8088")


async def test_unknown_conversation_is_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/request/" + "a" * 24)).status_code == 404
    assert (await client.get("/request/short")).status_code == 404


async def test_panel_requires_login(client: httpx.AsyncClient) -> None:
    for path in ("/panel", "/panel/inbox", "/panel/leads", "/panel/kb", "/panel/logs"):
        response = await client.get(path)
        assert response.status_code == 303
        assert response.headers["location"] == "/panel/login"
    htmx = await client.get("/panel/inbox", headers={"HX-Request": "true"})
    assert htmx.status_code == 401
    assert htmx.headers["HX-Redirect"] == "/panel/login"


async def test_login_wrong_password_and_throttling(client: httpx.AsyncClient) -> None:
    page = await client.get("/panel/login")
    token = CSRF.search(page.text).group(1)  # type: ignore[union-attr]
    for _ in range(5):
        response = await client.post("/panel/login", data={"password": "nope", "csrf_token": token})
        assert response.status_code == 401
    blocked = await client.post(
        "/panel/login", data={"password": "test-password", "csrf_token": token}
    )
    assert blocked.status_code == 429


async def test_login_is_disabled_without_configured_password(
    client: httpx.AsyncClient, ctx: AppContext
) -> None:
    ctx.settings.admin_password = ""
    page = await client.get("/panel/login")
    token = CSRF.search(page.text).group(1)  # type: ignore[union-attr]
    response = await client.post("/panel/login", data={"password": "", "csrf_token": token})
    assert response.status_code == 503


async def test_post_without_csrf_token_is_rejected(client: httpx.AsyncClient) -> None:
    await login(client)
    response = await client.post("/panel/settings", data={"confidence_threshold": "0.9"})
    assert response.status_code == 403
    assert (await client.post("/panel/login", data={"password": "x"})).status_code == 403


async def test_all_panel_pages_render_in_three_languages(
    client: httpx.AsyncClient, ctx: AppContext, kb: int
) -> None:
    await ingest(ctx, telegram_message("Сколько стоит замена масла?", msg_id="1"))
    await ingest(ctx, telegram_message("Ужасный сервис, верните деньги!", chat="2", msg_id="2"))
    await ingest(
        ctx,
        telegram_message(
            "Запишите на диагностику завтра в 10:00, тел 091123456", chat="3", msg_id="3"
        ),
    )
    await drain(ctx)
    await login(client)

    pages = (
        "/panel",
        "/panel/inbox",
        "/panel/inbox?status=handoff&category=complaint&q=сервис",
        "/panel/inbox/1",
        "/panel/inbox/2",
        "/panel/inbox/3",
        "/panel/handoffs",
        "/panel/handoffs?show=all",
        "/panel/leads",
        "/panel/leads?status=new",
        "/panel/kb",
        "/panel/kb/new",
        "/panel/kb/1",
        "/panel/kb/search?q=warranty",
        "/panel/prices",
        "/panel/reports",
        "/panel/logs",
        "/panel/logs?level=error",
        "/panel/settings",
    )
    for lang in ("en", "ru", "hy"):
        for path in pages:
            separator = "&" if "?" in path else "?"
            response = await client.get(f"{path}{separator}lang={lang}")
            assert response.status_code == 200, (path, lang)
            assert "ui." not in re.sub(r"<script.*?</script>", "", response.text), (path, lang)

    dashboard = await client.get("/panel?lang=en")
    assert "<svg" in dashboard.text
    assert "Answered by AI" in dashboard.text
    assert (await client.get("/panel/inbox/999")).status_code == 404
    htmx = await client.get("/panel/inbox", headers={"HX-Request": "true"})
    assert "<html" not in htmx.text  # fragment only


async def test_operator_replies_to_handoff_through_the_same_channel(
    client: httpx.AsyncClient, ctx: AppContext, kb: int
) -> None:
    await ingest(ctx, telegram_message("Ужасный сервис, верните деньги!"))
    await drain(ctx)
    token = await login(client)
    async with ctx.sessions() as session:
        handoff = await session.scalar(select(Handoff))
    assert handoff is not None

    response = await client.post(
        f"/panel/handoffs/{handoff.id}/resolve",
        data={"reply": "Приносим извинения, перезвоним вам сегодня.", "csrf_token": token},
    )
    assert response.status_code == 303
    await drain(ctx)

    channel = ctx.channels.get("telegram")
    assert isinstance(channel, FakeChannel)
    assert channel.sent[-1].text == "Приносим извинения, перезвоним вам сегодня."
    async with ctx.sessions() as session:
        resolved = await session.get(Handoff, handoff.id)
        human = await session.scalar(select(Outbound).where(Outbound.kind == "human_reply"))
    assert resolved is not None and resolved.status == "resolved"
    assert human is not None and human.status == "sent"
    again = await client.post(
        f"/panel/handoffs/{handoff.id}/resolve", data={"reply": "", "csrf_token": token}
    )
    assert again.status_code == 404  # already closed


async def test_lead_status_change_and_invoice_download(
    client: httpx.AsyncClient, ctx: AppContext, kb: int
) -> None:
    await ingest(
        ctx, telegram_message("Book me for an oil change tomorrow at 9:00, phone +37491123456")
    )
    await drain(ctx)
    token = await login(client)

    pdf = await client.get("/panel/leads/1/invoice")
    assert pdf.content.startswith(b"%PDF-")
    response = await client.post(
        "/panel/leads/1/status", data={"status": "confirmed", "csrf_token": token}
    )
    assert response.status_code == 303
    await drain(ctx)
    async with ctx.sessions() as session:
        lead = await session.get(Lead, 1)
    assert lead is not None and lead.status == "confirmed" and lead.sheet_synced_at is not None
    assert "confirmed" in ctx.settings.leads_csv.read_text(encoding="utf-8")
    bad = await client.post("/panel/leads/1/status", data={"status": "nope", "csrf_token": token})
    assert bad.status_code == 422


async def test_knowledge_base_crud_and_upload(client: httpx.AsyncClient, ctx: AppContext) -> None:
    token = await login(client)
    created = await client.post(
        "/panel/kb",
        data={
            "title": "Loyalty programme",
            "language": "en",
            "content": "Regular customers get a 10% discount after the fifth visit.",
            "csrf_token": token,
        },
    )
    assert created.status_code == 303
    search = await client.get("/panel/kb/search?q=discount for regular customers")
    assert "Loyalty programme" in search.text

    uploaded = await client.post(
        "/panel/kb",
        data={"language": "ru", "csrf_token": token},
        files={
            "upload": ("parking.md", "# Парковка\n\nПарковка бесплатная.".encode(), "text/markdown")
        },
    )
    assert uploaded.status_code == 303
    binary = await client.post(
        "/panel/kb",
        data={"language": "en", "csrf_token": token},
        files={"upload": ("x.bin", b"\xff\xfe\x00", "application/octet-stream")},
    )
    assert binary.status_code == 422
    missing = await client.post("/panel/kb", data={"title": "", "content": "", "csrf_token": token})
    assert missing.status_code == 422

    async with ctx.sessions() as session:
        documents = (await session.scalars(select(KBDocument).order_by(KBDocument.id))).all()
    assert [d.title for d in documents] == ["Loyalty programme", "parking"]

    edited = await client.post(
        "/panel/kb",
        data={
            "document_id": str(documents[0].id),
            "title": "Loyalty",
            "language": "en",
            "content": "Regular customers get a 15% discount.",
            "csrf_token": token,
        },
    )
    assert edited.status_code == 303
    assert (await client.post("/panel/kb/reindex", data={"csrf_token": token})).status_code == 303
    deleted = await client.post(f"/panel/kb/{documents[1].id}/delete", data={"csrf_token": token})
    assert deleted.status_code == 303
    async with ctx.sessions() as session:
        remaining = (await session.scalars(select(KBDocument))).all()
    assert [(d.title, d.content) for d in remaining] == [
        ("Loyalty", "Regular customers get a 15% discount.")
    ]


async def test_settings_threshold_changes_pipeline_behaviour(
    client: httpx.AsyncClient, ctx: AppContext, kb: int
) -> None:
    token = await login(client)
    saved = await client.post(
        "/panel/settings", data={"confidence_threshold": "0.99", "csrf_token": token}
    )
    assert saved.status_code == 303
    async with ctx.sessions() as session:
        stored = await session.get(Setting, "thresholds")
    assert stored is not None and stored.value == {"confidence_threshold": 0.99}

    # With a near-impossible threshold even a clear question goes to a human.
    message_id, _ = await ingest(ctx, telegram_message("Сколько стоит замена масла?"))
    await drain(ctx)
    async with ctx.sessions() as session:
        handoff = await session.scalar(select(Handoff).where(Handoff.message_id == message_id))
    assert handoff is not None and handoff.reason == "low_confidence"


async def test_dead_letter_can_be_retried_from_panel(
    client: httpx.AsyncClient, ctx: AppContext, kb: int
) -> None:
    channel = ctx.channels.get("telegram")
    assert isinstance(channel, FakeChannel)
    channel.fail_times = 100
    await ingest(ctx, telegram_message("What are your opening hours?"))
    await drain(ctx, max_rounds=30)
    channel.fail_times = 0
    token = await login(client)

    logs = await client.get("/panel/logs?lang=en")
    assert "send_outbound" in logs.text
    async with ctx.sessions() as session:
        dead = await session.scalar(select(DeadLetter).where(DeadLetter.job == "send_outbound"))
    assert dead is not None
    response = await client.post(f"/panel/logs/dead/{dead.id}/retry", data={"csrf_token": token})
    assert response.status_code == 303
    await drain(ctx)

    assert [m.text for m in channel.sent if "09:00" in m.text]  # delivered this time
    again = await client.post(f"/panel/logs/dead/{dead.id}/retry", data={"csrf_token": token})
    assert again.status_code == 404


async def test_manual_triggers_enqueue_jobs(client: httpx.AsyncClient, ctx: AppContext) -> None:
    token = await login(client)
    assert (await client.post("/panel/prices/run", data={"csrf_token": token})).status_code == 303
    assert (await client.post("/panel/reports/run", data={"csrf_token": token})).status_code == 303
    queue = ctx.queue
    assert isinstance(queue, RecordingQueue)
    assert queue.pop_all() == [("scrape_prices", None), ("morning_report", None)]


async def test_logout_and_language_switch(client: httpx.AsyncClient) -> None:
    token = await login(client)
    switched = await client.get("/panel/lang/hy", headers={"referer": "http://test/panel/leads"})
    assert switched.headers["location"] == "http://test/panel/leads"
    assert "Հայտեր" in (await client.get("/panel/leads")).text
    assert (await client.post("/panel/logout", data={"csrf_token": token})).status_code == 303
    assert (await client.get("/panel")).status_code == 303


async def test_telegram_webhook_requires_secret(client: httpx.AsyncClient, ctx: AppContext) -> None:
    update = {
        "update_id": 1,
        "message": {
            "message_id": 5,
            "date": 1780000000,
            "chat": {"id": 42, "type": "private"},
            "from": {"first_name": "Ani"},
            "text": "Что с гарантией?",
        },
    }
    assert (await client.post("/webhooks/telegram/anything", json=update)).status_code == 404
    ctx.channels.register(
        TelegramChannel(
            "token",
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, json={"ok": True, "result": []})
                )
            ),
        )
    )
    ctx.settings.telegram_webhook_secret = "s3cret-path"
    assert (await client.post("/webhooks/telegram/wrong", json=update)).status_code == 404
    ok = await client.post("/webhooks/telegram/s3cret-path", json=update)
    assert ok.status_code == 200
    async with ctx.sessions() as session:
        stored = await session.scalar(select(Message))
    assert stored is not None and stored.sender_name == "Ani"
