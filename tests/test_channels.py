"""Channel adapters: Telegram, email (SMTP / IMAP / Mailpit) and the registry."""

from __future__ import annotations

import json
from email.message import EmailMessage
from pathlib import Path
from typing import Any, ClassVar

import aiosmtplib
import httpx
import pytest
from sqlalchemy import select

from ai_operator.channels import email as email_module
from ai_operator.channels.base import ChannelError, ChannelRegistry
from ai_operator.channels.email import (
    EmailChannel,
    ImapInbox,
    MailpitInbox,
    build_inbox,
    parse_email,
    strip_quoted_reply,
)
from ai_operator.channels.simulated import SimulatedChannel
from ai_operator.channels.telegram import TelegramChannel, parse_update
from ai_operator.channels.telegram_bot import handle_update, poll_once
from ai_operator.channels.webform import WebFormChannel, form_to_inbound, new_conversation_token
from ai_operator.config import Settings
from ai_operator.context import AppContext
from ai_operator.core.message import Attachment, OutboundMessage
from ai_operator.db.models import Message, Setting
from ai_operator.queue import tasks
from ai_operator.runtime import build_channels

OWN = "service@demo-auto.example"


def update(
    text: str, *, update_id: int = 10, chat: int = 42, chat_type: str = "private"
) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 1795000000,
            "chat": {"id": chat, "type": chat_type},
            "from": {
                "first_name": "Ani",
                "last_name": "S",
                "language_code": "ru",
                "username": "ani",
            },
            "text": text,
        },
    }


class FakeTelegram:
    """Scripted Bot API server."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.fail_with: httpx.Response | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        assert "/bottest-token/" in request.url.path
        is_json = request.headers.get("content-type", "").startswith("application/json")
        self.calls.append((method, json.loads(request.content) if is_json else request.content))
        if self.fail_with is not None:
            return self.fail_with
        result: Any = self.updates if method == "getUpdates" else {"message_id": 1}
        return httpx.Response(200, json={"ok": True, "result": result})

    def channel(self) -> TelegramChannel:
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return TelegramChannel("test-token", client=client)

    def sent_texts(self) -> list[str]:
        return [payload["text"] for method, payload in self.calls if method == "sendMessage"]


class TestTelegram:
    def test_parse_update_maps_to_unified_message(self) -> None:
        inbound = parse_update(update("Привет"))
        assert inbound is not None
        assert (inbound.channel, inbound.external_id, inbound.sender_id) == (
            "telegram",
            "42:10",
            "42",
        )
        assert inbound.sender_name == "Ani S"
        assert inbound.meta["language_code"] == "ru"

    def test_non_text_and_group_messages_are_ignored(self) -> None:
        assert parse_update({"update_id": 1}) is None
        assert parse_update({"update_id": 1, "message": {"photo": []}}) is None
        assert parse_update(update("hi", chat_type="group")) is None

    async def test_send_splits_long_text_and_uploads_attachments(self, tmp_path: Path) -> None:
        api = FakeTelegram()
        channel = api.channel()
        pdf = tmp_path / "INV-1.pdf"
        pdf.write_bytes(b"%PDF-1.7 demo")
        await channel.send(
            OutboundMessage(
                channel="telegram",
                recipient="42",
                text="x" * 5000,
                attachments=[Attachment(filename="INV-1.pdf", path=str(pdf))],
            )
        )
        assert [method for method, _ in api.calls] == ["sendMessage", "sendMessage", "sendDocument"]
        assert [len(text) for text in api.sent_texts()] == [4096, 904]
        assert b"%PDF-1.7 demo" in api.calls[2][1]
        await channel.close()

    async def test_errors_never_leak_the_token(self) -> None:
        api = FakeTelegram()
        channel = api.channel()
        message = OutboundMessage(channel="telegram", recipient="42", text="hi")

        api.fail_with = httpx.Response(403, json={"ok": False, "description": "bot was blocked"})
        with pytest.raises(ChannelError, match="403 bot was blocked") as blocked:
            await channel.send(message)
        api.fail_with = httpx.Response(502, text="<html>bad gateway</html>")
        with pytest.raises(ChannelError, match="HTTP 502") as gateway:
            await channel.send(message)

        def explode(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to https://api.telegram.org/bottest-token/x")

        broken = TelegramChannel(
            "test-token", client=httpx.AsyncClient(transport=httpx.MockTransport(explode))
        )
        with pytest.raises(ChannelError, match="network error ConnectError") as network:
            await broken.send(message)
        for caught in (blocked, gateway, network):
            assert "test-token" not in str(caught.value)

    async def test_commands_and_language_choice(self, ctx: AppContext) -> None:
        api = FakeTelegram()
        channel = api.channel()

        await handle_update(ctx, channel, update("/start"))
        assert "Демо Автосервис Ереван" in api.sent_texts()[-1]  # user's Telegram locale is ru

        await handle_update(ctx, channel, update("/language"))
        keyboard = api.calls[-1][1]["reply_markup"]["inline_keyboard"][0]
        assert [button["callback_data"] for button in keyboard] == ["lang:en", "lang:ru", "lang:hy"]

        callback = {
            "update_id": 11,
            "callback_query": {"id": "cb1", "data": "lang:hy", "message": {"chat": {"id": 42}}},
        }
        await handle_update(ctx, channel, callback)
        assert api.calls[-2][0] == "answerCallbackQuery"
        assert "հայերեն" in api.sent_texts()[-1]

        await handle_update(ctx, channel, update("/help@demo_bot"))
        assert "Գրեք Ձեր հարցը" in api.sent_texts()[-1]  # the stored choice now wins

        bogus = {"update_id": 12, "callback_query": {"id": "cb2", "data": "lang:xx", "message": {}}}
        before = len(api.calls)
        await handle_update(ctx, channel, bogus)
        await handle_update(ctx, channel, {"update_id": 13})
        assert len(api.calls) == before
        async with ctx.sessions() as session:
            assert (await session.scalars(select(Message))).all() == []  # commands are not ingested

    async def test_language_preference_is_used_for_pipeline_replies(
        self, ctx: AppContext, kb: int
    ) -> None:
        from tests.conftest import FakeChannel, drain

        api = FakeTelegram()
        callback = {
            "update_id": 1,
            "callback_query": {"id": "c", "data": "lang:ru", "message": {"chat": {"id": 42}}},
        }
        await handle_update(ctx, api.channel(), callback)
        await handle_update(ctx, api.channel(), update("What are your opening hours?"))
        await drain(ctx)
        fake = ctx.channels.get("telegram")
        assert isinstance(fake, FakeChannel)
        assert "Источник" in fake.sent[0].text  # replied in the chosen language

    async def test_polling_stores_the_offset_and_survives_bad_updates(
        self, ctx: AppContext
    ) -> None:
        api = FakeTelegram()
        api.updates = [update("Где вы находитесь?", update_id=100), update("/start", update_id=101)]
        channel = api.channel()

        assert await poll_once(ctx, channel, wait_seconds=0) == 2
        async with ctx.sessions() as session:
            offset = await session.get(Setting, "telegram_offset")
            stored = (await session.scalars(select(Message))).all()
        assert offset is not None
        assert offset.value == {"offset": 102}
        assert [m.text for m in stored] == ["Где вы находитесь?"]

        api.updates = []
        await poll_once(ctx, channel, wait_seconds=0)
        assert api.calls[-1][1]["offset"] == 102

        # A reply that fails to send must not block the updates behind it.
        api.updates = [update("/start", update_id=102), update("Вопрос", update_id=103)]
        original = api.handler

        def flaky(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("sendMessage"):
                return httpx.Response(403, json={"ok": False, "description": "blocked"})
            return original(request)

        flaky_channel = TelegramChannel(
            "test-token", client=httpx.AsyncClient(transport=httpx.MockTransport(flaky))
        )
        assert await poll_once(ctx, flaky_channel, wait_seconds=0) == 2
        async with ctx.sessions() as session:
            assert len((await session.scalars(select(Message))).all()) == 2


def raw_email(
    body: str = "How much is an oil change?",
    *,
    sender: str = "Anna <Anna@Example.com>",
    subject: str = "Question",
    headers: dict[str, str] | None = None,
    html: bool = False,
) -> bytes:
    mail = EmailMessage()
    mail["From"] = sender
    mail["To"] = OWN
    mail["Subject"] = subject
    mail["Message-ID"] = "<abc@example.com>"
    mail["Date"] = "Mon, 16 Nov 2026 10:00:00 +0400"
    for key, value in (headers or {}).items():
        mail[key] = value
    if html:
        mail.set_content(body, subtype="html")
    else:
        mail.set_content(body)
    return mail.as_bytes()


class TestEmailParsing:
    def test_plain_message(self) -> None:
        inbound = parse_email(raw_email(), OWN)
        assert inbound is not None
        assert inbound.sender_id == "anna@example.com"
        assert inbound.sender_name == "Anna"
        assert inbound.external_id == "<abc@example.com>"
        assert inbound.subject == "Question"
        assert inbound.received_at.isoformat() == "2026-11-16T10:00:00+04:00"

    def test_html_body_is_reduced_to_text(self) -> None:
        html = (
            "<html><style>p{color:red}</style><body><p>Нужна <b>диагностика</b></p></body></html>"
        )
        inbound = parse_email(raw_email(html, html=True), OWN)
        assert inbound is not None
        assert "диагностика" in inbound.text
        assert "<" not in inbound.text
        assert "color" not in inbound.text

    def test_quoted_thread_is_stripped(self) -> None:
        text = "Yes, tomorrow works.\n\nOn Mon, 16 Nov 2026 at 10:00, Service wrote:\n> When?\n> Thanks"
        assert strip_quoted_reply(text) == "Yes, tomorrow works."
        assert strip_quoted_reply("Да\n> старое\nеще строка") == "Да\nеще строка"

    @pytest.mark.parametrize(
        "headers",
        [
            {"Auto-Submitted": "auto-replied"},
            {"Precedence": "bulk"},
            {"List-Id": "<news.example.com>"},
        ],
    )
    def test_robots_and_mailing_lists_are_never_answered(self, headers: dict[str, str]) -> None:
        assert parse_email(raw_email(headers=headers), OWN) is None

    def test_own_bounces_and_empty_mail_are_skipped(self) -> None:
        assert parse_email(raw_email(sender=OWN), OWN) is None
        assert parse_email(raw_email(sender="MAILER-DAEMON@mx.example"), OWN) is None
        assert parse_email(raw_email(sender=""), OWN) is None
        assert parse_email(raw_email("", subject=""), OWN) is None

    def test_missing_message_id_falls_back_to_a_content_hash(self) -> None:
        raw = b"From: a@b.co\r\nSubject: Hi\r\nDate: not a date\r\n\r\nHello there"
        first, second = parse_email(raw, OWN), parse_email(raw, OWN)
        assert first is not None
        assert second is not None
        assert len(first.external_id) == 64
        assert first.external_id == second.external_id


class TestEmailSending:
    def _settings(self, **overrides: Any) -> Settings:
        return Settings(_env_file=None, smtp_host="mail.test", smtp_port=1025, **overrides)  # type: ignore[call-arg]

    async def test_reply_is_threaded_marked_automatic_and_carries_the_invoice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        sent: dict[str, Any] = {}

        async def fake_send(message: EmailMessage, **kwargs: Any) -> None:
            sent.update(message=message, **kwargs)

        monkeypatch.setattr(aiosmtplib, "send", fake_send)
        pdf = tmp_path / "INV-7.pdf"
        pdf.write_bytes(b"%PDF-1.7")
        await EmailChannel(self._settings(smtp_starttls=False)).send(
            OutboundMessage(
                channel="email",
                recipient="anna@example.com",
                subject="Re: Question",
                text="Здравствуйте!",
                attachments=[Attachment(filename="INV-7.pdf", path=str(pdf))],
                meta={"in_reply_to": "<abc@example.com>"},
            )
        )
        mail = sent["message"]
        assert (sent["hostname"], sent["port"], sent["start_tls"]) == ("mail.test", 1025, False)
        assert mail["To"] == "anna@example.com"
        assert mail["In-Reply-To"] == "<abc@example.com>"
        assert mail["Auto-Submitted"] == "auto-replied"
        [attachment] = list(mail.iter_attachments())
        assert attachment.get_filename() == "INV-7.pdf"
        assert attachment.get_content_type() == "application/pdf"

    async def test_smtp_failures_become_channel_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise aiosmtplib.SMTPConnectError("connection refused")

        monkeypatch.setattr(aiosmtplib, "send", refuse)
        message = OutboundMessage(channel="email", recipient="a@b.co", text="x")
        with pytest.raises(ChannelError, match="SMTP error"):
            await EmailChannel(self._settings()).send(message)
        with pytest.raises(ChannelError, match="SMTP_HOST"):
            await EmailChannel(Settings(_env_file=None)).send(message)  # type: ignore[call-arg]


class FakeImap:
    """Just enough of imaplib.IMAP4_SSL."""

    stored: ClassVar[list[tuple[str, str, str]]] = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host

    def login(self, user: str, password: str) -> None:
        assert (user, password) == ("user", "secret")

    def select(self, mailbox: str) -> None:
        assert mailbox == "INBOX"

    def uid(self, command: str, *args: str) -> tuple[str, list[Any]]:
        if command == "SEARCH":
            return "OK", [b"5 6"]
        if command == "FETCH":
            assert args[1] == "(BODY.PEEK[])"  # must not mark the mail as seen
            body = raw_email() if args[0] == "5" else raw_email(headers={"Precedence": "bulk"})
            return "OK", [(b"5 (BODY[] {123}", body), b")"]
        FakeImap.stored.append((command, args[0], args[2]))
        return "OK", []

    def logout(self) -> None:
        return None


class TestInboxes:
    async def test_imap_peeks_first_and_marks_seen_only_on_ack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeImap.stored = []
        monkeypatch.setattr(email_module.imaplib, "IMAP4_SSL", FakeImap)
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            email_inbox="imap",
            imap_host="imap.test",
            imap_user="user",
            imap_password="secret",
        )
        inbox = build_inbox(settings)
        assert isinstance(inbox, ImapInbox)
        found = await inbox.fetch()
        assert [(ref, message is not None) for ref, message in found] == [("5", True), ("6", False)]
        assert FakeImap.stored == []
        await inbox.ack(["5", "6"])
        await inbox.ack([])
        assert FakeImap.stored == [("STORE", "5", "(\\Seen)"), ("STORE", "6", "(\\Seen)")]

    async def test_mailpit_inbox_and_poll_job_end_to_end(self, ctx: AppContext) -> None:
        requests: list[tuple[str, str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else None
            requests.append((request.method, request.url.path, body))
            if request.url.path == "/api/v1/search":
                assert "is:unread" in request.url.params["query"]
                return httpx.Response(200, json={"messages": [{"ID": "m1"}, {"ID": "m2"}]})
            if request.url.path == "/api/v1/message/m1/raw":
                return httpx.Response(200, content=raw_email("Сколько стоит диагностика?"))
            if request.url.path == "/api/v1/message/m2/raw":
                return httpx.Response(
                    200, content=raw_email(headers={"Auto-Submitted": "auto-generated"})
                )
            return httpx.Response(200, json={"ok": True})

        settings = Settings(_env_file=None, email_inbox="mailpit", mailpit_api_url="http://mailpit")  # type: ignore[call-arg]
        inbox = MailpitInbox(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        ctx.extras["inbox"] = inbox

        assert await tasks.poll_email({"app": ctx}) == 1
        async with ctx.sessions() as session:
            [stored] = (await session.scalars(select(Message))).all()
        assert (stored.channel, stored.sender_id) == ("email", "anna@example.com")
        # Both were acknowledged: the robot mail must not be fetched again either.
        assert requests[-1] == ("PUT", "/api/v1/messages", {"IDs": ["m1", "m2"], "Read": True})

        assert await tasks.poll_email({"app": ctx}) == 0  # same Message-ID: deduplicated
        await inbox.close()
        ctx.extras.clear()
        assert await tasks.poll_email({"app": ctx}) == 0
        assert build_inbox(Settings(_env_file=None)) is None  # type: ignore[call-arg]
        assert isinstance(build_inbox(settings), MailpitInbox)


class TestRegistry:
    async def test_unknown_channel_is_a_clear_error(self) -> None:
        registry = ChannelRegistry()
        registry.register(WebFormChannel())
        assert registry.names() == ["web"]
        assert "web" in registry
        with pytest.raises(ChannelError, match="'whatsapp' is not configured"):
            registry.get("whatsapp")
        await registry.close()

    def test_channels_are_built_from_settings(self) -> None:
        def names(**overrides: Any) -> list[str]:
            return build_channels(Settings(_env_file=None, **overrides)).names()  # type: ignore[call-arg]

        assert names() == ["web"]
        assert names(demo_mode=True) == ["telegram", "web"]
        assert names(telegram_bot_token="t", smtp_host="smtp.test") == ["email", "telegram", "web"]
        demo = build_channels(Settings(_env_file=None, demo_mode=True)).get("telegram")  # type: ignore[call-arg]
        assert isinstance(demo, SimulatedChannel)

    async def test_web_form_helpers(self) -> None:
        token = new_conversation_token()
        first = form_to_inbound(token=token, name=" Anna ", contact=" 091 ", text=" Hi ")
        second = form_to_inbound(token=token, name="", contact="", text="Again")
        assert first.sender_id == second.sender_id == token
        assert first.external_id != second.external_id
        assert (first.sender_name, first.text, first.meta) == ("Anna", "Hi", {"contact": "091"})
        assert second.sender_name is None
        await WebFormChannel().send(OutboundMessage(channel="web", recipient=token, text="x"))
        await SimulatedChannel("telegram").send(
            OutboundMessage(channel="telegram", recipient="1", text="x")
        )
