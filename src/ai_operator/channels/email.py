"""Email channel: SMTP out, IMAP (production) or Mailpit API (demo) in."""

from __future__ import annotations

import asyncio
import email
import email.policy
import hashlib
import imaplib
import re
from datetime import UTC, datetime
from email.message import EmailMessage
from email.utils import make_msgid, parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol

import aiosmtplib
import httpx

from ai_operator.channels.base import Channel, ChannelError
from ai_operator.config import Settings
from ai_operator.core.message import InboundMessage, OutboundMessage

_TAGS = re.compile(r"<(style|script)\b.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_QUOTE_HEADER = re.compile(
    r"^(On .{5,120} wrote:|.{0,80}(написал|пишет|писал)\(?а?\)?:|-{2,}\s*Original Message\s*-{2,}|"
    r"От:.*|From:.*)$",
    re.IGNORECASE,
)


def strip_quoted_reply(text: str) -> str:
    """Keep only what the customer just wrote, not the thread quoted below it."""
    kept: list[str] = []
    for line in text.splitlines():
        if _QUOTE_HEADER.match(line.strip()):
            break
        if line.lstrip().startswith(">"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _body_text(message: EmailMessage) -> str:
    part = message.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    content = part.get_content()
    if not isinstance(content, str):
        return ""
    if part.get_content_type() == "text/html":
        content = re.sub(r"\s+\n", "\n", _TAGS.sub(" ", content))
    return content


def parse_email(raw: bytes, own_address: str) -> InboundMessage | None:
    """Parse an RFC 822 message. Returns None for mail we must never answer."""
    message = email.message_from_bytes(raw, policy=email.policy.default)
    assert isinstance(message, EmailMessage)
    sender_name, sender_address = parseaddr(str(message.get("From", "")))
    sender_address = sender_address.lower()
    if not sender_address or sender_address == own_address.lower():
        return None
    # Mail-loop protection: never reply to robots, bounces and mailing lists.
    auto = str(message.get("Auto-Submitted", "no")).lower()
    precedence = str(message.get("Precedence", "")).lower()
    if auto != "no" or precedence in {"bulk", "junk", "list"} or message.get("List-Id"):
        return None
    if sender_address.startswith(("mailer-daemon@", "postmaster@", "noreply@", "no-reply@")):
        return None

    text = strip_quoted_reply(_body_text(message))
    subject = str(message.get("Subject", "")).strip() or None
    if not text and not subject:
        return None
    message_id = str(message.get("Message-ID", "")).strip()
    external_id = message_id or hashlib.sha256(raw).hexdigest()
    received_at = datetime.now(UTC)
    if message.get("Date"):
        try:
            parsed = parsedate_to_datetime(str(message["Date"]))
            received_at = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    return InboundMessage(
        channel=EmailChannel.name,
        external_id=external_id,
        sender_id=sender_address,
        sender_name=sender_name or None,
        subject=subject,
        text=text or subject or "",
        received_at=received_at,
        meta={"message_id": message_id or None},
    )


class EmailChannel(Channel):
    name = "email"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self, message: OutboundMessage) -> EmailMessage:
        mail = EmailMessage()
        mail["From"] = self._settings.email_address
        mail["To"] = message.recipient
        mail["Subject"] = message.subject or ""
        mail["Message-ID"] = make_msgid(domain=self._settings.email_address.split("@")[-1])
        mail["Auto-Submitted"] = "auto-replied"  # so that other robots do not answer us
        in_reply_to = message.meta.get("in_reply_to")
        if in_reply_to:
            mail["In-Reply-To"] = in_reply_to
            mail["References"] = in_reply_to
        mail.set_content(message.text)
        for attachment in message.attachments:
            maintype, _, subtype = attachment.mime_type.partition("/")
            mail.add_attachment(
                Path(attachment.path).read_bytes(),
                maintype=maintype,
                subtype=subtype or "octet-stream",
                filename=attachment.filename,
            )
        return mail

    async def send(self, message: OutboundMessage) -> None:
        settings = self._settings
        if not settings.smtp_host:
            raise ChannelError("email: SMTP_HOST is not configured")
        try:
            await aiosmtplib.send(
                self.build(message),
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_user or None,
                password=settings.smtp_password or None,
                start_tls=settings.smtp_starttls,
                timeout=30,
            )
        except (aiosmtplib.SMTPException, OSError) as exc:
            raise ChannelError(f"email: SMTP error {type(exc).__name__}: {exc}") from exc


class Inbox(Protocol):
    """Where new mail comes from. ``ack`` is called only after a message is stored."""

    async def fetch(self) -> list[tuple[str, InboundMessage | None]]: ...

    async def ack(self, refs: list[str]) -> None: ...


class ImapInbox:
    """Reads unseen mail with BODY.PEEK, so a crash before ``ack`` loses nothing."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _connect(self) -> imaplib.IMAP4_SSL:
        connection = imaplib.IMAP4_SSL(self._settings.imap_host, self._settings.imap_port)
        connection.login(self._settings.imap_user, self._settings.imap_password)
        connection.select("INBOX")
        return connection

    def _fetch(self) -> list[tuple[str, InboundMessage | None]]:
        connection = self._connect()
        try:
            _, data = connection.uid("SEARCH", "UNSEEN")
            uids = (data[0] or b"").split()[:50]
            found: list[tuple[str, InboundMessage | None]] = []
            for uid in uids:
                _, parts = connection.uid("FETCH", uid.decode(), "(BODY.PEEK[])")
                raw = next((p[1] for p in parts if isinstance(p, tuple)), None)
                if isinstance(raw, bytes):
                    found.append((uid.decode(), parse_email(raw, self._settings.email_address)))
            return found
        finally:
            connection.logout()

    def _ack(self, refs: list[str]) -> None:
        connection = self._connect()
        try:
            for uid in refs:
                connection.uid("STORE", uid, "+FLAGS", "(\\Seen)")
        finally:
            connection.logout()

    async def fetch(self) -> list[tuple[str, InboundMessage | None]]:
        return await asyncio.to_thread(self._fetch)

    async def ack(self, refs: list[str]) -> None:
        if refs:
            await asyncio.to_thread(self._ack, refs)


class MailpitInbox:
    """Demo inbox: reads the local Mailpit mail catcher through its REST API."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=15.0)
        self._base = settings.mailpit_api_url.rstrip("/")

    async def fetch(self) -> list[tuple[str, InboundMessage | None]]:
        query = f'to:"{self._settings.email_address}" is:unread'
        response = await self._client.get(
            f"{self._base}/api/v1/search", params={"query": query, "limit": 50}
        )
        response.raise_for_status()
        found: list[tuple[str, InboundMessage | None]] = []
        for summary in response.json().get("messages", []):
            ref = str(summary["ID"])
            raw = await self._client.get(f"{self._base}/api/v1/message/{ref}/raw")
            raw.raise_for_status()
            found.append((ref, parse_email(raw.content, self._settings.email_address)))
        return found

    async def ack(self, refs: list[str]) -> None:
        if refs:
            response = await self._client.put(
                f"{self._base}/api/v1/messages", json={"IDs": refs, "Read": True}
            )
            response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()


def build_inbox(settings: Settings) -> Any:
    if settings.email_inbox == "imap":
        return ImapInbox(settings)
    if settings.email_inbox == "mailpit":
        return MailpitInbox(settings)
    return None
