"""Telegram channel: a thin client over the Bot API.

Deliberately not a bot framework. The bot has no logic of its own - every
message goes into the same pipeline as email and the web form - so all this
adapter needs is ``getUpdates`` / ``sendMessage`` / ``sendDocument``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ai_operator.channels.base import Channel, ChannelError
from ai_operator.core.message import InboundMessage, OutboundMessage

API_BASE = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 4096


class TelegramChannel(Channel):
    name = "telegram"

    def __init__(
        self, token: str, *, client: httpx.AsyncClient | None = None, api_base: str = API_BASE
    ) -> None:
        self._base = f"{api_base}/bot{token}"
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(40.0, connect=10.0))

    async def call(
        self, method: str, *, files: dict[str, Any] | None = None, **payload: Any
    ) -> Any:
        """Call a Bot API method. Errors never include the URL (it contains the token)."""
        try:
            if files:
                response = await self._client.post(
                    f"{self._base}/{method}", data=payload, files=files
                )
            else:
                response = await self._client.post(f"{self._base}/{method}", json=payload)
        except httpx.HTTPError as exc:
            raise ChannelError(f"telegram {method}: network error {type(exc).__name__}") from None
        try:
            body = response.json()
        except ValueError:
            raise ChannelError(f"telegram {method}: HTTP {response.status_code}") from None
        if not body.get("ok"):
            description = body.get("description", "unknown error")
            raise ChannelError(f"telegram {method}: {response.status_code} {description}")
        return body.get("result")

    async def send(self, message: OutboundMessage) -> None:
        text = message.text
        for start in range(0, len(text), MAX_MESSAGE_CHARS):
            payload: dict[str, Any] = {
                "chat_id": message.recipient,
                "text": text[start : start + MAX_MESSAGE_CHARS],
            }
            if start == 0 and "reply_markup" in message.meta:
                payload["reply_markup"] = message.meta["reply_markup"]
            await self.call("sendMessage", **payload)
        for attachment in message.attachments:
            content = await asyncio.to_thread(Path(attachment.path).read_bytes)
            await self.call(
                "sendDocument",
                chat_id=message.recipient,
                files={"document": (attachment.filename, content, attachment.mime_type)},
            )

    async def get_updates(self, offset: int | None, wait_seconds: int = 25) -> list[dict[str, Any]]:
        """Long-poll for updates: Telegram holds the request open up to ``wait_seconds``."""
        payload: dict[str, Any] = {
            "timeout": wait_seconds,
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = offset
        result = await self.call("getUpdates", **payload)
        return list(result or [])

    async def close(self) -> None:
        await self._client.aclose()


def parse_update(update: dict[str, Any]) -> InboundMessage | None:
    """Translate a Telegram update into the unified format (text messages only)."""
    message = update.get("message")
    if not message or not isinstance(message.get("text"), str):
        return None
    chat = message.get("chat") or {}
    if chat.get("type") != "private":
        return None  # the assistant talks to customers one-to-one, never in groups
    sender = message.get("from") or {}
    name = " ".join(p for p in (sender.get("first_name"), sender.get("last_name")) if p)
    return InboundMessage(
        channel=TelegramChannel.name,
        external_id=f"{chat['id']}:{message['message_id']}",
        sender_id=str(chat["id"]),
        sender_name=name or None,
        text=message["text"],
        received_at=datetime.fromtimestamp(message.get("date", 0), tz=UTC)
        if message.get("date")
        else datetime.now(UTC),
        meta={
            "username": sender.get("username"),
            "language_code": sender.get("language_code"),
        },
    )
