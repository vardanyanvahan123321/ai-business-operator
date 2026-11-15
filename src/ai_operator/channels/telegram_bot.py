"""Telegram bot front end: commands, language choice, and hand-over to the pipeline."""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy.dialects.postgresql import insert

from ai_operator.channels.base import ChannelError
from ai_operator.channels.telegram import TelegramChannel, parse_update
from ai_operator.config import LANGS
from ai_operator.context import AppContext
from ai_operator.core.i18n import normalize_lang, t
from ai_operator.core.logging import get_logger
from ai_operator.core.message import OutboundMessage
from ai_operator.db.models import Setting
from ai_operator.db.session import session_scope
from ai_operator.pipeline import preference_key
from ai_operator.queue.ingest import ingest

log = get_logger(__name__)
OFFSET_KEY = "telegram_offset"

_LANGUAGE_KEYBOARD = {
    "inline_keyboard": [
        [
            {"text": "English", "callback_data": "lang:en"},
            {"text": "Русский", "callback_data": "lang:ru"},
            {"text": "Հայերեն", "callback_data": "lang:hy"},
        ]
    ]
}


async def _set_setting(ctx: AppContext, key: str, value: dict[str, Any]) -> None:
    async with session_scope(ctx.sessions) as session:
        await session.execute(
            insert(Setting)
            .values(key=key, value=value)
            .on_conflict_do_update(index_elements=[Setting.key], set_={"value": value})
        )


async def _get_setting(ctx: AppContext, key: str) -> dict[str, Any] | None:
    async with ctx.sessions() as session:
        row = await session.get(Setting, key)
        return dict(row.value) if row else None


async def _language_for(ctx: AppContext, chat_id: str, hint: str | None) -> str:
    stored = await _get_setting(ctx, preference_key(TelegramChannel.name, chat_id))
    if stored and stored.get("lang"):
        return normalize_lang(stored["lang"])
    return normalize_lang(hint)


async def handle_update(ctx: AppContext, channel: TelegramChannel, update: dict[str, Any]) -> None:
    """Process one update: a command, a language button, or a customer message."""
    callback = update.get("callback_query")
    if callback:
        data = str(callback.get("data") or "")
        chat_id = str(((callback.get("message") or {}).get("chat") or {}).get("id") or "")
        lang = data.removeprefix("lang:")
        if data.startswith("lang:") and lang in LANGS and chat_id:
            await _set_setting(ctx, preference_key(TelegramChannel.name, chat_id), {"lang": lang})
            await channel.call("answerCallbackQuery", callback_query_id=callback["id"])
            await channel.send(
                OutboundMessage(
                    channel=channel.name, recipient=chat_id, text=t(lang, "bot.language.set")
                )
            )
        return

    inbound = parse_update(update)
    if inbound is None:
        return
    command = inbound.text.strip().split()[0].split("@")[0].lower() if inbound.text.strip() else ""
    if command in {"/start", "/help", "/language"}:
        lang = await _language_for(ctx, inbound.sender_id, inbound.meta.get("language_code"))
        if command == "/language":
            reply = OutboundMessage(
                channel=channel.name,
                recipient=inbound.sender_id,
                text=t(lang, "bot.language.prompt"),
                meta={"reply_markup": _LANGUAGE_KEYBOARD},
            )
        else:
            key = "bot.start" if command == "/start" else "bot.help"
            reply = OutboundMessage(
                channel=channel.name,
                recipient=inbound.sender_id,
                text=t(lang, key, company=ctx.business.name.get(lang)),
            )
        await channel.send(reply)
        return
    await ingest(ctx, inbound)


async def poll_once(ctx: AppContext, channel: TelegramChannel, wait_seconds: int = 25) -> int:
    """Fetch and handle one batch of updates. Returns how many were handled."""
    stored = await _get_setting(ctx, OFFSET_KEY)
    offset = int(stored["offset"]) if stored else None
    updates = await channel.get_updates(offset, wait_seconds=wait_seconds)
    for update in updates:
        try:
            await handle_update(ctx, channel, update)
        except ChannelError as exc:
            # A failed command reply must not block the queue of updates behind it.
            log.warning("telegram.update_failed", error=str(exc))
        # Stored after each update: a crash re-delivers at most one, and ingest dedupes it.
        await _set_setting(ctx, OFFSET_KEY, {"offset": int(update["update_id"]) + 1})
    return len(updates)


async def run_polling(ctx: AppContext, channel: TelegramChannel) -> None:
    """Long-polling loop with backoff; runs until cancelled."""
    delay = 1.0
    log.info("telegram.polling_started")
    while True:
        try:
            await poll_once(ctx, channel)
            delay = 1.0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("telegram.poll_error", error=repr(exc), retry_in=delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
