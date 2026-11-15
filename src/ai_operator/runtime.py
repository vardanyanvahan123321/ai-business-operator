"""Wiring: build the application context from settings."""

from __future__ import annotations

from arq.connections import RedisSettings, create_pool

from ai_operator.actions.sheets import build_lead_sink
from ai_operator.ai.providers import build_provider
from ai_operator.ai.resilience import RedisCache
from ai_operator.channels.base import ChannelRegistry
from ai_operator.channels.email import EmailChannel, build_inbox
from ai_operator.channels.telegram import TelegramChannel
from ai_operator.channels.webform import WebFormChannel
from ai_operator.config import Settings, get_settings, load_business_config
from ai_operator.context import AppContext
from ai_operator.core.logging import configure_logging
from ai_operator.db.session import make_engine, make_session_factory
from ai_operator.queue.jobs import ArqQueue


def build_channels(settings: Settings) -> ChannelRegistry:
    """Register every configured channel. A new channel is one more line here."""
    registry = ChannelRegistry()
    registry.register(WebFormChannel())
    if settings.telegram_enabled:
        registry.register(TelegramChannel(settings.telegram_bot_token))
    if settings.smtp_host:
        registry.register(EmailChannel(settings))
    return registry


async def build_context(settings: Settings | None = None) -> AppContext:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    business = load_business_config(settings.business_config)
    engine = make_engine(settings.database_url)
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    ctx = AppContext(
        settings=settings,
        business=business,
        engine=engine,
        sessions=make_session_factory(engine),
        provider=build_provider(settings, business, RedisCache(redis)),
        channels=build_channels(settings),
        queue=ArqQueue(redis),
        lead_sink=build_lead_sink(settings),
        redis=redis,
    )
    inbox = build_inbox(settings)
    if inbox is not None:
        ctx.extras["inbox"] = inbox
    return ctx
