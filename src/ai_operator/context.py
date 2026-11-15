"""Application context: the wired-up dependencies every process shares."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from ai_operator.actions.sheets import LeadSink
from ai_operator.ai.providers.base import AIProvider
from ai_operator.channels.base import ChannelRegistry
from ai_operator.config import BusinessConfig, Settings
from ai_operator.db.session import SessionFactory
from ai_operator.queue.jobs import JobQueue


@dataclass
class AppContext:
    settings: Settings
    business: BusinessConfig
    engine: AsyncEngine
    sessions: SessionFactory
    provider: AIProvider
    channels: ChannelRegistry
    queue: JobQueue
    lead_sink: LeadSink
    redis: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    async def close(self) -> None:
        await self.channels.close()
        await self.provider.close()
        if self.redis is not None:
            await self.redis.aclose()
        await self.engine.dispose()
