"""A stand-in channel for demo mode.

The demo ships with sample Telegram conversations but no bot token. This
channel accepts sends and only logs them, so replying to such a conversation
from the panel works instead of failing with "channel is not configured".
"""

from __future__ import annotations

from ai_operator.channels.base import Channel
from ai_operator.core.logging import get_logger
from ai_operator.core.message import OutboundMessage

log = get_logger(__name__)


class SimulatedChannel(Channel):
    def __init__(self, name: str) -> None:
        self.name = name  # type: ignore[misc]

    async def send(self, message: OutboundMessage) -> None:
        log.info("channel.simulated_send", channel=self.name, chars=len(message.text))
