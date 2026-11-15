"""Channel contract.

A channel adapter has exactly two jobs:

* turn whatever its platform delivers into an ``InboundMessage`` and hand it to
  ``ingest`` (how it *receives* - polling, webhook, HTTP form - is its own business);
* deliver an ``OutboundMessage`` via ``send``.

Adding WhatsApp (or anything else) means writing one class like this and
registering it in ``build_channels``. Nothing in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ai_operator.core.message import OutboundMessage


class ChannelError(Exception):
    """Delivery failed; the outbox will retry."""


class Channel(ABC):
    name: ClassVar[str]

    @abstractmethod
    async def send(self, message: OutboundMessage) -> None:
        """Deliver a message to the customer. Raise ``ChannelError`` on failure."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release network resources, if any."""


class ChannelRegistry:
    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        self._channels[channel.name] = channel

    def get(self, name: str) -> Channel:
        try:
            return self._channels[name]
        except KeyError:
            raise ChannelError(f"channel {name!r} is not configured") from None

    def names(self) -> list[str]:
        return sorted(self._channels)

    def __contains__(self, name: str) -> bool:
        return name in self._channels

    async def close(self) -> None:
        for channel in self._channels.values():
            await channel.close()
