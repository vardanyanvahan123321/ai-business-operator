"""Web form channel.

The customer has no inbox we could push to: the reply is shown on the form's
status page, which polls for it. "Sending" therefore has nothing to do - the
outbox row itself is the delivery.
"""

from __future__ import annotations

import secrets

from ai_operator.channels.base import Channel
from ai_operator.core.message import InboundMessage, OutboundMessage


class WebFormChannel(Channel):
    name = "web"

    async def send(self, message: OutboundMessage) -> None:
        return None


def new_conversation_token() -> str:
    return secrets.token_urlsafe(18)


def form_to_inbound(*, token: str, name: str, contact: str, text: str) -> InboundMessage:
    """One form conversation = one token; each submission gets its own id."""
    return InboundMessage(
        channel=WebFormChannel.name,
        external_id=f"{token}:{secrets.token_hex(6)}",
        sender_id=token,
        sender_name=name.strip() or None,
        text=text.strip(),
        meta={"contact": contact.strip()},
    )
