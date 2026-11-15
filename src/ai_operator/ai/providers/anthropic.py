"""Anthropic Claude.

Claude has no embeddings endpoint, so retrieval uses the local hashing
embedder inherited from ``LLMProvider``.
"""

from __future__ import annotations

from typing import Any

import anthropic
from pydantic import BaseModel

from ai_operator.ai.providers.base import LLMProvider
from ai_operator.ai.resilience import ProviderError, ProviderTransientError

# Models whose safety classifiers may decline a request and that support the
# server-side fallback: the API then re-runs the request on a fallback model.
_FALLBACK_MODELS = frozenset(
    {
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
        "claude-fable-5-2",
        "claude-fable-5-1",
    }
)
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    # Lexical local embeddings score lower than neural ones.
    relevance_floor = 0.12

    def __init__(self, *, api_key: str, timeout: float = 45.0, **kwargs: Any) -> None:
        kwargs.setdefault("relevance_floor", 0.12)
        super().__init__(**kwargs)
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set (or use AI_PROVIDER=mock)")
        # Retries are handled by LLMProvider so that they respect our rate limiter.
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)

    async def _generate(self, system: str, user: str, schema: type[BaseModel] | None) -> str:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8000,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # Classification and extraction are routine work: low effort is enough.
            "output_config": {"effort": "low"},
        }
        if self.model in _FALLBACK_MODELS:
            params["betas"] = [_FALLBACK_BETA]
            params["fallbacks"] = "default"
        try:
            if schema is not None:
                response = await self._client.beta.messages.parse(output_format=schema, **params)
            else:
                response = await self._client.beta.messages.create(**params)
        except anthropic.RateLimitError as exc:
            raise ProviderTransientError(f"anthropic rate limit: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderTransientError(f"anthropic {exc.status_code}") from exc
            raise ProviderError(f"anthropic {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise ProviderTransientError(f"anthropic network error: {exc!r}") from exc

        if response.stop_reason == "refusal":
            raise ProviderError("anthropic declined the request (refusal)")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise ProviderError(f"anthropic returned no text (stop_reason={response.stop_reason})")
        return text

    async def close(self) -> None:
        await self._client.close()
