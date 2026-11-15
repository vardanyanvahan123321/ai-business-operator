"""OpenAI (chat completions with structured outputs + embeddings)."""

from __future__ import annotations

from typing import Any

import openai
from pydantic import BaseModel

from ai_operator.ai.providers.base import LLMProvider
from ai_operator.ai.resilience import ProviderError, ProviderTransientError
from ai_operator.config import EMBEDDING_DIM


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self, *, api_key: str, embedding_model: str, timeout: float = 45.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set (or use AI_PROVIDER=mock)")
        self.embedding_model = embedding_model
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    @staticmethod
    def _translate(exc: openai.OpenAIError) -> ProviderError:
        if isinstance(
            exc, openai.RateLimitError | openai.APIConnectionError | openai.InternalServerError
        ):
            return ProviderTransientError(f"openai transient error: {exc!r}")
        return ProviderError(f"openai error: {exc!r}")

    async def _generate(self, system: str, user: str, schema: type[BaseModel] | None) -> str:
        messages: list[Any] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            if schema is not None:
                parsed = await self._client.chat.completions.parse(
                    model=self.model, messages=messages, response_format=schema
                )
                content = parsed.choices[0].message.content
            else:
                plain = await self._client.chat.completions.create(
                    model=self.model, messages=messages
                )
                content = plain.choices[0].message.content
        except openai.OpenAIError as exc:
            raise self._translate(exc) from exc
        if not content:
            raise ProviderError("openai returned an empty response")
        return content

    async def _embed(self, texts: list[str], query: bool) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                model=self.embedding_model, input=list(texts), dimensions=EMBEDDING_DIM
            )
        except openai.OpenAIError as exc:
            raise self._translate(exc) from exc
        return [list(item.embedding) for item in response.data]

    async def close(self) -> None:
        await self._client.close()
