"""Google Gemini (default provider; works with a free Google AI Studio key)."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from ai_operator.ai.providers.base import LLMProvider
from ai_operator.ai.resilience import ProviderError, ProviderTransientError
from ai_operator.config import EMBEDDING_DIM


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self, *, api_key: str, embedding_model: str, timeout: float = 45.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is not set (or use AI_PROVIDER=mock)")
        self.embedding_model = embedding_model
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
        )

    @staticmethod
    def _translate(exc: Exception) -> ProviderError:
        if isinstance(exc, errors.APIError):
            # 429 = quota (very common on the free tier), 5xx = Google-side trouble.
            if exc.code == 429 or exc.code >= 500:
                return ProviderTransientError(f"gemini {exc.code}: {exc.message}")
            return ProviderError(f"gemini {exc.code}: {exc.message}")
        # The SDK may surface raw transport errors from its HTTP library.
        if isinstance(exc, TimeoutError | OSError) or type(exc).__module__.split(".")[0] in {
            "httpx",
            "httpx2",
            "httpcore",
            "aiohttp",
        }:
            return ProviderTransientError(f"gemini network error: {exc!r}")
        return ProviderError(f"gemini error: {exc!r}")

    async def _generate(self, system: str, user: str, schema: type[BaseModel] | None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=4096,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=user, config=config
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        text = response.text
        if not text:
            # Blocked by a safety filter or cut off: no point in retrying the same prompt.
            raise ProviderError("gemini returned an empty response")
        return text

    async def _embed(self, texts: list[str], query: bool) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIM,
        )
        contents: Any = list(texts)  # the SDK's union type does not include list[str]
        try:
            response = await self._client.aio.models.embed_content(
                model=self.embedding_model, contents=contents, config=config
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        return [list(e.values or []) for e in response.embeddings or []]
