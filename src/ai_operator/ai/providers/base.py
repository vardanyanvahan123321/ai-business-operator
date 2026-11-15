"""Provider contract and the shared machinery for real LLM vendors.

``AIProvider`` is what the pipeline talks to: four business-level operations,
no vendor concepts. ``LLMProvider`` implements those operations once, on top of
two small vendor hooks (``_generate`` and ``_embed``), and adds everything that
makes a paid/limited API safe to use in production: pacing, retries with
backoff, a response cache and validation of every reply against a schema.
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ai_operator.ai import prompts
from ai_operator.ai.embeddings import LOCAL_EMBEDDING_MODEL, hash_embed
from ai_operator.ai.resilience import (
    Cache,
    MemoryCache,
    ProviderError,
    ProviderTransientError,
    RateLimiter,
    cache_key,
)
from ai_operator.ai.schemas import (
    Analysis,
    AnalyzeRequest,
    GroundedAnswer,
    PriceChange,
    RetrievedChunk,
)
from ai_operator.config import EMBEDDING_DIM
from ai_operator.core.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class AIProvider(ABC):
    """The four things the business pipeline needs from "an AI"."""

    name: str
    model: str
    embedding_model: str
    # Retrieval scores below this are treated as "nothing relevant found".
    relevance_floor: float

    @abstractmethod
    async def analyze(self, request: AnalyzeRequest) -> Analysis:
        """Classify a message and extract booking data from it."""

    @abstractmethod
    async def answer(
        self, question: str, language: str, chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        """Answer strictly from the given knowledge-base chunks, or say "not found"."""

    @abstractmethod
    async def summarize_prices(self, changes: list[PriceChange], language: str) -> str:
        """Write a short owner-facing summary of competitor price changes."""

    @abstractmethod
    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Turn texts into ``EMBEDDING_DIM``-sized unit vectors."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release network resources, if any."""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply (tolerates code fences and chatter)."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return cleaned[start : end + 1]


def normalize_vector(vector: list[float]) -> list[float]:
    if len(vector) != EMBEDDING_DIM:
        raise ProviderError(f"embedding has {len(vector)} dimensions, expected {EMBEDDING_DIM}")
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class LLMProvider(AIProvider):
    """Base for Gemini / Anthropic / OpenAI."""

    relevance_floor = 0.30

    def __init__(
        self,
        *,
        model: str,
        business_name: str,
        timezone: str,
        cache: Cache | None = None,
        requests_per_minute: int = 10,
        max_retries: int = 4,
        cache_ttl: int = 24 * 3600,
        relevance_floor: float = 0.30,
        retry_initial_wait: float = 2.0,
    ) -> None:
        self.model = model
        self.relevance_floor = relevance_floor
        self._business_name = business_name
        self._timezone = timezone
        self._cache: Cache = cache or MemoryCache()
        self._limiter = RateLimiter(requests_per_minute)
        self._max_retries = max(1, max_retries)
        self._cache_ttl = cache_ttl
        self._retry_initial_wait = retry_initial_wait

    # --- vendor hooks ----------------------------------------------------

    @abstractmethod
    async def _generate(self, system: str, user: str, schema: type[BaseModel] | None) -> str:
        """One raw model call. Must raise ProviderTransientError for retryable failures."""

    async def _embed(self, texts: list[str], query: bool) -> list[list[float]]:
        """Vendors without an embeddings API fall back to local hashing."""
        return [hash_embed(t) for t in texts]

    embedding_model = LOCAL_EMBEDDING_MODEL

    # --- shared machinery ------------------------------------------------

    async def _with_retries(self, system: str, user: str, schema: type[BaseModel] | None) -> str:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(ProviderTransientError),
            wait=wait_exponential_jitter(initial=self._retry_initial_wait, max=60),
            stop=stop_after_attempt(self._max_retries),
            reraise=True,
        ):
            with attempt:
                await self._limiter.wait()
                return await self._generate(system, user, schema)
        raise ProviderError("unreachable")  # pragma: no cover

    async def _call(self, system: str, user: str, schema: type[BaseModel] | None) -> str:
        key = cache_key(self.name, self.model, system, user, schema.__name__ if schema else "")
        cached = await self._cache.get(key)
        if cached is not None:
            log.debug("ai.cache_hit", provider=self.name)
            return cached
        text = await self._with_retries(system, user, schema)
        await self._cache.set(key, text, self._cache_ttl)
        return text

    async def _structured(self, system: str, user: str, schema: type[T]) -> T:
        """Call the model and validate the reply; give it one chance to repair bad JSON."""
        raw = await self._call(system, user, schema)
        try:
            return schema.model_validate_json(extract_json(raw))
        except (ValidationError, ValueError) as first_error:
            log.warning("ai.invalid_output", provider=self.name, error=str(first_error)[:300])
            repair = (
                f"{user}\n\nYour previous reply was rejected by the validator:\n"
                f"{str(first_error)[:500]}\nReply again with one valid JSON object matching this "
                f"schema:\n{json.dumps(schema.model_json_schema())}"
            )
            raw = await self._call(system, repair, schema)
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as second_error:
                raise ProviderError(f"model returned invalid {schema.__name__}") from second_error

    # --- AIProvider ------------------------------------------------------

    async def analyze(self, request: AnalyzeRequest) -> Analysis:
        return await self._structured(
            prompts.analyze_system(self._business_name, self._timezone),
            prompts.analyze_user(request),
            Analysis,
        )

    async def answer(
        self, question: str, language: str, chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        return await self._structured(
            prompts.answer_system(self._business_name, language),
            prompts.answer_user(question, chunks),
            GroundedAnswer,
        )

    async def summarize_prices(self, changes: list[PriceChange], language: str) -> str:
        text = await self._call(
            prompts.price_summary_system(language), prompts.price_summary_user(changes), None
        )
        return text.strip()

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        if query and len(texts) == 1:
            key = cache_key("embed", self.embedding_model, texts[0])
            cached = await self._cache.get(key)
            if cached is not None:
                return [json.loads(cached)]
            vectors = await self._embed_with_retries(texts, query)
            await self._cache.set(key, json.dumps(vectors[0]), self._cache_ttl)
            return vectors
        return await self._embed_with_retries(texts, query)

    async def _embed_with_retries(self, texts: list[str], query: bool) -> list[list[float]]:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(ProviderTransientError),
            wait=wait_exponential_jitter(initial=self._retry_initial_wait, max=60),
            stop=stop_after_attempt(self._max_retries),
            reraise=True,
        ):
            with attempt:
                await self._limiter.wait()
                return [normalize_vector(v) for v in await self._embed(texts, query)]
        raise ProviderError("unreachable")  # pragma: no cover
