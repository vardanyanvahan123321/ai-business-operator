"""Provider factory: the one place that knows which vendors exist."""

from __future__ import annotations

from typing import Any

from ai_operator.ai.providers.base import AIProvider
from ai_operator.ai.resilience import Cache
from ai_operator.config import BusinessConfig, Settings


def build_provider(
    settings: Settings, business: BusinessConfig, cache: Cache | None = None
) -> AIProvider:
    """Create the provider selected by ``AI_PROVIDER``.

    Vendor SDKs are imported lazily so that, for example, a mock-only demo does
    not pay their import time.
    """
    if settings.ai_provider == "mock":
        from ai_operator.ai.providers.mock import MockProvider

        return MockProvider()

    common: dict[str, Any] = {
        "business_name": business.name.en,
        "timezone": business.timezone,
        "cache": cache,
        "requests_per_minute": settings.ai_requests_per_minute,
        "max_retries": settings.ai_max_retries,
        "cache_ttl": settings.ai_cache_ttl_seconds,
        "timeout": settings.ai_timeout_seconds,
    }
    if settings.ai_provider == "gemini":
        from ai_operator.ai.providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            embedding_model=settings.gemini_embedding_model,
            relevance_floor=settings.rag_min_score,
            **common,
        )
    if settings.ai_provider == "anthropic":
        from ai_operator.ai.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key, model=settings.anthropic_model, **common
        )
    from ai_operator.ai.providers.openai import OpenAIProvider

    return OpenAIProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
        relevance_floor=settings.rag_min_score,
        **common,
    )
