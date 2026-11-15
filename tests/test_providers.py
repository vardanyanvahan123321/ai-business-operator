"""AI providers: shared resilience machinery and the three vendor adapters (faked clients)."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

from ai_operator.ai import prompts
from ai_operator.ai.providers import build_provider
from ai_operator.ai.providers.anthropic import AnthropicProvider
from ai_operator.ai.providers.base import LLMProvider, extract_json, normalize_vector
from ai_operator.ai.providers.gemini import GeminiProvider
from ai_operator.ai.providers.mock import MockProvider
from ai_operator.ai.providers.openai import OpenAIProvider
from ai_operator.ai.resilience import (
    MemoryCache,
    ProviderError,
    ProviderTransientError,
    RateLimiter,
    RedisCache,
    cache_key,
)
from ai_operator.ai.schemas import (
    Analysis,
    AnalyzeRequest,
    Extraction,
    GroundedAnswer,
    HistoryTurn,
    PriceChange,
    RetrievedChunk,
    ServiceOption,
)
from ai_operator.config import EMBEDDING_DIM, BusinessConfig, Settings

ANALYSIS_JSON = json.dumps(
    {
        "category": "order",
        "confidence": 0.91,
        "language": "en",
        "summary": "Wants an oil change",
        "extraction": {"service_code": "oil_change", "scheduled_at": "2026-11-17T10:00"},
        "injection_suspected": False,
    }
)
REQUEST = AnalyzeRequest(
    text="Book me for an oil change tomorrow at 10",
    now=datetime(2026, 11, 16, 9, 0),
    services=(ServiceOption("oil_change", ("Oil change",), ("oil",)),),
    subject="Booking",
    history=(HistoryTurn("customer", "Hello"), HistoryTurn("business", "Hi!")),
    pending=Extraction(name="Anna"),
)
CHUNKS = [RetrievedChunk(7, 1, "Prices", "en", "An oil change costs 12,000 AMD.", 0.8)]
COMMON: dict[str, Any] = {
    "business_name": "Demo",
    "timezone": "Asia/Yerevan",
    "requests_per_minute": 0,
    "retry_initial_wait": 0.0,
}


class ScriptedProvider(LLMProvider):
    """An LLM whose replies (or failures) are scripted by the test."""

    name = "scripted"

    def __init__(self, script: list[str | Exception], **kwargs: Any) -> None:
        super().__init__(model="test-model", **COMMON, **kwargs)
        self.script = list(script)
        self.calls: list[tuple[str, str]] = []

    async def _generate(self, system: str, user: str, schema: type[BaseModel] | None) -> str:
        self.calls.append((system, user))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class TestSharedMachinery:
    async def test_valid_reply_is_parsed_into_the_schema(self) -> None:
        provider = ScriptedProvider([ANALYSIS_JSON])
        analysis = await provider.analyze(REQUEST)
        assert analysis.category.value == "order"
        assert analysis.extraction.service_code == "oil_change"
        system, user = provider.calls[0]
        assert "untrusted data" in system
        assert "<customer_message>" in user
        assert "Monday" in user  # the model is told what "tomorrow" is relative to
        assert '"name":"Anna"' in user  # already collected booking details

    async def test_transient_errors_are_retried(self) -> None:
        provider = ScriptedProvider(
            [ProviderTransientError("429"), ProviderTransientError("503"), ANALYSIS_JSON]
        )
        assert (await provider.analyze(REQUEST)).confidence == 0.91
        assert len(provider.calls) == 3

    async def test_gives_up_after_max_retries(self) -> None:
        provider = ScriptedProvider([ProviderTransientError("429")] * 3, max_retries=3)
        with pytest.raises(ProviderTransientError):
            await provider.analyze(REQUEST)
        assert len(provider.calls) == 3

    async def test_fatal_errors_are_not_retried(self) -> None:
        provider = ScriptedProvider([ProviderError("bad key"), ANALYSIS_JSON])
        with pytest.raises(ProviderError):
            await provider.analyze(REQUEST)
        assert len(provider.calls) == 1

    async def test_identical_prompt_is_served_from_cache(self) -> None:
        provider = ScriptedProvider([ANALYSIS_JSON])
        await provider.analyze(REQUEST)
        await provider.analyze(REQUEST)  # would fail: the script is empty
        assert len(provider.calls) == 1

    async def test_invalid_json_gets_one_repair_attempt(self) -> None:
        provider = ScriptedProvider(['{"category": "banana"}', f"```json\n{ANALYSIS_JSON}\n```"])
        analysis = await provider.analyze(REQUEST)
        assert analysis.category.value == "order"
        assert "rejected by the validator" in provider.calls[1][1]

    async def test_second_invalid_reply_is_a_provider_error(self) -> None:
        provider = ScriptedProvider(["not json at all", '{"found": "maybe?"}'])
        with pytest.raises(ProviderError, match="invalid GroundedAnswer"):
            await provider.answer("How much?", "en", CHUNKS)

    async def test_answer_prompt_contains_only_supplied_sources(self) -> None:
        provider = ScriptedProvider(['{"found": true, "answer": "12,000 AMD", "source_ids": [7]}'])
        answer = await provider.answer("How much is an oil change?", "ru", CHUNKS)
        assert answer == GroundedAnswer(found=True, answer="12,000 AMD", source_ids=[7])
        system, user = provider.calls[0]
        assert "Russian" in system
        assert "[source id=7 title='Prices']" in user

    async def test_price_summary_is_plain_text(self) -> None:
        provider = ScriptedProvider(["  Prices went down.  "])
        change = PriceChange("A", "Oil", "oil_change", Decimal(10), Decimal(9), Decimal(12), "AMD")
        assert await provider.summarize_prices([change], "en") == "Prices went down."
        assert "10 -> 9 AMD, our price 12" in provider.calls[0][1]

    async def test_local_embeddings_fallback_and_query_cache(self) -> None:
        provider = ScriptedProvider([])
        [vector] = await provider.embed(["oil change"], query=True)
        assert len(vector) == EMBEDDING_DIM
        assert await provider.embed(["oil change"], query=True) == [vector]
        assert await provider.embed([]) == []
        assert len(await provider.embed(["a", "b"])) == 2

    def test_helpers(self) -> None:
        assert extract_json('Sure! {"a": 1} hope it helps') == '{"a": 1}'
        with pytest.raises(ValueError, match="no JSON"):
            extract_json("nothing here")
        with pytest.raises(ProviderError):
            normalize_vector([1.0, 2.0])
        unit = normalize_vector([3.0] + [0.0] * (EMBEDDING_DIM - 1))
        assert unit[0] == 1.0
        assert cache_key("a", 1) == cache_key("a", 1) != cache_key("a", 2)
        assert "Subject (untrusted data): Booking" in prompts.analyze_user(REQUEST)


class TestResilience:
    async def test_rate_limiter_spaces_calls(self) -> None:
        import time

        limiter = RateLimiter(per_minute=1200)  # one call per 50 ms
        started = time.monotonic()
        for _ in range(3):
            await limiter.wait()
        assert time.monotonic() - started >= 0.09
        await RateLimiter(0).wait()  # zero disables pacing

    async def test_memory_cache_expires(self) -> None:
        cache = MemoryCache()
        await cache.set("k", "v", ttl=60)
        assert await cache.get("k") == "v"
        await cache.set("old", "v", ttl=-1)
        assert await cache.get("old") is None
        assert await cache.get("missing") is None

    async def test_redis_cache_roundtrip(self) -> None:
        from arq.connections import RedisSettings, create_pool

        from tests.conftest import TEST_REDIS_URL

        redis = await create_pool(RedisSettings.from_dsn(TEST_REDIS_URL))
        cache = RedisCache(redis, prefix="test-cache:")
        await cache.set("greeting", "բարև", ttl=30)
        assert await cache.get("greeting") == "բարև"
        assert await cache.get("absent") is None
        await redis.delete("test-cache:greeting")
        await redis.aclose()


class TestMock:
    async def test_price_summary_in_three_languages(self) -> None:
        provider = MockProvider()
        cheaper = PriceChange(
            "AutoFix", "Oil", "oil_change", Decimal(12500), Decimal(11500), Decimal(12000), "AMD"
        )
        dearer = PriceChange("Garage", "Tyres", None, Decimal(10000), Decimal(11000), None, "AMD")
        english = await provider.summarize_prices([cheaper, dearer], "en")
        assert "2 item(s): 1 went up, 1 went down" in english
        assert "cheaper than us" in english
        assert "дешевле нас" in await provider.summarize_prices([cheaper], "ru")
        assert "էժան" in await provider.summarize_prices([cheaper], "hy")
        assert "competitive" in await provider.summarize_prices([dearer], "en")
        first_seen = PriceChange("A", "Oil", None, None, Decimal(1), None, "AMD")
        assert "No competitor price changes" in await provider.summarize_prices([first_seen], "xx")

    async def test_answer_refuses_without_relevant_chunks(self) -> None:
        provider = MockProvider()
        assert not (await provider.answer("anything", "en", [])).found
        assert not (await provider.answer("Do you sell scooters?", "en", CHUNKS)).found
        found = await provider.answer("How much does an oil change cost?", "en", CHUNKS)
        assert found.found
        assert found.source_ids == [7]


def _fake_response(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(text=text)


class TestGemini:
    def _provider(self) -> GeminiProvider:
        return GeminiProvider(
            api_key="k", model="gemini-test", embedding_model="emb-test", **COMMON
        )

    async def test_generate_and_embed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider()
        seen: dict[str, Any] = {}

        async def generate_content(**kwargs: Any) -> SimpleNamespace:
            seen.update(kwargs)
            return _fake_response(ANALYSIS_JSON)

        async def embed_content(**kwargs: Any) -> SimpleNamespace:
            seen["embed"] = kwargs
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[2.0] + [0.0] * (EMBEDDING_DIM - 1))]
            )

        monkeypatch.setattr(provider._client.aio.models, "generate_content", generate_content)
        monkeypatch.setattr(provider._client.aio.models, "embed_content", embed_content)

        assert (await provider.analyze(REQUEST)).category.value == "order"
        assert seen["model"] == "gemini-test"
        assert seen["config"].response_mime_type == "application/json"
        assert seen["config"].temperature == 0.0
        [vector] = await provider.embed(["doc"])
        assert vector[0] == 1.0  # normalised
        assert seen["embed"]["config"].task_type == "RETRIEVAL_DOCUMENT"
        assert seen["embed"]["config"].output_dimensionality == EMBEDDING_DIM

    async def test_error_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider()
        quota = genai_errors.APIError(429, {"error": {"message": "quota", "status": "X"}})
        bad_key = genai_errors.APIError(400, {"error": {"message": "bad key", "status": "X"}})
        assert isinstance(provider._translate(quota), ProviderTransientError)
        assert not isinstance(provider._translate(bad_key), ProviderTransientError)
        assert isinstance(provider._translate(httpx.ReadTimeout("slow")), ProviderTransientError)
        assert not isinstance(provider._translate(KeyError("x")), ProviderTransientError)

        async def empty(**_kwargs: Any) -> SimpleNamespace:
            return _fake_response(None)

        monkeypatch.setattr(provider._client.aio.models, "generate_content", empty)
        with pytest.raises(ProviderError, match="empty response"):
            await provider._generate("s", "u", None)

        async def failing(**_kwargs: Any) -> SimpleNamespace:
            raise quota

        monkeypatch.setattr(provider._client.aio.models, "generate_content", failing)
        monkeypatch.setattr(provider._client.aio.models, "embed_content", failing)
        with pytest.raises(ProviderTransientError):
            await provider._generate("s", "u", Analysis)
        with pytest.raises(ProviderTransientError):
            await provider._embed(["x"], query=True)

    def test_missing_key_is_reported_clearly(self) -> None:
        with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
            GeminiProvider(api_key="", model="m", embedding_model="e", **COMMON)


def _status_error(cls: type, status: int) -> Exception:
    request = httpx.Request("POST", "https://api.example/v1")
    response = httpx.Response(status, request=request, json={"error": {"message": "x"}})
    try:
        return cls("x", response=response, body=None)  # type: ignore[no-any-return]
    except TypeError:
        # Newer SDKs are built on httpx2; build the same objects from that library.
        import httpx2

        request2 = httpx2.Request("POST", "https://api.example/v1")
        response2 = httpx2.Response(status, request=request2, json={"error": {"message": "x"}})
        return cls("x", response=response2, body=None)  # type: ignore[no-any-return]


class TestAnthropic:
    def _provider(self, model: str = "claude-opus-5-5") -> AnthropicProvider:
        return AnthropicProvider(api_key="k", model=model, **COMMON)

    def _message(self, text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
        return SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        )

    async def test_structured_call_uses_fallbacks_on_supported_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = self._provider()
        seen: dict[str, Any] = {}

        async def parse(**kwargs: Any) -> SimpleNamespace:
            seen.update(kwargs)
            return self._message(ANALYSIS_JSON)

        monkeypatch.setattr(provider._client.beta.messages, "parse", parse)
        assert (await provider.analyze(REQUEST)).category.value == "order"
        assert seen["output_format"] is Analysis
        assert seen["fallbacks"] == "default"
        assert seen["betas"] == ["server-side-fallback-2026-07-01"]
        assert seen["output_config"] == {"effort": "low"}
        assert "temperature" not in seen
        assert provider.embedding_model == "local-hash-v1"

    async def test_plain_call_without_fallbacks_on_other_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = self._provider("claude-haiku-5-5")
        seen: dict[str, Any] = {}

        async def create(**kwargs: Any) -> SimpleNamespace:
            seen.update(kwargs)
            return self._message("Summary text")

        monkeypatch.setattr(provider._client.beta.messages, "create", create)
        assert await provider._generate("s", "u", None) == "Summary text"
        assert "fallbacks" not in seen

    async def test_refusal_and_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider()

        async def refuse(**_kwargs: Any) -> SimpleNamespace:
            return self._message("", stop_reason="refusal")

        monkeypatch.setattr(provider._client.beta.messages, "create", refuse)
        with pytest.raises(ProviderError, match="refusal"):
            await provider._generate("s", "u", None)

        async def no_text(**_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(stop_reason="max_tokens", content=[])

        monkeypatch.setattr(provider._client.beta.messages, "create", no_text)
        with pytest.raises(ProviderError, match="no text"):
            await provider._generate("s", "u", None)

        for error, transient in (
            (_status_error(anthropic.RateLimitError, 429), True),
            (_status_error(anthropic.InternalServerError, 500), True),
            (_status_error(anthropic.BadRequestError, 400), False),
            (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), True),
        ):

            async def fail(error: Exception = error, **_kwargs: Any) -> SimpleNamespace:
                raise error

            monkeypatch.setattr(provider._client.beta.messages, "create", fail)
            with pytest.raises(ProviderError) as caught:
                await provider._generate("s", "u", None)
            assert isinstance(caught.value, ProviderTransientError) is transient
        await provider.close()

    def test_missing_key(self) -> None:
        with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider(api_key="", model="m", **COMMON)


class TestOpenAI:
    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key="k", model="gpt-test", embedding_model="emb", **COMMON)

    async def test_generate_and_embed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider()
        seen: dict[str, Any] = {}

        def completion(text: str | None) -> SimpleNamespace:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

        async def parse(**kwargs: Any) -> SimpleNamespace:
            seen["parse"] = kwargs
            return completion(ANALYSIS_JSON)

        async def create(**kwargs: Any) -> SimpleNamespace:
            return completion("plain")

        async def embeddings(**kwargs: Any) -> SimpleNamespace:
            seen["embed"] = kwargs
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.0] * (EMBEDDING_DIM - 1) + [5.0])]
            )

        monkeypatch.setattr(provider._client.chat.completions, "parse", parse)
        monkeypatch.setattr(provider._client.chat.completions, "create", create)
        monkeypatch.setattr(provider._client.embeddings, "create", embeddings)

        assert (await provider.analyze(REQUEST)).language == "en"
        assert seen["parse"]["response_format"] is Analysis
        assert seen["parse"]["messages"][0]["role"] == "system"
        assert await provider._generate("s", "u", None) == "plain"
        [vector] = await provider.embed(["doc"])
        assert vector[-1] == 1.0
        assert seen["embed"]["dimensions"] == EMBEDDING_DIM

        async def empty(**_kwargs: Any) -> SimpleNamespace:
            return completion(None)

        monkeypatch.setattr(provider._client.chat.completions, "create", empty)
        with pytest.raises(ProviderError, match="empty"):
            await provider._generate("s", "u", None)

        async def limited(**_kwargs: Any) -> SimpleNamespace:
            raise _status_error(openai.RateLimitError, 429)

        async def rejected(**_kwargs: Any) -> SimpleNamespace:
            raise _status_error(openai.BadRequestError, 400)

        monkeypatch.setattr(provider._client.chat.completions, "create", limited)
        with pytest.raises(ProviderTransientError):
            await provider._generate("s", "u", None)
        monkeypatch.setattr(provider._client.embeddings, "create", rejected)
        with pytest.raises(ProviderError) as caught:
            await provider._embed(["x"], query=False)
        assert not isinstance(caught.value, ProviderTransientError)
        await provider.close()

    def test_missing_key(self) -> None:
        with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
            OpenAIProvider(api_key="", model="m", embedding_model="e", **COMMON)


def test_factory_builds_the_provider_selected_in_settings(business: BusinessConfig) -> None:
    def build(name: str) -> Any:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            ai_provider=name,
            gemini_api_key="g",
            anthropic_api_key="a",
            openai_api_key="o",
        )
        return build_provider(settings, business)

    assert isinstance(build("mock"), MockProvider)
    assert isinstance(build("gemini"), GeminiProvider)
    assert isinstance(build("anthropic"), AnthropicProvider)
    assert isinstance(build("openai"), OpenAIProvider)
    assert build("anthropic").relevance_floor == 0.12
    assert build("gemini").relevance_floor == 0.30
