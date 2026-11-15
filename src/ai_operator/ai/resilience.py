"""Rate limiting and response caching for AI calls.

Free-tier LLM quotas are small (a handful of requests per minute). Instead of
running into 429s we pace ourselves, and we never pay twice for the same
prompt: a retried job reuses the cached response.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Protocol


class ProviderError(Exception):
    """The provider rejected the request; retrying the same call will not help."""


class ProviderTransientError(ProviderError):
    """Rate limit, timeout or server error: worth retrying after a pause."""


class RateLimiter:
    """Spaces calls evenly: at most ``per_minute`` call starts per minute."""

    def __init__(self, per_minute: int) -> None:
        self._interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait(self) -> None:
        if self._interval == 0.0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            self._next_at = max(now, self._next_at) + self._interval
        if delay > 0:
            await asyncio.sleep(delay)


class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl: int) -> None: ...


class MemoryCache:
    """In-process cache for tests and single-process runs."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            del self._data[key]
            return None
        return value

    async def set(self, key: str, value: str, ttl: int) -> None:
        self._data[key] = (time.monotonic() + ttl, value)


class RedisCache:
    def __init__(self, redis: Any, prefix: str = "ai-cache:") -> None:
        self._redis = redis
        self._prefix = prefix

    async def get(self, key: str) -> str | None:
        raw = await self._redis.get(self._prefix + key)
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    async def set(self, key: str, value: str, ttl: int) -> None:
        await self._redis.set(self._prefix + key, value, ex=ttl)


def cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
