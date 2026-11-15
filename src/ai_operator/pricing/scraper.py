"""A polite scraper for competitor price pages.

"Polite" is enforced in code, not left to configuration discipline:

* ``robots.txt`` is fetched and obeyed for our user agent; if it cannot be
  read because of a server error, we assume "disallowed";
* requests to one host are spaced by the larger of our own minimum delay and
  the site's ``Crawl-delay``;
* the bot identifies itself honestly in the User-Agent.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from ai_operator.config import Competitor, ScrapeConfig


class ScrapeError(Exception):
    """The page could not be fetched or parsed."""


class ScrapeDisallowed(ScrapeError):
    """robots.txt forbids fetching this URL."""


@dataclass(frozen=True, slots=True)
class ScrapedItem:
    name: str
    price: Decimal


_NUMBER = re.compile(r"\d[\d\s.,  ]*")


def parse_price(text: str) -> Decimal | None:
    """``"12 000 AMD"`` -> ``Decimal("12000")``; ``"12,500.50"`` -> ``Decimal("12500.50")``."""
    match = _NUMBER.search(text)
    if not match:
        return None
    raw = re.sub(r"[\s  ]", "", match.group(0)).rstrip(".,")
    # A separator followed by exactly 1-2 digits at the end is a decimal point.
    decimal_part = re.search(r"[.,](\d{1,2})$", raw)
    whole = raw[: decimal_part.start()] if decimal_part else raw
    cleaned = re.sub(r"[.,]", "", whole) + (f".{decimal_part.group(1)}" if decimal_part else "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def parse_prices(html: str, competitor: Competitor) -> list[ScrapedItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[ScrapedItem] = []
    for node in soup.select(competitor.item_selector):
        name_node = node.select_one(competitor.name_selector)
        price_node = node.select_one(competitor.price_selector)
        if name_node is None or price_node is None:
            continue
        price = parse_price(price_node.get_text(" ", strip=True))
        name = name_node.get_text(" ", strip=True)
        if name and price is not None:
            items.append(ScrapedItem(name=name, price=price))
    return items


class PoliteFetcher:
    def __init__(self, config: ScrapeConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            headers={"User-Agent": config.user_agent},
            follow_redirects=True,
        )
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last_request: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _robots_for(self, origin: str) -> RobotFileParser | None:
        """Parsed robots.txt, or None meaning "everything is disallowed"."""
        if origin in self._robots:
            return self._robots[origin]
        parser: RobotFileParser | None = RobotFileParser()
        assert parser is not None
        try:
            response = await self._client.get(f"{origin}/robots.txt")
        except httpx.HTTPError:
            parser = None
        else:
            if response.status_code >= 500:
                parser = None  # the site is in trouble: stay away
            elif response.status_code >= 400:
                parser.parse([])  # no robots.txt: nothing is forbidden
            else:
                parser.parse(response.text.splitlines())
        self._robots[origin] = parser
        return parser

    async def get(self, url: str) -> str:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:  # one request per host at a time
            robots = await self._robots_for(origin)
            agent = self._config.user_agent
            if robots is None or not robots.can_fetch(agent, url):
                raise ScrapeDisallowed(f"robots.txt disallows {url}")
            crawl_delay = robots.crawl_delay(agent)
            delay = max(self._config.min_delay_seconds, float(crawl_delay or 0))
            wait = self._last_request.get(origin, 0.0) + delay - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                response = await self._client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ScrapeError(f"cannot fetch {url}: {type(exc).__name__}") from exc
            finally:
                self._last_request[origin] = time.monotonic()
            return response.text

    async def close(self) -> None:
        await self._client.aclose()
