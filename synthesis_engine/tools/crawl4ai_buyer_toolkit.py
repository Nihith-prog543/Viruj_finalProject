"""
Buyer-focused Crawl4AI toolkit.

Allows crawling of regulator and pharmaceutical product websites to extract
evidence for finished dosage forms. Intended for verified API buyer discovery.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from urllib.parse import urlparse

from agno.tools import Toolkit

try:
    from crawl4ai import AsyncWebCrawler, CacheMode
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "`crawl4ai` is required for buyer discovery. Install it with `pip install crawl4ai`."
    ) from exc


ALLOWED_BUYER_DOMAINS = (
    # Regulatory
    "fda.gov",
    "orangebook.fda.gov",
    "ema.europa.eu",
    "pmda.go.jp",
    "pmda.jp",
    "cdsco.gov.in",
    "gov.uk",
    "mhra.gov.uk",
    "tga.gov.au",
    "health-canada.ca",
    # Pharmaceutical product catalogues / major manufacturers
    "cipla.com",
    "sunpharma.com",
    "drreddys.com",
    "lupin.com",
    "glenmarkpharma.com",
    "zyduscadila.com",
    "intaspharma.com",
    "alembicpharmaceuticals.com",
    "torrentpharma.com",
    "wockhardt.com",
    "mylan.com",
    "astrazeneca.com",
    # Trusted pharmacy listings
    "1mg.com",
    "netmeds.com",
    "drugs.com",
    "pharmeasy.in",
    "medPlusMart.com".lower(),
    "medindia.net",
    "tabletwise.net",
    "pharmacompass.com",
    "pharmaoffer.com",
)


class Crawl4aiBuyerTools(Toolkit):
    """Crawl4AI toolkit tailored for verified buyer discovery."""

    def __init__(self, max_length: Optional[int] = 2000):
        super().__init__(name="buyer_crawl4ai")
        self.max_length = max_length
        self.register(self.web_crawler)

    def _run_coroutine(self, coro):
        """Run coroutine, even if an event loop is already running."""
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # Probably called within an existing event loop (e.g., agent tool execution)
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

    def _is_allowed_domain(self, domain: str) -> bool:
        return any(domain.endswith(allowed) or allowed in domain for allowed in ALLOWED_BUYER_DOMAINS)

    def web_crawler(self, url: str, max_length: Optional[int] = None) -> str:
        """Synchronously crawl an allowed URL and return truncated markdown."""
        if not url:
            return "No URL provided"

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not self._is_allowed_domain(domain):
            return f"Blocked domain (not in allowed list): {domain}"

        target_length = max_length or self.max_length
        return self._run_coroutine(self._async_web_crawler(url, target_length))

    async def _async_web_crawler(self, url: str, max_length: Optional[int]) -> str:
        async with AsyncWebCrawler(thread_safe=True) as crawler:
            result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS)
            markdown = (result.markdown or "").strip()
            if not markdown:
                return "No result"

            snippet = markdown[:max_length] if max_length else markdown
            return snippet.replace("  ", " ")


