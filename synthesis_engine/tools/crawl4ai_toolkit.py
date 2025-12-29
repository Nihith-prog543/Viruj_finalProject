import asyncio
from typing import Optional
from urllib.parse import urlparse

from agno.tools import Toolkit

try:
    from crawl4ai import AsyncWebCrawler, CacheMode
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "`crawl4ai` is required for regulator crawling. Install it with `pip install crawl4ai`."
    ) from exc


ALLOWED_REGULATOR_DOMAINS = (
    "fda.gov",
    "ema.europa.eu",
    "pmda.go.jp",
    "cdsco.gov.in",
    "gov.uk",
    "health-canada.ca",
    "tga.gov.au",
    "pmda.jp",
    "mhra.gov.uk",
    "dcgi.gov.in",
)


class Crawl4aiTools(Toolkit):
    """
    Minimal drop-in replacement for the phi Crawl4ai toolkit.
    Restricts crawling to regulator-backed domains and returns markdown snippets.
    """

    def __init__(self, max_length: Optional[int] = 2000):
        super().__init__(name="regulator_crawl4ai")
        self.max_length = max_length
        self.register(self.web_crawler)

    def web_crawler(self, url: str, max_length: Optional[int] = None) -> str:
        """Synchronously crawl an allowed URL and return truncated markdown."""
        if not url:
            return "No URL provided"

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not any(domain.endswith(allowed) for allowed in ALLOWED_REGULATOR_DOMAINS):
            return f"Blocked non-regulator domain: {domain}"

        return asyncio.run(self._async_web_crawler(url, max_length or self.max_length))

    async def _async_web_crawler(self, url: str, max_length: Optional[int]) -> str:
        async with AsyncWebCrawler(thread_safe=True) as crawler:
            result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS)
            markdown = (result.markdown or "").strip()
            if not markdown:
                return "No result"

            snippet = markdown[:max_length] if max_length else markdown
            return snippet.replace("  ", " ")


