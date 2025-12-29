import asyncio
from typing import Optional
from urllib.parse import urlparse

from agno.tools import Toolkit

try:
    from crawl4ai import AsyncWebCrawler, CacheMode
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "`crawl4ai` is required for buyer discovery crawling. Install it with `pip install crawl4ai`."
    ) from exc


ALLOWED_BUYER_DOMAINS = (
    # Regulatory / official drug listings
    "cdsco.gov.in",
    "fda.gov",
    "orangebook.fda.gov",
    "ema.europa.eu",
    "pmda.go.jp",
    "gov.uk",
    "tga.gov.au",
    "health-canada.ca",
    "medsafe.govt.nz",
    # Major pharmaceutical manufacturers with verified product catalogs
    "cipla.com",
    "sunpharma.com",
    "drreddys.com",
    "lupin.com",
    "glenmarkpharma.com",
    "torrentpharma.com",
    "alembicpharmaceuticals.com",
    "zyduscadila.com",
    "intaspharma.com",
    "alkemlabs.com",
    "wockhardt.com",
    "mylan.com",
    "pfizer.com",
    "astrazeneca.com",
    "novartis.com",
    "sanofi.com",
    # Trusted pharmacy/product portals
    "1mg.com",
    "netmeds.com",
    "drugs.com",
    "pharmeasy.in",
    "apollo247.com",
    # Trade / directory / import-export style sources
    "pharmacompass.com",
    "chemicalbook.com",
    "importyeti.com",
    "zauba.com",
    "panjiva.com",
)


class BuyerCrawl4aiTools(Toolkit):
    """
    Crawl4AI toolkit configured for API buyer discovery.
    Restricts crawling to regulator and trusted pharmaceutical domains.
    """

    def __init__(self, max_length: Optional[int] = 2000):
        super().__init__(name="buyer_crawl4ai")
        self.max_length = max_length
        self.register(self.web_crawler)

    def _is_allowed_domain(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(domain.endswith(allowed) or allowed in domain for allowed in ALLOWED_BUYER_DOMAINS)

    def web_crawler(self, url: str, max_length: Optional[int] = None) -> str:
        """
        Synchronously crawl an allowed URL and return truncated markdown snippet.

        Uses Crawl4AI with deeper crawling configuration (best-effort):
        - max_depth=3  (follow links a few hops)
        - crawl_all=True (try to capture full content on the page)
        """
        if not url:
            return "No URL provided"

        if not self._is_allowed_domain(url):
            return f"Blocked domain: {url}"

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(
                    self._async_web_crawler(url, max_length or self.max_length)
                )
            finally:
                new_loop.close()

        return asyncio.run(self._async_web_crawler(url, max_length or self.max_length))

    async def _async_web_crawler(self, url: str, max_length: Optional[int]) -> str:
        # Best-effort deep crawl: rely on Crawl4AI's options where available.
        async with AsyncWebCrawler(thread_safe=True) as crawler:
            result = await crawler.arun(
                url=url,
                cache_mode=CacheMode.BYPASS,
                max_depth=3,
                crawl_all=True,
            )
            markdown = (result.markdown or "").strip()
            if not markdown:
                return "No result"

            snippet = markdown[:max_length] if max_length else markdown
            return snippet.replace("  ", " ")

