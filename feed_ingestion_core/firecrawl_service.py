"""Firecrawl SDK wrapper for article scraping and web search.

API key is passed in or read from ``FIRECRAWL_API_KEY`` — no app config dependency.
"""

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

from .logger import get_logger

logger = get_logger(__name__)

try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:  # pragma: no cover
    FIRECRAWL_AVAILABLE = False

# Max chars to return.
MAX_CONTENT_LENGTH = 4000

# Timeouts (seconds) — prevent indefinite hangs.
BATCH_SCRAPE_TIMEOUT = 120
SINGLE_SCRAPE_TIMEOUT = 30

# Singleton instance.
_instance: Optional["FirecrawlService"] = None


class FirecrawlService:
    """Wrapper around the Firecrawl SDK for scraping and search."""

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not api_key or not FIRECRAWL_AVAILABLE:
            self._client = None
        else:
            self._client = FirecrawlApp(api_key=api_key)
        self._quota_exhausted = False

    @property
    def available(self) -> bool:
        return self._client is not None and not self._quota_exhausted

    def _extract_markdown(self, doc) -> str:
        """Extract markdown from a scrape result (handles dict or object)."""
        if isinstance(doc, dict):
            return doc.get("markdown", "") or ""
        return getattr(doc, "markdown", "") or ""

    def _extract_metadata(self, doc) -> dict:
        """Extract metadata from a scrape result (handles dict or object)."""
        if isinstance(doc, dict):
            return doc.get("metadata", {}) or {}
        meta = getattr(doc, "metadata", None)
        if meta is None:
            return {}
        if isinstance(meta, dict):
            return meta
        return meta.dict() if hasattr(meta, "dict") else {}

    def scrape_article(self, url: str, max_length: int = MAX_CONTENT_LENGTH) -> Optional[str]:
        """Scrape a single URL and return markdown content, or None on failure."""
        if not self._client:
            return None
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._client.scrape, url, formats=["markdown"])
                doc = future.result(timeout=SINGLE_SCRAPE_TIMEOUT)
            markdown = self._extract_markdown(doc)
            if markdown:
                return markdown[:max_length]
            return None
        except FuturesTimeoutError:
            logger.warning("Firecrawl scrape timed out for %s", url)
            return None
        except Exception as e:
            error_msg = str(e)
            if "Payment Required" in error_msg or "Insufficient credits" in error_msg:
                if not self._quota_exhausted:
                    logger.warning("Firecrawl credits exhausted — skipping remaining scrapes")
                    self._quota_exhausted = True
                return None
            logger.warning("Firecrawl scrape failed for %s: %s", url, e)
            return None

    def batch_scrape(self, urls: list[str]) -> dict[str, str]:
        """Scrape multiple URLs, returning {url: markdown}. Failures are omitted."""
        if not self._client or not urls or self._quota_exhausted:
            return {}

        results: dict[str, str] = {}
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._client.batch_scrape, urls, formats=["markdown"])
                docs = future.result(timeout=BATCH_SCRAPE_TIMEOUT)
            for doc in docs:
                meta = self._extract_metadata(doc)
                source_url = meta.get("sourceURL", "") or meta.get("url", "")
                markdown = self._extract_markdown(doc)
                if source_url and markdown:
                    results[source_url] = markdown[:MAX_CONTENT_LENGTH]
        except FuturesTimeoutError:
            logger.warning("Firecrawl batch scrape timed out after %ss — skipping", BATCH_SCRAPE_TIMEOUT)
            return results
        except Exception as e:
            error_msg = str(e)
            if "Payment Required" in error_msg or "Insufficient credits" in error_msg:
                if not self._quota_exhausted:
                    logger.warning("Firecrawl credits exhausted — skipping remaining scrapes")
                    self._quota_exhausted = True
                return results
            logger.warning("Firecrawl batch scrape failed: %s", e)
            # Fall back to individual scrapes.
            for url in urls:
                text = self.scrape_article(url)
                if text:
                    results[url] = text
                if self._quota_exhausted:
                    break
        return results

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search the web; return a list of {url, title, markdown} dicts."""
        if not self._client:
            return []
        try:
            search_data = self._client.search(query)
            results = []
            items = search_data
            if hasattr(search_data, "data"):
                items = search_data.data or []
            elif hasattr(search_data, "web"):
                items = search_data.web or []
            elif not isinstance(search_data, list):
                items = []
            for item in items:
                if isinstance(item, dict):
                    url = item.get("url", "")
                    title = item.get("title", "")
                    markdown = (item.get("markdown", "") or "")[:MAX_CONTENT_LENGTH]
                else:
                    url = getattr(item, "url", "")
                    title = getattr(item, "title", "")
                    markdown = (getattr(item, "markdown", "") or "")[:MAX_CONTENT_LENGTH]
                if url:
                    results.append({"url": url, "title": title, "markdown": markdown})
                if len(results) >= limit:
                    break
            return results
        except Exception as e:
            logger.warning("Firecrawl search failed for '%s': %s", query, e)
            return []


def get_firecrawl_service(api_key: Optional[str] = None) -> FirecrawlService:
    """Get or create the FirecrawlService singleton."""
    global _instance
    if _instance is None:
        _instance = FirecrawlService(api_key=api_key)
    return _instance
