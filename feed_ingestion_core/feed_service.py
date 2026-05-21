"""Parallel feed fetching.

Only the domain-agnostic ``fetch_feeds_parallel`` is part of the package. The news
agent's ``update_feed_statuses`` coupled to its own DB (``web.database``/``web.models``)
and stays app-side — each consumer persists feed status its own way.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

from .logger import get_logger

logger = get_logger(__name__)


def fetch_feeds_parallel(
    feed_urls: List[str],
    parse_fn: Callable[[str], list],
    max_workers: int = 10,
    timeout: int = 15,
) -> dict:
    """Fetch multiple feeds concurrently.

    Args:
        feed_urls: Feed URLs to fetch.
        parse_fn: Callable taking a URL and returning a list of items.
        max_workers: Max concurrent threads.
        timeout: Per-future timeout in seconds.

    Returns:
        Dict mapping feed_url -> list of items returned by parse_fn (empty list on error).
    """
    results: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(parse_fn, url): url for url in feed_urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                items = future.result(timeout=timeout)
                results[url] = items or []
            except Exception as e:
                logger.warning("Feed fetch failed for %s: %s", url, e)
                results[url] = []

    return results
