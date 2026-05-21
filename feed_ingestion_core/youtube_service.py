"""Shared YouTube Transcript API instance with optional Webshare proxy.

transcript_service imports ``youtube_api`` from here so the proxy is configured once.
Proxy credentials come from the environment (``WEBSHARE_PROXY_USERNAME`` /
``WEBSHARE_PROXY_PASSWORD``) — cloud IPs are frequently blocked by YouTube, so a proxy
is recommended in production.
"""

import os
from typing import Optional

from .logger import get_logger

logger = get_logger(__name__)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    from youtube_transcript_api.proxies import WebshareProxyConfig

    YOUTUBE_TRANSCRIPT_AVAILABLE = True

    _proxy_username = os.getenv("WEBSHARE_PROXY_USERNAME")
    _proxy_password = os.getenv("WEBSHARE_PROXY_PASSWORD")

    if _proxy_username and _proxy_password:
        youtube_api: Optional[YouTubeTranscriptApi] = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=_proxy_username,
                proxy_password=_proxy_password,
            )
        )
        logger.info("YouTube transcript API: using Webshare proxy")
    else:
        youtube_api = YouTubeTranscriptApi()

except ImportError:  # pragma: no cover
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    youtube_api = None

    # Stub error classes so callers can still reference them.
    class TranscriptsDisabled(Exception):  # type: ignore[no-redef]
        pass

    class NoTranscriptFound(Exception):  # type: ignore[no-redef]
        pass

    class VideoUnavailable(Exception):  # type: ignore[no-redef]
        pass

    logger.warning("youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
