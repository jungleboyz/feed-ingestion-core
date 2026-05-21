"""Cascading transcript service.

Cascade order:
1. Web transcript scraping (episode page via Firecrawl/BS4)
2. YouTube transcript (youtube-transcript-api)
3. RSS description fallback (always available)
"""

import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .firecrawl_service import get_firecrawl_service
from .youtube_service import (
    youtube_api,
    YOUTUBE_TRANSCRIPT_AVAILABLE,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

_USER_AGENT = "Mozilla/5.0 (compatible; FeedIngestionCore/1.0)"

# Markers that indicate a transcript section on a web page.
TRANSCRIPT_MARKERS = [
    "transcript", "full text", "show notes", "episode transcript",
    "read the transcript", "full transcript",
]

# Minimum length to consider scraped text a usable transcript.
MIN_TRANSCRIPT_LENGTH = 500


class TranscriptService:
    """Cascading transcript fetcher: web -> YouTube -> description."""

    def get_transcript(
        self,
        title: str,
        link: str,
        audio_url: Optional[str] = None,
        video_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> tuple[str, str]:
        """Try each source in order and return (text, source).

        source is one of 'web', 'youtube', or 'description'.
        """
        web_text = self._try_web_transcript(link)
        if web_text:
            return web_text, "web"

        yt_text = self._try_youtube_transcript(title, video_id=video_id)
        if yt_text:
            return yt_text, "youtube"

        return self._use_description(description or ""), "description"

    def _try_web_transcript(self, link: str) -> Optional[str]:
        """Scrape the episode page for transcript content."""
        if not link:
            return None

        fc = get_firecrawl_service()
        if fc.available:
            markdown = fc.scrape_article(link)
            if markdown and self._looks_like_transcript(markdown):
                return markdown

        try:
            html = requests.get(
                link, timeout=10, headers={"User-Agent": _USER_AGENT}
            ).text
            soup = BeautifulSoup(html, "html.parser")

            for marker in TRANSCRIPT_MARKERS:
                for tag in soup.find_all(["h1", "h2", "h3", "h4", "div", "section"]):
                    if marker in (tag.get_text() or "").lower():
                        parent = tag.parent or tag
                        text = parent.get_text(separator=" ", strip=True)
                        if len(text) >= MIN_TRANSCRIPT_LENGTH:
                            return text[:12000]

            paragraphs = soup.find_all("p")
            full_text = " ".join(p.get_text(strip=True) for p in paragraphs)
            if len(full_text) >= MIN_TRANSCRIPT_LENGTH * 2:
                return full_text[:12000]
        except Exception:
            pass

        return None

    def _looks_like_transcript(self, text: str) -> bool:
        """Heuristic: does scraped text look like a transcript?"""
        if len(text) < MIN_TRANSCRIPT_LENGTH:
            return False
        sentence_count = len(re.findall(r'[.!?]\s', text))
        return sentence_count >= 10

    def _try_youtube_transcript(
        self, title: str, video_id: Optional[str] = None
    ) -> Optional[str]:
        """Fetch a YouTube transcript if a video_id is available or discoverable."""
        if not YOUTUBE_TRANSCRIPT_AVAILABLE or youtube_api is None:
            return None
        if not video_id:
            video_id = self._search_youtube_id(title)
        if not video_id:
            return None

        try:
            transcript = youtube_api.fetch(video_id)
            full_text = " ".join(entry.text for entry in transcript)
            return full_text[:12000] if full_text else None
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
            return None
        except Exception:
            return None

    def _search_youtube_id(self, title: str) -> Optional[str]:
        """Find a YouTube video ID by scraping a search results page (no API key)."""
        try:
            query = f"{title} full episode"
            url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
            resp = requests.get(url, timeout=10, headers={"User-Agent": _USER_AGENT})
            match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
            if match:
                return match.group(1)
        except Exception:
            pass
        return None

    def _use_description(self, description: str) -> str:
        """Return the RSS description, stripped of HTML."""
        if "<" in description:
            soup = BeautifulSoup(description, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        return description.strip()
