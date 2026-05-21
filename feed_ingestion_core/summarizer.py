"""Summarization core — domain-agnostic, prompts injected by the consumer.

The news agent hardcoded "AI news digest" prompts. Here the prompts come from an
injected ``SummarizerPrompts`` so the health-insights agent (or any consumer) supplies
its own. Tries Anthropic Claude first (preferred), falls back to OpenAI; returns None if
neither is available or both fail, so the caller can apply its own fallback text.
"""

import os
from dataclasses import dataclass
from typing import Optional

from .logger import get_logger

logger = get_logger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False


@dataclass
class SummarizerPrompts:
    """Injected prompt set. ``user_template`` may reference ``{content}``,
    ``{title}`` and ``{source_name}`` placeholders."""

    system: str
    user_template: str

    def render_user(self, content: str, title: str = "", source_name: str = "") -> str:
        return self.user_template.format(
            content=content, title=title or "", source_name=source_name or ""
        )


class Summarizer:
    """Content summarizer with an Anthropic-then-OpenAI cascade."""

    def __init__(
        self,
        prompts: SummarizerPrompts,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        anthropic_model: str = "claude-3-haiku-20240307",
        prefer_anthropic: bool = True,
        max_content_chars: int = 12000,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.prompts = prompts
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        self.prefer_anthropic = prefer_anthropic
        self.max_content_chars = max_content_chars
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    def _anthropic_client(self):
        if not ANTHROPIC_AVAILABLE or not self._anthropic_key:
            return None
        return anthropic.Anthropic(api_key=self._anthropic_key)

    def _openai_client(self):
        if not OPENAI_AVAILABLE or not self._openai_key or self._openai_key == "your-openai-api-key":
            return None
        return openai.OpenAI(api_key=self._openai_key)

    def summarize(self, content: str, title: str = "", source_name: str = "") -> Optional[str]:
        """Summarize *content*. Returns the summary, or None if unavailable/failed."""
        if not content or not content.strip():
            return None

        trimmed = content[: self.max_content_chars]
        user_prompt = self.prompts.render_user(trimmed, title=title, source_name=source_name)

        order = ["anthropic", "openai"] if self.prefer_anthropic else ["openai", "anthropic"]
        for provider in order:
            try:
                if provider == "anthropic":
                    result = self._summarize_anthropic(user_prompt)
                else:
                    result = self._summarize_openai(user_prompt)
                if result:
                    return result
            except Exception as e:
                logger.warning("Summarizer provider %s failed: %s", provider, e)
        return None

    def _summarize_anthropic(self, user_prompt: str) -> Optional[str]:
        client = self._anthropic_client()
        if client is None:
            return None
        response = client.messages.create(
            model=self.anthropic_model,
            max_tokens=self.max_tokens,
            system=self.prompts.system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    def _summarize_openai(self, user_prompt: str) -> Optional[str]:
        client = self._openai_client()
        if client is None:
            return None
        response = client.chat.completions.create(
            model=self.openai_model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.prompts.system},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
