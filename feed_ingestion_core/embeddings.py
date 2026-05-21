"""Embedding service with cascading provider support (OpenAI -> Jina).

Self-contained: keys are passed in or read from the environment — no app config import.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

from .logger import get_logger

logger = get_logger(__name__)

import requests
import tiktoken
from openai import OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Constants
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIMENSIONS = 1536
OPENAI_MAX_TOKENS = 8191

JINA_MODEL = "jina-embeddings-v3"
JINA_DIMENSIONS = 1024
JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    name: str
    dimensions: int

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this provider has the required configuration."""
        ...

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    @abstractmethod
    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class OpenAIProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small provider."""

    name = "openai"
    dimensions = OPENAI_DIMENSIONS

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._tokenizer = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL)
        return self._tokenizer

    def truncate_text(self, text: str, max_tokens: int = OPENAI_MAX_TOKENS) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def get_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        text = self.truncate_text(text.strip())
        try:
            response = self.client.embeddings.create(
                model=OPENAI_MODEL,
                input=text,
                dimensions=OPENAI_DIMENSIONS,
            )
            return response.data[0].embedding
        except RateLimitError:
            raise

    def batch_embed(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        if not texts:
            return []

        processed_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                processed_texts.append(self.truncate_text(text.strip()))
                valid_indices.append(i)

        if not processed_texts:
            return [[] for _ in texts]

        all_embeddings = []
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            response = self._call_api(batch)
            all_embeddings.extend([item.embedding for item in response.data])

        result = [[] for _ in texts]
        for idx, embedding in zip(valid_indices, all_embeddings):
            result[idx] = embedding
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _call_api(self, texts: list[str]):
        try:
            return self.client.embeddings.create(
                model=OPENAI_MODEL,
                input=texts,
                dimensions=OPENAI_DIMENSIONS,
            )
        except RateLimitError:
            raise

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


class JinaProvider(EmbeddingProvider):
    """Jina AI embedding provider (free tier, OpenAI-compatible API)."""

    name = "jina"
    dimensions = JINA_DIMENSIONS

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("JINA_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": JINA_MODEL,
            "input": texts,
            "dimensions": JINA_DIMENSIONS,
        }
        resp = requests.post(JINA_ENDPOINT, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def get_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        embeddings = self._call_api([text.strip()])
        return embeddings[0]

    def batch_embed(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        if not texts:
            return []

        processed_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                processed_texts.append(text.strip())
                valid_indices.append(i)

        if not processed_texts:
            return [[] for _ in texts]

        all_embeddings = []
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            all_embeddings.extend(self._call_api(batch))

        result = [[] for _ in texts]
        for idx, embedding in zip(valid_indices, all_embeddings):
            result[idx] = embedding
        return result


# Provider registry
PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "jina": JinaProvider,
}


class EmbeddingService:
    """Generate embeddings with cascading provider fallback.

    Tries providers in order. Default cascade: OpenAI -> Jina. Keys are passed in or
    read from the environment (``OPENAI_API_KEY`` / ``JINA_API_KEY``).
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        embedding_providers: str = "openai,jina",
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")
        provider_names = [p.strip() for p in embedding_providers.split(",") if p.strip()]

        self._providers: list[EmbeddingProvider] = []
        for name in provider_names:
            cls = PROVIDER_CLASSES.get(name)
            if cls is None:
                logger.warning("Unknown embedding provider: %s, skipping", name)
                continue
            if name == "openai":
                provider = cls(api_key=openai_api_key)
            elif name == "jina":
                provider = cls(api_key=jina_api_key)
            else:
                provider = cls()
            self._providers.append(provider)

        if not self._providers:
            raise RuntimeError("No embedding providers configured")

        # Expose dimensions from the first provider.
        self.dimensions = self._providers[0].dimensions
        self.model = OPENAI_MODEL  # backward-compat

        self._openai_provider = next(
            (p for p in self._providers if isinstance(p, OpenAIProvider)), None
        )
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self._openai_provider:
                self._tokenizer = self._openai_provider.tokenizer
            else:
                self._tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL)
        return self._tokenizer

    def truncate_text(self, text: str, max_tokens: int = OPENAI_MAX_TOKENS) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text, cascading through providers."""
        last_error = None
        for provider in self._providers:
            if not provider.available:
                continue
            try:
                result = provider.get_embedding(text)
                self.dimensions = provider.dimensions
                return result
            except Exception as e:
                logger.warning("Embedding provider %s failed: %s, trying next...", provider.name, e)
                last_error = e
        raise RuntimeError(f"All embedding providers failed. Last error: {last_error}")

    def batch_embed(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        """Generate embeddings for multiple texts, cascading through providers."""
        if not texts:
            return []

        last_error = None
        for provider in self._providers:
            if not provider.available:
                continue
            try:
                result = provider.batch_embed(texts, batch_size=batch_size)
                self.dimensions = provider.dimensions
                logger.info("Embeddings generated via %s (%d texts)", provider.name, len(texts))
                return result
            except Exception as e:
                logger.warning("Embedding provider %s failed: %s, trying next...", provider.name, e)
                last_error = e
        raise RuntimeError(f"All embedding providers failed. Last error: {last_error}")
