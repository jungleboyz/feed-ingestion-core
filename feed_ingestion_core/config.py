"""Lightweight config for feed-ingestion-core.

The package is self-contained — it does NOT import any consuming app's config module.
Each service accepts its keys directly; ``CoreConfig`` is a convenience bundle that
``service_registry.configure()`` accepts, with environment-variable fallbacks.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CoreConfig:
    """Runtime configuration for the ingestion engine."""

    openai_api_key: Optional[str] = None
    jina_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    embedding_providers: str = "openai,jina"
    webshare_proxy_username: Optional[str] = None
    webshare_proxy_password: Optional[str] = None
    chromadb_dir: str = "chromadb_data"
    use_semantic_scoring: bool = True

    @classmethod
    def from_env(cls) -> "CoreConfig":
        """Build a config from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            jina_api_key=os.getenv("JINA_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
            embedding_providers=os.getenv("EMBEDDING_PROVIDERS", "openai,jina"),
            webshare_proxy_username=os.getenv("WEBSHARE_PROXY_USERNAME"),
            webshare_proxy_password=os.getenv("WEBSHARE_PROXY_PASSWORD"),
            chromadb_dir=os.getenv("CHROMADB_DIR", "chromadb_data"),
            use_semantic_scoring=os.getenv("USE_SEMANTIC_SCORING", "true").lower() == "true",
        )
