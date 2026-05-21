"""Configurable singleton registry for the ingestion engine.

The consumer calls ``configure(interest_profile=..., config=...)`` once at startup;
``get_*`` accessors then return shared, lazily-built services. This replaces the news
agent's hardcoded registry — the domain now comes from the injected InterestProfile.
"""

from typing import Optional

from .config import CoreConfig
from .embeddings import EmbeddingService
from .interest_profile import InterestProfile
from .scoring_service import Scorer
from .semantic_scorer import SemanticScorer

_config: Optional[CoreConfig] = None
_interest_profile: Optional[InterestProfile] = None

_embedding_service: Optional[EmbeddingService] = None
_vector_store = None
_semantic_scorer: Optional[SemanticScorer] = None
_scorer: Optional[Scorer] = None


def configure(
    interest_profile: InterestProfile,
    config: Optional[CoreConfig] = None,
) -> None:
    """Configure the engine. Call once at application startup.

    Args:
        interest_profile: The domain definition (keywords, interest sentences, domains).
        config: Runtime config (API keys, chromadb dir, …). Defaults to ``CoreConfig.from_env()``.
    """
    global _config, _interest_profile
    global _embedding_service, _vector_store, _semantic_scorer, _scorer

    _interest_profile = interest_profile
    _config = config or CoreConfig.from_env()
    # Drop any previously built singletons so a re-configure takes effect.
    _embedding_service = None
    _vector_store = None
    _semantic_scorer = None
    _scorer = None


def _require_configured() -> None:
    if _interest_profile is None or _config is None:
        raise RuntimeError(
            "feed_ingestion_core is not configured — "
            "call configure(interest_profile=...) before using get_* accessors"
        )


def get_config() -> CoreConfig:
    _require_configured()
    return _config  # type: ignore[return-value]


def get_interest_profile() -> InterestProfile:
    _require_configured()
    return _interest_profile  # type: ignore[return-value]


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    _require_configured()
    if _embedding_service is None:
        _embedding_service = EmbeddingService(
            openai_api_key=_config.openai_api_key,
            jina_api_key=_config.jina_api_key,
            embedding_providers=_config.embedding_providers,
        )
    return _embedding_service


def get_semantic_scorer() -> SemanticScorer:
    """Get or create the semantic scorer singleton."""
    global _semantic_scorer
    _require_configured()
    if _semantic_scorer is None:
        _semantic_scorer = SemanticScorer(
            interest_profile=_interest_profile,
            embedding_service=get_embedding_service(),
        )
    return _semantic_scorer


def get_scorer() -> Scorer:
    """Get or create the unified Scorer singleton."""
    global _scorer
    _require_configured()
    if _scorer is None:
        _scorer = Scorer(
            interest_profile=_interest_profile,
            embedding_service=get_embedding_service(),
            semantic_scorer=get_semantic_scorer(),
            use_semantic=_config.use_semantic_scoring,
        )
    return _scorer


def get_vector_store():
    """Get or create the vector store singleton."""
    global _vector_store
    _require_configured()
    if _vector_store is None:
        # Lazy import — chromadb is heavy and only needed when dedup/search is used.
        from .vector_store import VectorStore
        _vector_store = VectorStore(
            persist_dir=_config.chromadb_dir,
            embedding_service=get_embedding_service(),
        )
    return _vector_store


def reset() -> None:
    """Clear all configuration and singletons (used by tests)."""
    global _config, _interest_profile
    global _embedding_service, _vector_store, _semantic_scorer, _scorer
    _config = None
    _interest_profile = None
    _embedding_service = None
    _vector_store = None
    _semantic_scorer = None
    _scorer = None
