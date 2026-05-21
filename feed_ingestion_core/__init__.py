"""feed-ingestion-core — domain-agnostic content ingestion engine.

Lazy public API (PEP 562): ``import feed_ingestion_core`` is cheap; heavy dependencies
(chromadb, openai, …) are only imported when the relevant symbol is first accessed.
"""

__version__ = "0.1.0"

# Public name -> submodule it lives in.
_EXPORTS = {
    "InterestProfile": "interest_profile",
    "CoreConfig": "config",
    "EmbeddingService": "embeddings",
    "SemanticScorer": "semantic_scorer",
    "cosine_similarity": "semantic_scorer",
    "Scorer": "scoring_service",
    "VectorStore": "vector_store",
    "fetch_feeds_parallel": "feed_service",
    "TranscriptService": "transcript_service",
    "FirecrawlService": "firecrawl_service",
    "get_firecrawl_service": "firecrawl_service",
    "Summarizer": "summarizer",
    "SummarizerPrompts": "summarizer",
    "ReferenceVoice": "guest_detector",
    "GuestMatch": "guest_detector",
    "GuestDetector": "guest_detector",
    "configure": "service_registry",
    "get_config": "service_registry",
    "get_interest_profile": "service_registry",
    "get_embedding_service": "service_registry",
    "get_semantic_scorer": "service_registry",
    "get_scorer": "service_registry",
    "get_vector_store": "service_registry",
    "reset": "service_registry",
    "get_logger": "logger",
    "load_json": "cache_service",
    "save_json": "cache_service",
    "load_set": "cache_service",
    "save_set": "cache_service",
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str):
    """Lazily import a public symbol from its submodule (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
