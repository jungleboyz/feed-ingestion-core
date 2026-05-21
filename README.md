# feed-ingestion-core

Domain-agnostic content ingestion engine — feed fetching, transcript extraction,
embeddings, semantic scoring, vector dedup, summarization, and interview-show guest
detection.

Extracted from `ai-news-agent` per **ARCH-008** so multiple apps can share one engine.
The domain (AI news, health research, …) is supplied by the consumer as an
`InterestProfile`; the engine itself knows nothing about any subject area.

## Consumers

- `health-insights-agent` — health research ingestion (ARCH-008)
- `ai-news-agent` — AI news (migrated onto the package in ARCH-008 P5)

## Usage sketch

```python
from feed_ingestion_core import InterestProfile, configure, get_scorer

profile = InterestProfile(
    keywords=["sleep", "hypertrophy", "VO2 max", ...],
    interest_sentences=["evidence-based strength training", ...],
    domains={"sleep": ["circadian", "sleep quality"], "nutrition": [...]},
)
configure(interest_profile=profile, chromadb_dir="chromadb_data")

scorer = get_scorer()
score = scorer.score_semantic(title, description, transcript)
```

## Versioning

Consumers pin a version. A package change must not silently break a live app — bump the
version and run consumer CI before adopting.
