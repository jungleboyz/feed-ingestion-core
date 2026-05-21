"""InterestProfile — the injected domain definition.

This is the central abstraction that turns a generic ingestion engine into a
domain-specific one. The AI news agent and the health-insights agent are identical
engines configured with different InterestProfiles.

- ``keywords``           — terms for fast keyword-match scoring.
- ``interest_sentences`` — natural-language descriptions, embedded and averaged into a
                           single vector for semantic-similarity scoring.
- ``domains``            — label -> terms map, used to classify content into a taxonomy
                           (e.g. the 16-domain health taxonomy in ARCH-008).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List


def _norm(text: str) -> str:
    """Lowercase + collapse whitespace for consistent matching."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


@dataclass
class InterestProfile:
    """Domain definition injected into the scoring engine."""

    keywords: List[str]
    interest_sentences: List[str]
    domains: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.keywords and not self.interest_sentences:
            raise ValueError(
                "InterestProfile needs at least keywords or interest_sentences"
            )
        # Pre-normalise keywords once for matching.
        self._norm_keywords = [_norm(k) for k in self.keywords if k and k.strip()]
        self._norm_domains = {
            label: [_norm(t) for t in terms if t and t.strip()]
            for label, terms in self.domains.items()
        }

    def classify_domains(self, text: str) -> List[str]:
        """Return the domain labels whose terms appear in *text*.

        Used to tag ingested content into the consumer's taxonomy. Returns labels in
        descending order of match count (most-relevant domain first).
        """
        haystack = _norm(text)
        if not haystack:
            return []
        hits: List[tuple] = []
        for label, terms in self._norm_domains.items():
            count = sum(1 for t in terms if t and t in haystack)
            if count:
                hits.append((label, count))
        hits.sort(key=lambda x: x[1], reverse=True)
        return [label for label, _ in hits]

    def to_dict(self) -> dict:
        """Serialise (for logging / persistence)."""
        return {
            "keywords": self.keywords,
            "interest_sentences": self.interest_sentences,
            "domains": self.domains,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InterestProfile":
        """Rebuild from a ``to_dict()`` payload."""
        return cls(
            keywords=data.get("keywords", []),
            interest_sentences=data.get("interest_sentences", []),
            domains=data.get("domains", {}),
        )
