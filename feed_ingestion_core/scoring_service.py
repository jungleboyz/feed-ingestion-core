"""Unified scoring — keyword + semantic — driven by an injected InterestProfile.

Replaces the news agent's module-level functions (which hardcoded ``AI_KEYWORDS``) with
a ``Scorer`` class that takes the keyword list and interest vector from an
``InterestProfile``. Quota state is per-Scorer instance, not a module global.
"""

import re
from typing import List, Optional, Tuple

from .embeddings import EmbeddingService
from .interest_profile import InterestProfile
from .logger import get_logger
from .semantic_scorer import SemanticScorer

logger = get_logger(__name__)


def norm(text: str) -> str:
    """Lowercase + collapse whitespace so matching is consistent."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


class Scorer:
    """Score content against an InterestProfile.

    Keyword scoring works with no API access. Semantic scoring lazily builds an
    embedding service + semantic scorer; on a quota/rate error it flips
    ``quota_exceeded`` and every subsequent call falls back to keyword scoring.
    """

    def __init__(
        self,
        interest_profile: InterestProfile,
        embedding_service: Optional[EmbeddingService] = None,
        semantic_scorer: Optional[SemanticScorer] = None,
        use_semantic: bool = True,
    ):
        self.interest_profile = interest_profile
        self.use_semantic = use_semantic
        self.quota_exceeded = False
        self._embedding_service = embedding_service
        self._semantic_scorer = semantic_scorer
        # Pre-normalise keywords once.
        self._norm_keywords = [norm(k) for k in interest_profile.keywords if k and k.strip()]

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def semantic_scorer(self) -> SemanticScorer:
        if self._semantic_scorer is None:
            self._semantic_scorer = SemanticScorer(
                interest_profile=self.interest_profile,
                embedding_service=self.embedding_service,
            )
        return self._semantic_scorer

    def score_keywords(self, title: str, description: str = "", transcript: str = "") -> int:
        """Keyword-based scoring. Title matches weight 3x, body matches 1x."""
        title_lower = norm(title)
        rest = norm(f"{description} {transcript}")
        score = 0
        for kw in self._norm_keywords:
            if kw in title_lower:
                score += 3
            elif kw in rest:
                score += 1
        return score

    def score_semantic(self, title: str, description: str = "", transcript: str = "") -> int:
        """Semantic score (0-10 int), falling back to keywords on failure/quota."""
        if not self.use_semantic or self.quota_exceeded:
            return self.score_keywords(title, description, transcript)

        try:
            text = f"{title} {description} {transcript}".strip()
            semantic_score = self.semantic_scorer.score_text(text)
            return self.semantic_scorer.score_to_int(semantic_score, scale=10)
        except Exception as e:
            self._handle_scoring_error(e)
            return self.score_keywords(title, description, transcript)

    def score_single_with_embedding(
        self, title: str, description: str = "", transcript: str = ""
    ) -> Tuple[int, Optional[float], Optional[list]]:
        """Score one item, returning ``(int_score, float_score, embedding)``.

        Callers that persist embeddings (podcast/video ingestion) use this.
        """
        if not self.use_semantic or self.quota_exceeded:
            return self.score_keywords(title, description, transcript), None, None

        try:
            text = f"{title} {description} {transcript}".strip()
            embedding = self.embedding_service.get_embedding(text)
            semantic_score = self.semantic_scorer.score_item(embedding)
            int_score = self.semantic_scorer.score_to_int(semantic_score, scale=10)
            return int_score, semantic_score, embedding
        except Exception as e:
            self._handle_scoring_error(e)
            return self.score_keywords(title, description, transcript), None, None

    def score_items_batch(self, items: List[dict]) -> List[dict]:
        """Score many items using batch embedding for efficiency.

        Each item needs ``title`` and ``summary``; ``score``, ``semantic_score`` and
        ``embedding`` are added in place.
        """
        total = len(items)

        if not self.use_semantic or not items or self.quota_exceeded:
            logger.info("Scoring %d items with keyword matching...", total)
            for item in items:
                item["score"] = self.score_keywords(item["title"], item.get("summary", ""))
                item["semantic_score"] = None
                item["embedding"] = None
            return items

        try:
            logger.info("Generating embeddings for %d items...", total)
            texts = [f"{it['title']} {it.get('summary', '')}".strip() for it in items]
            embeddings = self.embedding_service.batch_embed(texts)

            scored_count = 0
            fallback_count = 0
            for item, embedding in zip(items, embeddings):
                if embedding:
                    semantic_score = self.semantic_scorer.score_item(embedding)
                    item["score"] = self.semantic_scorer.score_to_int(semantic_score, scale=10)
                    item["semantic_score"] = semantic_score
                    item["embedding"] = embedding
                    scored_count += 1
                else:
                    item["score"] = self.score_keywords(item["title"], item.get("summary", ""))
                    item["semantic_score"] = None
                    item["embedding"] = None
                    fallback_count += 1

            logger.info(
                "Semantic scoring complete: %d semantic, %d keyword fallback",
                scored_count, fallback_count,
            )
            return items
        except Exception as e:
            self._handle_scoring_error(e)
            logger.info("Scoring %d items with keyword matching...", total)
            for item in items:
                item["score"] = self.score_keywords(item["title"], item.get("summary", ""))
                item["semantic_score"] = None
                item["embedding"] = None
            return items

    def _handle_scoring_error(self, e: Exception) -> None:
        """Flip the quota flag on a rate/quota error so later calls skip the API."""
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            if not self.quota_exceeded:
                logger.warning("Embedding API quota/rate limit hit — switching to keyword scoring")
                self.quota_exceeded = True
        else:
            logger.warning("Semantic scoring failed, using keywords: %s", e)
