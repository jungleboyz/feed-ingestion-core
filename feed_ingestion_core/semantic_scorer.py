"""Semantic scoring — cosine similarity between content and an InterestProfile."""

from typing import Optional

import numpy as np

from .embeddings import EmbeddingService
from .interest_profile import InterestProfile


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, in [-1, 1]."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


class SemanticScorer:
    """Score items by semantic similarity to an InterestProfile's interest sentences."""

    def __init__(
        self,
        interest_profile: InterestProfile,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """Initialize the scorer.

        Args:
            interest_profile: The injected domain definition. Its ``interest_sentences``
                are embedded and averaged into the reference vector.
            embedding_service: EmbeddingService instance. Creates one if not provided.
        """
        self.interest_profile = interest_profile
        self.interests = interest_profile.interest_sentences
        self.embedding_service = embedding_service or EmbeddingService()
        self._interest_embedding: Optional[list[float]] = None

    @property
    def interest_embedding(self) -> list[float]:
        """Get or generate the combined interest embedding."""
        if self._interest_embedding is None:
            self._interest_embedding = self.generate_interest_embedding()
        return self._interest_embedding

    def generate_interest_embedding(self) -> list[float]:
        """Generate a single embedding by averaging the interest-sentence embeddings."""
        if not self.interests:
            raise ValueError("InterestProfile has no interest_sentences for semantic scoring")

        embeddings = self.embedding_service.batch_embed(self.interests)
        valid_embeddings = [e for e in embeddings if e]

        if not valid_embeddings:
            raise ValueError("Failed to generate embeddings for interest sentences")

        avg_embedding = np.mean(valid_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding.tolist()

    def score_item(
        self,
        item_embedding: list[float],
        min_score: float = 0.0,
        max_score: float = 1.0,
    ) -> float:
        """Score an item's embedding against the interest embedding."""
        if not item_embedding:
            return min_score

        similarity = cosine_similarity(self.interest_embedding, item_embedding)
        return max(min_score, min(max_score, similarity))

    def score_text(self, text: str) -> float:
        """Score raw text by embedding it first. Returns a score in [0.0, 1.0]."""
        if not text or not text.strip():
            return 0.0

        embedding = self.embedding_service.get_embedding(text)
        return self.score_item(embedding)

    def score_items_batch(self, embeddings: list[list[float]]) -> list[float]:
        """Score multiple pre-computed embeddings."""
        return [self.score_item(emb) for emb in embeddings]

    def is_relevant(self, score: float, threshold: float = 0.3) -> bool:
        """Whether a score clears the relevance threshold."""
        return score >= threshold

    def score_to_int(self, score: float, scale: int = 10) -> int:
        """Convert a [0.0, 1.0] float score to an integer on a 0-``scale`` range."""
        return int(round(score * scale))
