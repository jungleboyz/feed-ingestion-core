"""ChromaDB vector store wrapper for semantic search and dedup.

Collection names are injectable so each consumer can name its own collections
(the news agent uses news/podcast/video; the health agent uses its own).
"""

import os
from typing import Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover
    CHROMADB_AVAILABLE = False

from .embeddings import EmbeddingService

DEFAULT_COLLECTIONS = {
    "news": "news_items",
    "podcast": "podcasts",
    "video": "videos",
}


class VectorStore:
    """ChromaDB wrapper for storing and searching embeddings."""

    def __init__(
        self,
        persist_dir: str = "./chromadb_data",
        embedding_service: Optional[EmbeddingService] = None,
        collections: Optional[dict[str, str]] = None,
    ):
        """Initialize the vector store.

        Args:
            persist_dir: Directory for persistent storage.
            embedding_service: EmbeddingService for generating embeddings.
            collections: Map of item_type -> ChromaDB collection name. Defaults to the
                news agent's news/podcast/video set; consumers may pass their own.
        """
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

        self.collections = collections or dict(DEFAULT_COLLECTIONS)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_service = embedding_service or EmbeddingService()
        self._collections: dict[str, "chromadb.Collection"] = {}

    def _get_collection(self, item_type: str) -> "chromadb.Collection":
        """Get or create a collection for the given item type."""
        if item_type not in self.collections:
            raise ValueError(
                f"Unknown item type: {item_type}. Must be one of {list(self.collections.keys())}"
            )

        collection_name = self.collections[item_type]
        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[collection_name]

    def add_item(
        self,
        item_id: str,
        text: str,
        item_type: str,
        metadata: Optional[dict[str, Any]] = None,
        embedding: Optional[list[float]] = None,
    ) -> str:
        """Add (upsert) an item to the vector store."""
        collection = self._get_collection(item_type)

        if embedding is None:
            embedding = self.embedding_service.get_embedding(text)

        meta = metadata or {}
        meta["item_type"] = item_type

        collection.upsert(
            ids=[item_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[text[:5000]],
        )
        return item_id

    def add_items_batch(
        self,
        items: list[dict[str, Any]],
        item_type: str,
    ) -> list[str]:
        """Add multiple items. Each dict: id, text, metadata (opt), embedding (opt)."""
        if not items:
            return []

        collection = self._get_collection(item_type)

        ids = [item["id"] for item in items]
        texts = [item["text"] for item in items]

        embeddings: list = []
        texts_to_embed = []
        embed_indices = []

        for i, item in enumerate(items):
            if "embedding" in item and item["embedding"]:
                embeddings.append(item["embedding"])
            else:
                embeddings.append(None)
                texts_to_embed.append(texts[i])
                embed_indices.append(i)

        if texts_to_embed:
            new_embeddings = self.embedding_service.batch_embed(texts_to_embed)
            for i, embedding in zip(embed_indices, new_embeddings):
                embeddings[i] = embedding

        valid_ids = []
        valid_embeddings = []
        valid_metadatas = []
        valid_documents = []

        for i, (item_id, embedding, item) in enumerate(zip(ids, embeddings, items)):
            if embedding:
                valid_ids.append(item_id)
                valid_embeddings.append(embedding)
                meta = item.get("metadata", {})
                meta["item_type"] = item_type
                valid_metadatas.append(meta)
                valid_documents.append(texts[i][:5000])

        if valid_ids:
            collection.upsert(
                ids=valid_ids,
                embeddings=valid_embeddings,
                metadatas=valid_metadatas,
                documents=valid_documents,
            )
        return valid_ids

    def search(
        self,
        query_text: str,
        item_type: Optional[str] = None,
        limit: int = 10,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar items by text."""
        query_embedding = self.embedding_service.get_embedding(query_text)
        return self.search_by_embedding(query_embedding, item_type, limit, where)

    def search_by_embedding(
        self,
        embedding: list[float],
        item_type: Optional[str] = None,
        limit: int = 10,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar items by a pre-computed embedding."""
        results = []

        collections_to_search = (
            [self._get_collection(item_type)]
            if item_type
            else [self._get_collection(t) for t in self.collections.keys()]
        )

        for collection in collections_to_search:
            try:
                query_result = collection.query(
                    query_embeddings=[embedding],
                    n_results=limit,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )

                if query_result["ids"] and query_result["ids"][0]:
                    for i, item_id in enumerate(query_result["ids"][0]):
                        distance = query_result["distances"][0][i]
                        similarity = 1 - distance
                        results.append({
                            "id": item_id,
                            "text": query_result["documents"][0][i] if query_result["documents"] else "",
                            "metadata": query_result["metadatas"][0][i] if query_result["metadatas"] else {},
                            "similarity": similarity,
                        })
            except Exception:
                continue

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def find_similar(
        self,
        embedding: list[float],
        item_type: str,
        threshold: float = 0.95,
        exclude_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Find items similar to *embedding* above *threshold* (duplicate detection)."""
        results = self.search_by_embedding(embedding, item_type, limit=10)

        filtered = []
        for result in results:
            if result["similarity"] >= threshold:
                if exclude_ids and result["id"] in exclude_ids:
                    continue
                filtered.append(result)
        return filtered

    def get_item(self, item_id: str, item_type: str) -> Optional[dict[str, Any]]:
        """Get a specific item by ID."""
        collection = self._get_collection(item_type)
        try:
            result = collection.get(
                ids=[item_id],
                include=["documents", "metadatas", "embeddings"],
            )
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0] if result["documents"] else "",
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                    "embedding": result["embeddings"][0] if result["embeddings"] else [],
                }
        except Exception:
            pass
        return None

    def delete_item(self, item_id: str, item_type: str) -> bool:
        """Delete an item from the vector store."""
        collection = self._get_collection(item_type)
        try:
            collection.delete(ids=[item_id])
            return True
        except Exception:
            return False

    def get_collection_count(self, item_type: str) -> int:
        """Number of items in a collection."""
        collection = self._get_collection(item_type)
        return collection.count()
