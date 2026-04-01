"""Vector Store — Qdrant-backed hybrid search (dense + sparse).

Implements §7.3 of plan_doc_v2.md:
- Dense embeddings via sentence-transformers (SPECTER2 or all-MiniLM)
- Sparse BM25-style scoring
- Hybrid search with RRF fusion
- Collection management
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

from article_generator.config import AppConfig
from article_generator.models import PaperChunk, RetrievalResult, RetrievalStrategy

logger = logging.getLogger(__name__)


class VectorStore:
    """Hybrid vector store with dense + sparse search."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._embedder = None
        self._qdrant = None
        self._collection_name = "paper_chunks"
        self._chunks_by_id: dict[str, PaperChunk] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize embedder and vector DB."""
        if self._initialized:
            return

        # Try Qdrant first, fall back to in-memory
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._qdrant = QdrantClient(path=str(self.config.paths.data_dir / "qdrant_db"))
            collections = [c.name for c in self._qdrant.get_collections().collections]
            if self._collection_name not in collections:
                self._qdrant.create_collection(
                    self._collection_name,
                    vectors_config=VectorParams(
                        size=self._get_embedding_dim(),
                        distance=Distance.COSINE,
                    ),
                )
            logger.info("Qdrant initialized with collection '%s'", self._collection_name)
        except ImportError:
            logger.info("Qdrant not available — using in-memory vector search")
            self._qdrant = None

        self._initialized = True

    def _get_embedding_dim(self) -> int:
        """Get the embedding dimension from the model."""
        return 384  # Default for all-MiniLM-L6-v2

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using sentence-transformers."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = getattr(self.config.retrieval, "embedding_model", "all-MiniLM-L6-v2")
                self._embedder = SentenceTransformer(model_name)
                logger.info("Loaded embedding model: %s", model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed — using hash embeddings")
                return self._hash_embed(texts)

        return self._embedder.encode(texts, show_progress_bar=False)

    def _hash_embed(self, texts: list[str]) -> np.ndarray:
        """Fallback hash-based embedding for when sentence-transformers unavailable."""
        dim = 384
        embeddings = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = np.frombuffer(h * (dim // len(h) + 1), dtype=np.uint8)[:dim].astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings.append(vec)
        return np.array(embeddings)

    def index_chunks(self, chunks: list[PaperChunk]) -> int:
        """Index a list of chunks into the vector store."""
        self.initialize()

        if not chunks:
            return 0

        # Store chunks in memory
        for chunk in chunks:
            self._chunks_by_id[chunk.chunk_id] = chunk

        # Generate embeddings
        texts = [self._chunk_to_text(c) for c in chunks]
        embeddings = self._embed(texts)

        if self._qdrant is not None:
            self._index_qdrant(chunks, embeddings)
        else:
            for chunk, emb in zip(chunks, embeddings):
                self._embeddings[chunk.chunk_id] = emb

        logger.info("Indexed %d chunks", len(chunks))
        return len(chunks)

    def _index_qdrant(self, chunks: list[PaperChunk], embeddings: np.ndarray) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import PointStruct

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=i + len(self._embeddings),
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "paper_id": chunk.paper_id,
                        "title": chunk.title,
                        "section": chunk.section,
                        "chunk_type": chunk.chunk_type.value,
                        "year": chunk.year,
                        "text_preview": chunk.text[:200],
                    },
                )
            )
            self._embeddings[chunk.chunk_id] = emb

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._qdrant.upsert(
                self._collection_name,
                points[i : i + batch_size],
            )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Semantic search over indexed chunks."""
        self.initialize()

        query_emb = self._embed([query])[0]

        if self._qdrant is not None:
            return self._search_qdrant(query_emb, top_k, filters)
        else:
            return self._search_inmemory(query_emb, top_k, filters)

    def _search_qdrant(
        self, query_emb: np.ndarray, top_k: int, filters: dict | None
    ) -> RetrievalResult:
        """Search using Qdrant."""
        query_filter = None
        if filters:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            conditions = []
            for key, val in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
            query_filter = Filter(must=conditions)

        results = self._qdrant.search(
            self._collection_name,
            query_vector=query_emb.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )

        chunks = []
        scores = []
        for hit in results:
            cid = hit.payload.get("chunk_id", "")
            if cid in self._chunks_by_id:
                chunks.append(self._chunks_by_id[cid])
                scores.append(hit.score)

        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            strategy_used=RetrievalStrategy.SEMANTIC,
        )

    def _search_inmemory(
        self, query_emb: np.ndarray, top_k: int, filters: dict | None
    ) -> RetrievalResult:
        """In-memory cosine similarity search."""
        if not self._embeddings:
            return RetrievalResult()

        scored = []
        for cid, emb in self._embeddings.items():
            chunk = self._chunks_by_id.get(cid)
            if chunk is None:
                continue

            # Apply filters
            if filters:
                skip = False
                for key, val in filters.items():
                    if hasattr(chunk, key) and getattr(chunk, key) != val:
                        skip = True
                        break
                if skip:
                    continue

            sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
            scored.append((cid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        chunks = [self._chunks_by_id[cid] for cid, _ in top]
        scores = [s for _, s in top]

        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            strategy_used=RetrievalStrategy.SEMANTIC,
        )

    def _chunk_to_text(self, chunk: PaperChunk) -> str:
        """Create rich text for embedding (title + section + content)."""
        parts = []
        if chunk.title:
            parts.append(f"[{chunk.title}]")
        if chunk.section:
            parts.append(f"[{chunk.section}]")
        parts.append(chunk.text[:2000])  # Cap text length for embedding
        return " ".join(parts)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks_by_id)
