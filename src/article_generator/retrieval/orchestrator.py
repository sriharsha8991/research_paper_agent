"""Retrieval Orchestrator — CRAG-style adaptive retrieval with hybrid search.

Implements §8 of plan_doc_v2.md:
- Strategy selection (semantic, hybrid, KG-guided, multi-hop)
- Corrective RAG: assess → correct → re-retrieve if low confidence
- Query decomposition for complex questions
- Reciprocal Rank Fusion for combining results
"""

from __future__ import annotations

import logging
from typing import Any

from article_generator.config import AppConfig
from article_generator.knowledge.knowledge_graph import KnowledgeGraph
from article_generator.knowledge.vector_store import VectorStore
from article_generator.llm_client import LLMClient
from article_generator.models import (
    PaperChunk,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
)

logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    """Orchestrates retrieval across vector store and knowledge graph."""

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        llm: LLMClient,
        config: AppConfig,
    ):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.llm = llm
        self.config = config

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Main retrieval entry point with strategy routing."""
        strategy = query.strategy

        if strategy == RetrievalStrategy.SEMANTIC:
            result = self._semantic_search(query)
        elif strategy == RetrievalStrategy.HYBRID:
            result = self._hybrid_search(query)
        elif strategy == RetrievalStrategy.KG_GUIDED:
            result = self._kg_guided_search(query)
        elif strategy == RetrievalStrategy.MULTI_HOP:
            result = self._multi_hop_search(query)
        else:
            result = self._hybrid_search(query)

        # CRAG: Assess retrieval quality
        result = self._crag_assess(query, result)

        return result

    def retrieve_for_section(
        self, section: str, queries: list[str], top_k: int = 15
    ) -> RetrievalResult:
        """Retrieve chunks relevant to a specific paper section."""
        # Map sections to appropriate strategies
        strategy_map = {
            "literature_review": RetrievalStrategy.HYBRID,
            "method": RetrievalStrategy.KG_GUIDED,
            "math": RetrievalStrategy.SEMANTIC,
            "experiment": RetrievalStrategy.HYBRID,
            "results": RetrievalStrategy.HYBRID,
            "discussion": RetrievalStrategy.MULTI_HOP,
        }
        strategy = strategy_map.get(section, RetrievalStrategy.HYBRID)

        all_chunks: list[PaperChunk] = []
        all_scores: list[float] = []

        for query_text in queries:
            rq = RetrievalQuery(
                query=query_text,
                top_k=top_k,
                strategy=strategy,
                filters={"section": section} if strategy != RetrievalStrategy.MULTI_HOP else {},
            )
            result = self.retrieve(rq)
            all_chunks.extend(result.chunks)
            all_scores.extend(result.scores)

        # Deduplicate and re-rank
        seen = set()
        unique_chunks = []
        unique_scores = []
        for chunk, score in sorted(zip(all_chunks, all_scores), key=lambda x: x[1], reverse=True):
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique_chunks.append(chunk)
                unique_scores.append(score)

        return RetrievalResult(
            chunks=unique_chunks[:top_k],
            scores=unique_scores[:top_k],
            strategy_used=strategy,
            confidence=self._compute_confidence(unique_scores[:top_k]),
        )

    def _semantic_search(self, query: RetrievalQuery) -> RetrievalResult:
        """Pure semantic vector search."""
        return self.vector_store.search(
            query.query,
            top_k=query.top_k,
            filters=query.filters if query.filters else None,
        )

    def _hybrid_search(self, query: RetrievalQuery) -> RetrievalResult:
        """Combine semantic search with keyword matching."""
        # Get semantic results
        semantic = self.vector_store.search(query.query, top_k=query.top_k * 2)

        # Simple keyword boosting on top of semantic results
        keywords = set(query.query.lower().split())
        boosted_chunks = []
        boosted_scores = []

        for chunk, score in zip(semantic.chunks, semantic.scores):
            text_lower = chunk.text.lower()
            keyword_hits = sum(1 for kw in keywords if kw in text_lower and len(kw) > 3)
            boost = min(0.2, keyword_hits * 0.05)
            boosted_chunks.append(chunk)
            boosted_scores.append(score + boost)

        # Re-sort
        pairs = sorted(zip(boosted_chunks, boosted_scores), key=lambda x: x[1], reverse=True)
        chunks = [p[0] for p in pairs[:query.top_k]]
        scores = [p[1] for p in pairs[:query.top_k]]

        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            strategy_used=RetrievalStrategy.HYBRID,
            confidence=self._compute_confidence(scores),
        )

    def _kg_guided_search(self, query: RetrievalQuery) -> RetrievalResult:
        """Use knowledge graph to guide vector search."""
        # First do semantic search
        semantic = self.vector_store.search(query.query, top_k=query.top_k)

        if not semantic.chunks:
            return semantic

        # Get paper IDs from top results
        top_paper_ids = list({c.paper_id for c in semantic.chunks[:5]})

        # Find related papers via KG
        kg_paper_ids = set()
        for pid in top_paper_ids:
            similar = self.kg.find_similar_papers(pid)
            kg_paper_ids.update(similar[:3])

        # Search for content from KG-recommended papers
        kg_chunks: list[PaperChunk] = []
        kg_scores: list[float] = []
        if kg_paper_ids:
            for pid in list(kg_paper_ids)[:5]:
                # Search within this paper's chunks
                result = self.vector_store.search(
                    query.query,
                    top_k=3,
                    filters={"paper_id": pid.replace("paper:", "")},
                )
                kg_chunks.extend(result.chunks)
                kg_scores.extend(result.scores)

        # Merge with RRF
        all_chunks = semantic.chunks + kg_chunks
        all_scores = semantic.scores + [s * 0.8 for s in kg_scores]

        # Deduplicate
        seen = set()
        final_chunks = []
        final_scores = []
        for c, s in sorted(zip(all_chunks, all_scores), key=lambda x: x[1], reverse=True):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                final_chunks.append(c)
                final_scores.append(s)

        return RetrievalResult(
            chunks=final_chunks[:query.top_k],
            scores=final_scores[:query.top_k],
            strategy_used=RetrievalStrategy.KG_GUIDED,
            confidence=self._compute_confidence(final_scores[:query.top_k]),
        )

    def _multi_hop_search(self, query: RetrievalQuery) -> RetrievalResult:
        """Multi-hop retrieval — decompose query, retrieve for each hop.

        Step 1: Decompose query into sub-questions
        Step 2: Retrieve for each sub-question
        Step 3: Merge results
        """
        sub_queries = self._decompose_query(query.query)

        all_chunks: list[PaperChunk] = []
        all_scores: list[float] = []

        for sq in sub_queries:
            result = self.vector_store.search(sq, top_k=query.top_k // len(sub_queries) + 1)
            all_chunks.extend(result.chunks)
            all_scores.extend(result.scores)

        # Deduplicate and rank
        seen = set()
        final_chunks = []
        final_scores = []
        for c, s in sorted(zip(all_chunks, all_scores), key=lambda x: x[1], reverse=True):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                final_chunks.append(c)
                final_scores.append(s)

        return RetrievalResult(
            chunks=final_chunks[:query.top_k],
            scores=final_scores[:query.top_k],
            strategy_used=RetrievalStrategy.MULTI_HOP,
            confidence=self._compute_confidence(final_scores[:query.top_k]),
        )

    def _decompose_query(self, query: str) -> list[str]:
        """Decompose a complex query into sub-queries using LLM."""
        try:
            response = self.llm.generate_fast(
                f"Decompose this research question into 2-3 simpler sub-questions.\n"
                f"Return ONLY a JSON array of strings.\n\n"
                f"Question: {query}\n\n"
                f'Example output: ["What is X?", "How does Y relate to Z?"]',
                max_tokens=300,
            )
            import json
            # Extract JSON array
            text = response.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            logger.debug("Query decomposition failed: %s", e)

        # Fallback: just return original
        return [query]

    def _crag_assess(self, query: RetrievalQuery, result: RetrievalResult) -> RetrievalResult:
        """Corrective RAG — assess and potentially re-retrieve.

        If top results have low scores, try alternative strategies.
        """
        if not result.scores:
            result.confidence = "low"
            return result

        avg_score = sum(result.scores) / len(result.scores)
        top_score = result.scores[0] if result.scores else 0

        if top_score >= 0.75:
            result.confidence = "high"
        elif top_score >= 0.5:
            result.confidence = "medium"
        elif result.strategy_used != RetrievalStrategy.HYBRID:
            # Re-try with hybrid if semantic didn't work well
            logger.info("CRAG: Low confidence (%.2f), re-retrieving with hybrid", top_score)
            fallback = self._hybrid_search(query)
            if fallback.scores and fallback.scores[0] > top_score:
                fallback.confidence = "medium"
                return fallback
            result.confidence = "low"
        else:
            result.confidence = "low"

        return result

    def _compute_confidence(self, scores: list[float]) -> str:
        """Compute confidence level from scores."""
        if not scores:
            return "low"
        avg = sum(scores) / len(scores)
        if avg >= 0.7:
            return "high"
        if avg >= 0.4:
            return "medium"
        return "low"
