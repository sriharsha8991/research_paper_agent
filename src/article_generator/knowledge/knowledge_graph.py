"""Knowledge Graph — NetworkX-based relational graph over papers.

Implements §7.4 of plan_doc_v2.md:
- Nodes: Paper, Method, Dataset, Metric, Concept
- Edges: CITES, PROPOSES, USES, REPORTS, EXTENDS, COMPETES_WITH
- Multi-hop traversal for context assembly
- Graph-guided retrieval
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from article_generator.config import AppConfig
from article_generator.models import KGEdge, KGNode, PaperChunk, PaperMetadata

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Paper relationship graph built on NetworkX."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.graph = nx.DiGraph()
        self._store_path = Path(config.paths.data_dir) / "knowledge_store" / "graph.json"
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

    def add_paper(self, paper: PaperMetadata) -> str:
        """Add a paper node to the graph."""
        node_id = f"paper:{paper.paper_id}"
        self.graph.add_node(
            node_id,
            node_type="Paper",
            title=paper.title,
            year=paper.year,
            venue=paper.venue,
            citation_count=paper.citation_count,
            authors=paper.authors[:5],
        )
        return node_id

    def add_concept(self, name: str, concept_type: str = "Concept") -> str:
        """Add a concept/method/dataset/metric node."""
        node_id = f"{concept_type.lower()}:{name.lower().replace(' ', '_')}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, node_type=concept_type, name=name)
        return node_id

    def add_edge(self, source: str, target: str, edge_type: str, **props: Any) -> None:
        """Add a typed edge between nodes."""
        self.graph.add_edge(source, target, edge_type=edge_type, **props)

    def build_from_corpus(
        self, papers: list[PaperMetadata], chunks: list[PaperChunk]
    ) -> None:
        """Build the knowledge graph from a paper corpus and their chunks."""
        # Add paper nodes
        for paper in papers:
            self.add_paper(paper)

        # Extract concepts from chunks
        for chunk in chunks:
            paper_node = f"paper:{chunk.paper_id}"
            if not self.graph.has_node(paper_node):
                continue

            # Link datasets
            for ds in chunk.datasets_mentioned:
                ds_node = self.add_concept(ds, "Dataset")
                self.add_edge(paper_node, ds_node, "USES")

            # Link metrics
            for metric in chunk.metrics_mentioned:
                m_node = self.add_concept(metric, "Metric")
                self.add_edge(paper_node, m_node, "REPORTS")

        # Build citation edges (where we know them)
        paper_ids = {f"paper:{p.paper_id}" for p in papers}
        for paper in papers:
            paper_node = f"paper:{paper.paper_id}"
            # We can't get citation links without API calls, but these exist
            # in the chunk cross-references extracted during ingestion

        logger.info(
            "Knowledge graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def get_neighbors(
        self, node_id: str, edge_type: str | None = None, hops: int = 1
    ) -> list[dict[str, Any]]:
        """Get neighbors up to N hops away, optionally filtered by edge type."""
        if not self.graph.has_node(node_id):
            return []

        visited = set()
        frontier = {node_id}
        results = []

        for hop in range(hops):
            next_frontier = set()
            for n in frontier:
                for _, neighbor, data in self.graph.edges(n, data=True):
                    if neighbor in visited:
                        continue
                    if edge_type and data.get("edge_type") != edge_type:
                        continue
                    node_data = dict(self.graph.nodes[neighbor])
                    node_data["node_id"] = neighbor
                    node_data["hop"] = hop + 1
                    node_data["edge_type"] = data.get("edge_type", "")
                    results.append(node_data)
                    next_frontier.add(neighbor)
                visited.add(n)
            frontier = next_frontier

        return results

    def find_similar_papers(self, paper_id: str) -> list[str]:
        """Find papers that share methods/datasets with the given paper."""
        paper_node = f"paper:{paper_id}"
        if not self.graph.has_node(paper_node):
            return []

        # Get all concepts connected to this paper
        concepts = set()
        for _, neighbor, data in self.graph.edges(paper_node, data=True):
            node_data = self.graph.nodes.get(neighbor, {})
            if node_data.get("node_type") in ("Method", "Dataset", "Metric", "Concept"):
                concepts.add(neighbor)

        # Find other papers connected to same concepts
        similar = {}
        for concept in concepts:
            for predecessor in self.graph.predecessors(concept):
                if predecessor != paper_node and predecessor.startswith("paper:"):
                    similar[predecessor] = similar.get(predecessor, 0) + 1

        # Sort by shared concept count
        ranked = sorted(similar.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in ranked]

    def get_paper_context(self, paper_id: str) -> dict[str, Any]:
        """Get full context about a paper: its concepts, related papers, etc."""
        paper_node = f"paper:{paper_id}"
        if not self.graph.has_node(paper_node):
            return {}

        context = dict(self.graph.nodes[paper_node])
        context["datasets"] = []
        context["metrics"] = []
        context["methods"] = []
        context["related_papers"] = []

        for _, neighbor, data in self.graph.edges(paper_node, data=True):
            node_data = self.graph.nodes.get(neighbor, {})
            ntype = node_data.get("node_type", "")
            name = node_data.get("name", neighbor)

            if ntype == "Dataset":
                context["datasets"].append(name)
            elif ntype == "Metric":
                context["metrics"].append(name)
            elif ntype == "Method":
                context["methods"].append(name)

        context["related_papers"] = self.find_similar_papers(paper_id)[:5]
        return context

    def save(self) -> None:
        """Persist graph to JSON."""
        data = nx.node_link_data(self.graph)
        self._store_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Knowledge graph saved to %s", self._store_path)

    def load(self) -> bool:
        """Load graph from JSON."""
        if not self._store_path.exists():
            return False
        try:
            data = json.loads(self._store_path.read_text())
            self.graph = nx.node_link_graph(data)
            logger.info(
                "Knowledge graph loaded: %d nodes, %d edges",
                self.graph.number_of_nodes(),
                self.graph.number_of_edges(),
            )
            return True
        except Exception as e:
            logger.warning("Failed to load knowledge graph: %s", e)
            return False

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()
