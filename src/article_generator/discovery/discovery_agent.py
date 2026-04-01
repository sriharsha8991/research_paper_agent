"""Paper Discovery Agent — multi-source academic paper retrieval.

Implements §6.2 of plan_doc_v2.md:
- Semantic Scholar API search
- arXiv API search
- Multi-phase: seed → expand → contrastive
- Deduplication and relevance ranking
- Coverage analysis (method / temporal / venue diversity)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from article_generator.agents.base import BaseAgent
from article_generator.config import AppConfig
from article_generator.context.memory import Blackboard
from article_generator.llm_client import LLMClient
from article_generator.models import (
    CoverageReport,
    DiscoveryOutput,
    PaperMetadata,
    PaperRole,
    RequirementOutput,
)

logger = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
S2_FIELDS = "paperId,title,authors,year,venue,abstract,citationCount,externalIds,isOpenAccess,openAccessPdf"


class DiscoveryAgent(BaseAgent):
    agent_id = "paper_discovery"
    system_prompt = "You are a paper discovery orchestrator."

    def __init__(self, llm: LLMClient, config: AppConfig, blackboard: Blackboard):
        super().__init__(llm, config, blackboard)
        self._http = httpx.Client(timeout=30, follow_redirects=True)
        self._seen_ids: set[str] = set()

    def execute(self, context: str, **kwargs: Any) -> DiscoveryOutput:
        req: RequirementOutput = kwargs["requirement"]

        all_papers: list[PaperMetadata] = []

        # Phase 1: Seed retrieval from positive queries
        for query in req.positive_queries:
            papers = self._search_semantic_scholar(query, limit=20)
            papers.extend(self._search_arxiv(query, limit=10))
            for p in papers:
                p.role = PaperRole.SEED
            all_papers.extend(self._deduplicate(papers))
            time.sleep(0.5)  # rate limiting

        # Phase 2: Citation expansion from top seeds
        seeds_sorted = sorted(all_papers, key=lambda p: p.citation_count, reverse=True)
        hub_papers = self._expand_citations(seeds_sorted[:10])
        for p in hub_papers:
            p.role = PaperRole.HUB
        all_papers.extend(self._deduplicate(hub_papers))

        # Phase 3: Recent papers (last 2 years)
        recent = self._search_recent(req)
        for p in recent:
            p.role = PaperRole.RECENT
        all_papers.extend(self._deduplicate(recent))

        # Phase 4: Contrastive (negative) queries for alternative approaches
        contrastive = self._search_contrastive(req)
        for p in contrastive:
            p.role = PaperRole.CONTRASTIVE
        all_papers.extend(self._deduplicate(contrastive))

        # Score and rank
        all_papers = self._rank_papers(all_papers, req)

        # Trim to target range
        max_papers = req.target_papers_max
        all_papers = all_papers[:max_papers]

        # Coverage analysis
        coverage = self._analyze_coverage(all_papers, req)

        return DiscoveryOutput(
            corpus=all_papers,
            total_papers=len(all_papers),
            coverage_report=coverage,
        )

    def _search_semantic_scholar(self, query: str, limit: int = 20) -> list[PaperMetadata]:
        """Search Semantic Scholar API."""
        papers = []
        try:
            resp = self._http.get(
                f"{S2_API_BASE}/paper/search",
                params={
                    "query": query,
                    "limit": min(limit, 100),
                    "fields": S2_FIELDS,
                },
            )
            if resp.status_code == 429:
                time.sleep(2)
                return papers
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", []):
                paper = self._s2_to_metadata(item)
                if paper:
                    papers.append(paper)
        except Exception as e:
            logger.warning("Semantic Scholar search failed for '%s': %s", query, e)
        return papers

    def _search_arxiv(self, query: str, limit: int = 10) -> list[PaperMetadata]:
        """Search arXiv API."""
        papers = []
        try:
            encoded_query = urllib.parse.quote(query)
            resp = self._http.get(
                ARXIV_API_BASE,
                params={
                    "search_query": f"all:{encoded_query}",
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            resp.raise_for_status()
            papers = self._parse_arxiv_xml(resp.text)
        except Exception as e:
            logger.warning("arXiv search failed for '%s': %s", query, e)
        return papers

    def _parse_arxiv_xml(self, xml_text: str) -> list[PaperMetadata]:
        """Parse arXiv Atom XML feed into PaperMetadata."""
        papers = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                arxiv_id_raw = entry.findtext("atom:id", "", ns)
                arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw
                title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
                authors = [
                    a.findtext("atom:name", "", ns)
                    for a in entry.findall("atom:author", ns)
                ]
                published = entry.findtext("atom:published", "", ns)
                year = int(published[:4]) if published else 0

                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")

                paper = PaperMetadata(
                    paper_id=f"arxiv:{arxiv_id}",
                    source="arxiv",
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    pdf_url=pdf_url,
                    arxiv_id=arxiv_id,
                )
                papers.append(paper)
        except ET.ParseError as e:
            logger.warning("Failed to parse arXiv XML: %s", e)
        return papers

    def _s2_to_metadata(self, item: dict) -> PaperMetadata | None:
        """Convert Semantic Scholar API item to PaperMetadata."""
        if not item.get("title"):
            return None
        ext = item.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv", "")
        doi = ext.get("DOI", "")
        pdf_url = ""
        oap = item.get("openAccessPdf")
        if oap and isinstance(oap, dict):
            pdf_url = oap.get("url", "")

        return PaperMetadata(
            paper_id=item.get("paperId", ""),
            source="semantic_scholar",
            title=item.get("title", ""),
            authors=[a.get("name", "") for a in (item.get("authors") or [])],
            year=item.get("year") or 0,
            venue=item.get("venue") or "",
            abstract=item.get("abstract") or "",
            citation_count=item.get("citationCount") or 0,
            pdf_url=pdf_url,
            arxiv_id=arxiv_id,
            doi=doi,
        )

    def _expand_citations(self, seed_papers: list[PaperMetadata]) -> list[PaperMetadata]:
        """Get papers cited by seed papers (1-hop expansion)."""
        expanded = []
        for paper in seed_papers[:5]:
            if not paper.paper_id or paper.source != "semantic_scholar":
                continue
            try:
                resp = self._http.get(
                    f"{S2_API_BASE}/paper/{paper.paper_id}/citations",
                    params={"fields": S2_FIELDS, "limit": 10},
                )
                if resp.status_code == 429:
                    time.sleep(2)
                    continue
                resp.raise_for_status()
                for item in resp.json().get("data", []):
                    citing = item.get("citingPaper", {})
                    meta = self._s2_to_metadata(citing)
                    if meta:
                        expanded.append(meta)
                time.sleep(0.5)
            except Exception as e:
                logger.warning("Citation expansion failed for %s: %s", paper.paper_id, e)
        return expanded

    def _search_recent(self, req: RequirementOutput) -> list[PaperMetadata]:
        """Find very recent papers (last 2 years)."""
        papers = []
        for query in req.positive_queries[:3]:
            try:
                resp = self._http.get(
                    f"{S2_API_BASE}/paper/search",
                    params={
                        "query": query,
                        "limit": 10,
                        "fields": S2_FIELDS,
                        "year": f"{req.time_horizon_end - 2}-{req.time_horizon_end}",
                    },
                )
                if resp.status_code == 429:
                    time.sleep(2)
                    continue
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("data", []):
                    meta = self._s2_to_metadata(item)
                    if meta:
                        papers.append(meta)
                time.sleep(0.5)
            except Exception as e:
                logger.warning("Recent search failed: %s", e)
        return papers

    def _search_contrastive(self, req: RequirementOutput) -> list[PaperMetadata]:
        """Search for competing/alternative approaches."""
        papers = []
        # Use secondary questions and negative queries
        queries = req.secondary_questions[:2]
        for query in queries:
            found = self._search_semantic_scholar(query, limit=10)
            papers.extend(found)
            time.sleep(0.5)
        return papers

    def _deduplicate(self, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        """Deduplicate by normalized title hash."""
        unique = []
        for p in papers:
            norm_title = re.sub(r"[^a-z0-9]", "", p.title.lower())
            tid = hashlib.md5(norm_title.encode()).hexdigest()  # noqa: S324
            if tid not in self._seen_ids:
                self._seen_ids.add(tid)
                unique.append(p)
        return unique

    def _rank_papers(
        self, papers: list[PaperMetadata], req: RequirementOutput
    ) -> list[PaperMetadata]:
        """Score and rank papers by relevance, recency, and citations."""
        for paper in papers:
            score = 0.0

            # Recency bonus (0-30 points)
            if paper.year >= req.time_horizon_end - 1:
                score += 30
            elif paper.year >= req.time_horizon_end - 3:
                score += 20
            elif paper.year >= req.time_horizon_start:
                score += 10

            # Citation bonus (0-25 points, log scale)
            import math
            if paper.citation_count > 0:
                score += min(25, 5 * math.log10(paper.citation_count + 1))

            # Role bonus
            role_bonus = {PaperRole.SEED: 10, PaperRole.HUB: 15, PaperRole.RECENT: 20, PaperRole.CONTRASTIVE: 5}
            score += role_bonus.get(paper.role, 0)

            # Title/abstract keyword overlap with queries
            text = (paper.title + " " + paper.abstract).lower()
            keyword_hits = sum(
                1 for q in req.positive_queries
                if any(w in text for w in q.lower().split() if len(w) > 3)
            )
            score += min(25, keyword_hits * 3)

            paper.relevance_score = score

        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers

    def _analyze_coverage(
        self, papers: list[PaperMetadata], req: RequirementOutput
    ) -> CoverageReport:
        """Analyze how well the corpus covers the research space."""
        if not papers:
            return CoverageReport()

        # Temporal coverage: proportion of years in range that are covered
        years = {p.year for p in papers if p.year > 0}
        expected_years = set(range(req.time_horizon_start, req.time_horizon_end + 1))
        temporal = len(years & expected_years) / max(len(expected_years), 1)

        # Venue diversity
        venues = {p.venue for p in papers if p.venue}
        venue_div = min(1.0, len(venues) / 5.0)

        # Method coverage: how many domain labels appear in at least one paper
        all_labels = set(req.domain_labels.primary + req.domain_labels.method + req.domain_labels.task)
        if all_labels:
            covered = 0
            for label in all_labels:
                label_lower = label.lower().replace("_", " ")
                if any(label_lower in (p.title + " " + p.abstract).lower() for p in papers):
                    covered += 1
            method_cov = covered / len(all_labels)
        else:
            method_cov = 0.5

        return CoverageReport(
            method_coverage=round(method_cov, 3),
            temporal_coverage=round(temporal, 3),
            venue_diversity=round(venue_div, 3),
        )

    def _summarize(self, result: Any) -> str:
        if isinstance(result, DiscoveryOutput):
            return (
                f"Discovered {result.total_papers} papers.\n"
                f"Method coverage: {result.coverage_report.method_coverage:.1%}\n"
                f"Temporal coverage: {result.coverage_report.temporal_coverage:.1%}\n"
                f"Venue diversity: {result.coverage_report.venue_diversity:.1%}"
            )
        return str(result)[:500]
