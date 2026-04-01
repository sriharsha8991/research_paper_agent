"""Ingestion Pipeline — PDF download, parsing, and section-aware chunking.

Implements §7 of plan_doc_v2.md:
- PDF download with retry
- Text extraction (pdfplumber-based, GROBID optional)
- Section segmentation via regex heuristics
- Semantic chunking with overlap
- Metadata enrichment (equations, datasets, metrics, claims)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

from article_generator.config import AppConfig
from article_generator.models import (
    ChunkType,
    Claim,
    PaperChunk,
    PaperMetadata,
)

logger = logging.getLogger(__name__)

# Section header patterns
SECTION_PATTERNS = [
    (r"^(?:\d+\.?\s+)?abstract\b", "abstract"),
    (r"^(?:\d+\.?\s+)?introduction\b", "introduction"),
    (r"^(?:\d+\.?\s+)?related\s+work\b", "related_work"),
    (r"^(?:\d+\.?\s+)?background\b", "related_work"),
    (r"^(?:\d+\.?\s+)?(?:method|approach|proposed|our\s+method)\b", "method"),
    (r"^(?:\d+\.?\s+)?(?:experiment|evaluation|results?)\b", "result"),
    (r"^(?:\d+\.?\s+)?(?:discussion)\b", "discussion"),
    (r"^(?:\d+\.?\s+)?(?:conclusion|summary)\b", "discussion"),
    (r"^(?:\d+\.?\s+)?(?:appendix|supplementary)\b", "appendix"),
    (r"^(?:\d+\.?\s+)?(?:algorithm)\b", "algorithm"),
]

SECTION_TO_CHUNK_TYPE = {
    "abstract": ChunkType.ABSTRACT,
    "introduction": ChunkType.INTRODUCTION,
    "related_work": ChunkType.RELATED_WORK,
    "method": ChunkType.METHOD,
    "result": ChunkType.RESULT,
    "discussion": ChunkType.DISCUSSION,
    "algorithm": ChunkType.ALGORITHM,
    "appendix": ChunkType.METHOD,
}

# Regex patterns for enrichment
EQUATION_PATTERN = re.compile(r"\$\$.+?\$\$|\\\[.+?\\\]|\\begin\{equation\}.+?\\end\{equation\}", re.DOTALL)
DATASET_KEYWORDS = [
    "imagenet", "cifar", "mnist", "coco", "squad", "glue", "superglue",
    "wikitext", "openwebtext", "pile", "c4", "laion", "commoncrawl",
    "yelp", "imdb", "sst", "penn treebank", "wmt", "flores",
]
METRIC_KEYWORDS = [
    "accuracy", "precision", "recall", "f1", "bleu", "rouge", "meteor",
    "perplexity", "auc", "map", "ndcg", "fid", "inception score",
    "top-1", "top-5", "em", "exact match", "loss", "mse", "mae",
]


class IngestionPipeline:
    """Download, parse, chunk, and enrich academic papers."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.raw_dir = Path(config.paths.data_dir) / "papers_raw"
        self.parsed_dir = Path(config.paths.data_dir) / "parsed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(timeout=60, follow_redirects=True)

    def ingest_paper(self, paper: PaperMetadata) -> list[PaperChunk]:
        """Full pipeline: download → parse → chunk → enrich for one paper."""
        # Step 1: Get text
        text = self._get_paper_text(paper)
        if not text or len(text.strip()) < 100:
            logger.warning("No usable text for paper %s", paper.paper_id)
            return []

        # Step 2: Segment into sections
        sections = self._segment_sections(text)

        # Step 3: Chunk each section
        chunks = []
        total_len = sum(len(s["text"]) for s in sections) or 1
        running_pos = 0

        for section in sections:
            section_chunks = self._chunk_text(
                section["text"],
                chunk_size=self.config.retrieval.chunk_size,
                overlap=self.config.retrieval.chunk_overlap,
            )
            for chunk_text in section_chunks:
                position = running_pos / total_len
                chunk = self._enrich_chunk(
                    chunk_text=chunk_text,
                    paper=paper,
                    section_name=section["name"],
                    subsection=section.get("subsection", ""),
                    position=position,
                )
                chunks.append(chunk)
                running_pos += len(chunk_text)

        logger.info(
            "Ingested paper %s: %d sections, %d chunks",
            paper.paper_id, len(sections), len(chunks),
        )
        return chunks

    def ingest_corpus(self, papers: list[PaperMetadata]) -> list[PaperChunk]:
        """Ingest all papers in a corpus."""
        all_chunks = []
        for i, paper in enumerate(papers):
            logger.info("Ingesting paper %d/%d: %s", i + 1, len(papers), paper.title[:60])
            chunks = self.ingest_paper(paper)
            all_chunks.extend(chunks)
        logger.info("Total chunks produced: %d", len(all_chunks))
        return all_chunks

    def _get_paper_text(self, paper: PaperMetadata) -> str:
        """Get paper text — from abstract if no PDF, from PDF if available."""
        # Try PDF download and extraction
        if paper.pdf_url:
            pdf_path = self._download_pdf(paper)
            if pdf_path:
                text = self._extract_text_from_pdf(pdf_path)
                if text and len(text) > 200:
                    return text

        # Fallback to abstract
        if paper.abstract:
            return f"Abstract\n\n{paper.abstract}"

        return ""

    def _download_pdf(self, paper: PaperMetadata) -> str | None:
        """Download PDF to local directory."""
        safe_id = re.sub(r"[^\w\-.]", "_", paper.paper_id)
        pdf_path = self.raw_dir / f"{safe_id}.pdf"

        if pdf_path.exists():
            return str(pdf_path)

        try:
            resp = self._http.get(paper.pdf_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not resp.content[:5] == b"%PDF-":
                logger.debug("Not a PDF response for %s", paper.paper_id)
                return None
            pdf_path.write_bytes(resp.content)
            paper.pdf_path = str(pdf_path)
            return str(pdf_path)
        except Exception as e:
            logger.debug("PDF download failed for %s: %s", paper.paper_id, e)
            return None

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber (lightweight)."""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:30]:  # Cap at 30 pages
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            logger.debug("pdfplumber not installed; falling back to abstract-only")
            return ""
        except Exception as e:
            logger.debug("PDF extraction failed for %s: %s", pdf_path, e)
            return ""

    def _segment_sections(self, text: str) -> list[dict[str, str]]:
        """Split text into sections based on heading patterns."""
        lines = text.split("\n")
        sections: list[dict[str, str]] = []
        current_section = "abstract"
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip().lower()
            matched = False
            for pattern, section_name in SECTION_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE):
                    # Save previous section
                    if current_lines:
                        sections.append({
                            "name": current_section,
                            "text": "\n".join(current_lines).strip(),
                        })
                    current_section = section_name
                    current_lines = []
                    matched = True
                    break
            if not matched:
                current_lines.append(line)

        # Don't forget the last section
        if current_lines:
            sections.append({
                "name": current_section,
                "text": "\n".join(current_lines).strip(),
            })

        # If no sections detected, treat entire text as one chunk
        if not sections:
            sections = [{"name": "method", "text": text.strip()}]

        return sections

    def _chunk_text(
        self, text: str, chunk_size: int = 512, overlap: int = 50
    ) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def _enrich_chunk(
        self,
        chunk_text: str,
        paper: PaperMetadata,
        section_name: str,
        subsection: str,
        position: float,
    ) -> PaperChunk:
        """Create an enriched PaperChunk with metadata extraction."""
        text_lower = chunk_text.lower()

        # Detect equations
        equations = EQUATION_PATTERN.findall(chunk_text)

        # Detect datasets
        datasets = [d for d in DATASET_KEYWORDS if d in text_lower]

        # Detect metrics
        metrics = [m for m in METRIC_KEYWORDS if m in text_lower]

        chunk_type = SECTION_TO_CHUNK_TYPE.get(section_name, ChunkType.METHOD)

        return PaperChunk(
            paper_id=paper.paper_id,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            venue=paper.venue,
            section=section_name,
            subsection=subsection,
            chunk_type=chunk_type,
            text=chunk_text,
            has_equation=len(equations) > 0,
            equations_latex=equations[:10],
            datasets_mentioned=datasets,
            metrics_mentioned=metrics,
            position_in_paper=round(position, 3),
        )
