"""Novelty & Confidence Scoring — assesses originality and hallucination risk.

Implements §8.3-8.5 of plan_doc_v2.md:
- Textual novelty (n-gram overlap)
- Structural novelty (section structure comparison)
- Contribution novelty (claim uniqueness)
- Per-section confidence with hallucination risk tagging
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

from article_generator.models import (
    ConfidenceReport,
    NoveltyReport,
    PaperChunk,
    ProvenanceRecord,
    RetrievalResult,
    SectionConfidence,
)

logger = logging.getLogger(__name__)


class NoveltyScorer:
    """Computes novelty scores for generated text against source corpus."""

    def score(
        self,
        generated_sections: dict[str, str],
        source_chunks: list[PaperChunk],
    ) -> NoveltyReport:
        """Compute comprehensive novelty report."""
        all_gen_text = " ".join(generated_sections.values())
        all_source_text = " ".join(c.text for c in source_chunks)

        textual = self._textual_novelty(all_gen_text, all_source_text)
        structural = self._structural_novelty(generated_sections)
        contribution = self._contribution_novelty(all_gen_text, all_source_text)

        overall = 0.4 * textual + 0.3 * structural + 0.3 * contribution

        overlap_regions = self._find_overlaps(all_gen_text, all_source_text)

        return NoveltyReport(
            textual_novelty=round(textual, 3),
            structural_novelty=round(structural, 3),
            contribution_novelty=round(contribution, 3),
            overall=round(overall, 3),
            overlap_regions=overlap_regions,
        )

    def _textual_novelty(self, generated: str, source: str, n: int = 4) -> float:
        """Compute n-gram novelty: 1 - (overlap / total generated n-grams)."""
        gen_ngrams = self._get_ngrams(generated, n)
        src_ngrams = self._get_ngrams(source, n)

        if not gen_ngrams:
            return 1.0

        overlap = gen_ngrams & src_ngrams
        novelty = 1.0 - (sum(overlap.values()) / sum(gen_ngrams.values()))
        return max(0.0, min(1.0, novelty))

    def _structural_novelty(self, sections: dict[str, str]) -> float:
        """Score how novel the document structure is.

        Expected structure earns a baseline; unusual sections boost novelty.
        """
        standard_sections = {
            "abstract", "introduction", "related_work", "method",
            "experiments", "results", "discussion", "conclusion",
        }
        gen_sections = set(sections.keys())
        overlap = gen_sections & standard_sections
        novel_sections = gen_sections - standard_sections

        # Standard structure gets 0.5, each novel section adds up to 0.5
        base = 0.5
        bonus = min(0.5, len(novel_sections) * 0.1)
        return base + bonus

    def _contribution_novelty(self, generated: str, source: str) -> float:
        """Estimate contribution novelty via unique phrases."""
        gen_phrases = self._get_ngrams(generated, 6)
        src_phrases = self._get_ngrams(source, 6)

        if not gen_phrases:
            return 1.0

        unique_count = sum(1 for p in gen_phrases if p not in src_phrases)
        return min(1.0, unique_count / max(len(gen_phrases), 1))

    def _get_ngrams(self, text: str, n: int) -> Counter:
        """Extract n-grams from text."""
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i : i + n])
            ngrams.append(ngram)
        return Counter(ngrams)

    def _find_overlaps(
        self, generated: str, source: str, n: int = 8
    ) -> list[dict[str, Any]]:
        """Find regions of high overlap (potential plagiarism)."""
        gen_ngrams = self._get_ngrams(generated, n)
        src_ngrams = self._get_ngrams(source, n)

        overlaps = []
        common = gen_ngrams & src_ngrams
        for ngram, count in common.most_common(5):
            overlaps.append({
                "ngram": " ".join(ngram),
                "count": count,
                "severity": "high" if n >= 8 else "medium",
            })

        return overlaps


class ConfidenceScorer:
    """Compute per-section confidence and overall hallucination risk."""

    def score(
        self,
        sections: dict[str, str],
        provenance_records: list[ProvenanceRecord],
        verification_confidence: float = 1.0,
    ) -> ConfidenceReport:
        """Generate confidence report across all sections."""
        section_scores = {}

        for section_name, content in sections.items():
            # Get provenance records for this section
            section_provenance = [
                p for p in provenance_records
                if any(p.text[:50] in content for _ in [1])  # Approx match
            ]

            sc = self._score_section(
                section_name, content, section_provenance, verification_confidence
            )
            section_scores[section_name] = sc

        # Overall confidence
        if section_scores:
            overall = sum(s.confidence for s in section_scores.values()) / len(section_scores)
        else:
            overall = 0.5

        # Hallucination risk
        if overall >= 0.8:
            risk = "low"
        elif overall >= 0.5:
            risk = "medium"
        else:
            risk = "high"

        recommendation = self._generate_recommendation(section_scores, risk)

        return ConfidenceReport(
            section_scores=section_scores,
            overall_confidence=round(overall, 3),
            hallucination_risk=risk,
            recommendation=recommendation,
        )

    def _score_section(
        self,
        name: str,
        content: str,
        provenance: list[ProvenanceRecord],
        cove_score: float,
    ) -> SectionConfidence:
        """Score a single section."""
        if not provenance:
            # Sections without provenance data get moderate confidence
            return SectionConfidence(
                confidence=0.5,
                source_coverage=0.0,
                cove_score=cove_score,
                note="No provenance data available",
            )

        # Source coverage: what fraction of sentences have source backing
        sourced = sum(1 for p in provenance if p.provenance_type.value != "original")
        source_coverage = sourced / max(len(provenance), 1)

        # Average confidence from provenance
        avg_conf = sum(p.confidence for p in provenance) / max(len(provenance), 1)

        # Combined confidence
        confidence = 0.4 * avg_conf + 0.3 * source_coverage + 0.3 * cove_score

        return SectionConfidence(
            confidence=round(confidence, 3),
            source_coverage=round(source_coverage, 3),
            cove_score=round(cove_score, 3),
        )

    def _generate_recommendation(
        self, scores: dict[str, SectionConfidence], risk: str
    ) -> str:
        """Generate actionable recommendation."""
        low_conf = [name for name, sc in scores.items() if sc.confidence < 0.5]

        if risk == "low":
            return "Paper confidence is high. Review flagged overlap regions if any."
        elif risk == "medium":
            sections_str = ", ".join(low_conf[:3])
            return f"Medium risk. Review these sections for accuracy: {sections_str}"
        else:
            sections_str = ", ".join(low_conf[:5])
            return f"HIGH HALLUCINATION RISK. Sections needing review: {sections_str}"
