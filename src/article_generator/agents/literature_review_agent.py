"""Literature Review Agent — generates a structured, citation-grounded literature review.

Implements §6.5 of plan_doc_v2.md:
- Taxonomy-driven organization
- Dense citation with provenance tracking
- Gap identification
- Narrative coherence
"""

from __future__ import annotations

import logging
from typing import Any

from article_generator.agents.base import BaseAgent
from article_generator.models import LiteratureReviewOutput, RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a world-class AI researcher writing the Related Work / Literature Review \
section of a top-tier conference paper. Your writing must be:

1. **Taxonomically organized** — group papers by approach category, not chronologically
2. **Citation-dense** — every factual claim references a specific paper [AuthorYear]
3. **Critical** — highlight limitations of each approach
4. **Gap-revealing** — build toward the gap that motivates the current paper
5. **Concise** — conference-quality, not survey-level verbosity

Output ONLY valid JSON:
{
  "narrative": "The full literature review text with [AuthorYear] citations...",
  "taxonomy": {
    "Category A": ["paper_id_1", "paper_id_2"],
    "Category B": ["paper_id_3"]
  },
  "citations_used": ["paper_id_1", "paper_id_2", ...],
  "gaps_identified": ["Gap 1", "Gap 2", ...],
  "confidence": 0.0 to 1.0
}

Writing rules:
- Use [Author et al., Year] citation format
- Open with a broad framing paragraph
- Each category gets a paragraph with 3-5 paper discussions
- End with a gap synthesis paragraph that motivates the proposed work
- Do NOT hallucinate citations — only cite papers from the retrieved context
"""


class LiteratureReviewAgent(BaseAgent):
    agent_id = "literature_review"
    system_prompt = SYSTEM_PROMPT

    def execute(self, context: str, **kwargs: Any) -> LiteratureReviewOutput:
        title: str = kwargs["title"]
        contributions: str = kwargs.get("contributions_summary", "")
        retrieval_result: RetrievalResult = kwargs.get("retrieval_result", RetrievalResult())

        lit_context = self._format_papers(retrieval_result)

        prompt = (
            f"Paper Title: {title}\n\n"
            f"Our Contributions (to position the review):\n{contributions}\n\n"
            f"Retrieved Literature:\n{lit_context}\n\n"
            f"Blackboard Context:\n{context}\n\n"
            f"Write the Related Work section now."
        )

        response = self.call_llm(
            prompt,
            temperature=0.4,
            max_tokens=4096,
        )

        return self.parse_json_response(response, LiteratureReviewOutput)

    def _format_papers(self, retrieval_result: RetrievalResult) -> str:
        parts = []
        seen_papers = set()
        for chunk in retrieval_result.chunks:
            if chunk.paper_id in seen_papers:
                continue
            seen_papers.add(chunk.paper_id)
            authors_str = ", ".join(chunk.authors[:3])
            if len(chunk.authors) > 3:
                authors_str += " et al."
            parts.append(
                f"Paper: {chunk.title} ({authors_str}, {chunk.year})\n"
                f"Venue: {chunk.venue}\n"
                f"Key content [{chunk.section}]: {chunk.text[:500]}\n"
            )
        return "\n---\n".join(parts[:20]) if parts else "No papers retrieved."

    def _summarize(self, result: Any) -> str:
        if isinstance(result, LiteratureReviewOutput):
            return (
                f"Literature review: {len(result.citations_used)} citations used\n"
                f"Categories: {list(result.taxonomy.keys())}\n"
                f"Gaps: {len(result.gaps_identified)}\n"
                f"Confidence: {result.confidence:.2f}"
            )
        return str(result)[:500]
