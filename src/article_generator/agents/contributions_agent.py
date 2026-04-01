"""Contributions Agent — identifies and frames novel contributions.

Implements §6.4 of plan_doc_v2.md:
- Gap analysis from literature review
- Contribution framing (theoretical, empirical, hybrid)
- Novelty justification via ontology positioning
- Positioning statement generation
"""

from __future__ import annotations

import logging
from typing import Any

from article_generator.agents.base import BaseAgent
from article_generator.models import (
    ContributionClaim,
    ContributionsOutput,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior AI researcher tasked with identifying the novel contributions \
of a proposed research paper. You analyze the existing literature (provided as \
retrieved context) and determine what gaps exist and how the proposed work fills them.

You must output ONLY valid JSON:
{
  "contributions": [
    {
      "claim": "Clear statement of what this paper contributes",
      "contribution_type": "theoretical" | "empirical" | "hybrid",
      "novelty_justification": "Why this is novel vs existing work",
      "supporting_evidence_needed": "What experiments/proofs would support this claim",
      "ontology_position": "Where this fits in the method taxonomy",
      "confidence": 0.0 to 1.0
    }
  ],
  "positioning_statement": "A 2-3 sentence summary of how this work advances the field",
  "gap_analysis": ["Gap 1 description", "Gap 2 description", ...]
}

Rules:
1. Be specific — avoid vague claims like "improves performance"
2. Each contribution must be falsifiable and testable
3. novelty_justification must reference specific prior work limitations
4. confidence reflects how novel this claim truly is (1.0 = clearly novel, 0.3 = incremental)
5. Identify at least 3 contributions and 3 gaps
"""


class ContributionsAgent(BaseAgent):
    agent_id = "contributions"
    system_prompt = SYSTEM_PROMPT

    def execute(self, context: str, **kwargs: Any) -> ContributionsOutput:
        title: str = kwargs["title"]
        abstract: str = kwargs.get("abstract", "")
        retrieval_result: RetrievalResult = kwargs.get("retrieval_result", RetrievalResult())

        # Build literature context from retrieved chunks
        lit_context = self._format_literature(retrieval_result)

        prompt = (
            f"Paper Title: {title}\n"
            f"Abstract: {abstract}\n\n"
            f"Existing Literature (retrieved from top papers):\n{lit_context}\n\n"
            f"Blackboard Context:\n{context}\n\n"
            f"Analyze the gaps and identify the novel contributions of this paper."
        )

        response = self.call_llm(prompt, temperature=0.3, max_tokens=3000)
        return self.parse_json_response(response, ContributionsOutput)

    def _format_literature(self, retrieval_result: RetrievalResult) -> str:
        parts = []
        for i, chunk in enumerate(retrieval_result.chunks[:15]):
            parts.append(
                f"[{i+1}] ({chunk.title}, {chunk.year}) [{chunk.section}]:\n"
                f"{chunk.text[:400]}"
            )
        return "\n\n".join(parts) if parts else "No literature retrieved yet."

    def _summarize(self, result: Any) -> str:
        if isinstance(result, ContributionsOutput):
            claims = "; ".join(c.claim[:80] for c in result.contributions[:3])
            return (
                f"Contributions ({len(result.contributions)}): {claims}\n"
                f"Gaps: {len(result.gap_analysis)}\n"
                f"Positioning: {result.positioning_statement[:150]}"
            )
        return str(result)[:500]
