"""Requirement Gathering Agent — expands title + abstract into a full research specification.

Implements §6.1 of plan_doc_v2.md:
- Generates positive/negative search queries
- Classifies domain labels (primary, method, task)
- Determines contribution type
- Sets time horizon and paper count targets
"""

from __future__ import annotations

import logging
from typing import Any

from article_generator.agents.base import BaseAgent
from article_generator.models import (
    ContributionType,
    GenerationInput,
    RequirementOutput,
    VenueStyle,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior AI/ML research analyst. Your task is to analyze a proposed \
research paper title (and optional abstract/direction) and produce a comprehensive \
research specification that will guide an automated literature review and paper \
generation pipeline.

You must output ONLY valid JSON matching the schema below — no markdown, no commentary.

{
  "primary_question": "The central research question (one clear sentence)",
  "secondary_questions": ["Sub-question 1", "Sub-question 2", ...],
  "positive_queries": [
    "Semantic Scholar / arXiv search query 1",
    "query 2",  ... // at least 8 diverse queries
  ],
  "negative_queries": [
    "queries to EXCLUDE irrelevant results"
  ],
  "domain_labels": {
    "primary": ["machine_learning", "deep_learning", ...],
    "method": ["transformer", "attention", ...],
    "task": ["image_classification", "text_generation", ...]
  },
  "contribution_type": "theoretical" | "empirical" | "hybrid",
  "target_papers_min": 30,
  "target_papers_max": 80,
  "time_horizon_start": 2020,
  "time_horizon_end": 2026,
  "venue_style": "neurips"
}

Rules:
1. Generate at least 8 diverse positive queries covering: core method, related approaches, \
benchmark datasets, theoretical foundations, competing methods, and applications.
2. Include 2-3 negative queries to filter noise.
3. Domain labels should be specific enough for ontology matching.
4. Infer contribution_type from the title: "theoretical" if proving/analyzing, \
"empirical" if benchmarking/experimenting, "hybrid" if both.
5. Use venue_style to set the right formality level.
"""


class RequirementAgent(BaseAgent):
    agent_id = "requirement_gathering"
    system_prompt = SYSTEM_PROMPT

    def execute(self, context: str, **kwargs: Any) -> RequirementOutput:
        gen_input: GenerationInput = kwargs["generation_input"]

        user_prompt = self._build_prompt(gen_input, context)

        response = self.call_llm(
            user_prompt,
            temperature=0.2,
            max_tokens=2048,
        )

        result = self.parse_json_response(response, RequirementOutput)

        # Apply overrides from user constraints
        if gen_input.constraints.time_horizon_start:
            result.time_horizon_start = gen_input.constraints.time_horizon_start
        if gen_input.constraints.time_horizon_end:
            result.time_horizon_end = gen_input.constraints.time_horizon_end
        result.venue_style = gen_input.target_venue

        # Ensure minimums
        if len(result.positive_queries) < 5:
            result.positive_queries.extend(self._fallback_queries(gen_input))

        return result

    def _build_prompt(self, gen_input: GenerationInput, context: str) -> str:
        parts = [f"Research Paper Title: {gen_input.title}"]

        if gen_input.abstract:
            parts.append(f"\nDraft Abstract:\n{gen_input.abstract}")
        if gen_input.research_direction:
            parts.append(f"\nResearch Direction:\n{gen_input.research_direction}")

        parts.append(f"\nTarget Venue: {gen_input.target_venue.value}")

        if gen_input.constraints.must_cite:
            parts.append(f"\nMust-cite papers: {', '.join(gen_input.constraints.must_cite)}")
        if gen_input.constraints.exclude_methods:
            parts.append(f"\nExclude methods: {', '.join(gen_input.constraints.exclude_methods)}")

        parts.append(f"\nTime range: {gen_input.constraints.time_horizon_start}-{gen_input.constraints.time_horizon_end}")

        if context:
            parts.append(f"\n\nAdditional context from previous agents:\n{context}")

        parts.append("\nProduce the JSON specification now.")
        return "\n".join(parts)

    def _fallback_queries(self, gen_input: GenerationInput) -> list[str]:
        """Generate basic queries if the LLM didn't produce enough."""
        title = gen_input.title
        return [
            title,
            f"{title} survey",
            f"{title} benchmark",
            f"{title} deep learning",
            f"{title} state of the art",
        ]

    def _summarize(self, result: Any) -> str:
        if isinstance(result, RequirementOutput):
            return (
                f"Research question: {result.primary_question}\n"
                f"Domain: {', '.join(result.domain_labels.primary)}\n"
                f"Queries: {len(result.positive_queries)} positive, "
                f"{len(result.negative_queries)} negative\n"
                f"Type: {result.contribution_type.value}\n"
                f"Target papers: {result.target_papers_min}-{result.target_papers_max}"
            )
        return str(result)[:500]
