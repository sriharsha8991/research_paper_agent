"""Experimentation Agent — designs experiments and generates simulated results.

Implements §6.8 of plan_doc_v2.md:
- Experimental design: research questions, datasets, baselines, metrics
- Simulated result generation (clearly marked)
- Table formatting
- Statistical test suggestions
"""

from __future__ import annotations

import logging
from typing import Any

from article_generator.agents.base import BaseAgent
from article_generator.models import (
    ExperimentDesign,
    ExperimentResult,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an experimental AI researcher designing and analyzing experiments for a \
research paper. You create rigorous experimental setups with appropriate baselines.

Output ONLY valid JSON:
{
  "table_data": [
    {
      "method": "Baseline A",
      "dataset": "CIFAR-10",
      "accuracy": "92.3",
      "params": "11M",
      "note": "reproduced"
    },
    {
      "method": "Ours",
      "dataset": "CIFAR-10",
      "accuracy": "94.1 ± 0.3",
      "params": "8M",
      "note": "SIMULATED — values are illustrative projections"
    }
  ],
  "figures": [],
  "analysis": "Detailed analysis of results including ablation study...",
  "statistical_tests": [
    {
      "test": "paired t-test",
      "comparison": "Ours vs Baseline A",
      "p_value": "< 0.05",
      "significant": true
    }
  ],
  "is_simulated": true
}

CRITICAL RULES:
1. ALL results for the proposed method MUST be marked is_simulated=true
2. Baseline results should come from the retrieved literature (with citations)
3. Simulated results should be PLAUSIBLE but clearly labeled
4. Include ablation studies removing each component
5. Use at least 3 datasets and 5 baselines
6. Statistical tests should be appropriate for the data type
"""


class ExperimentationAgent(BaseAgent):
    agent_id = "experimentation"
    system_prompt = SYSTEM_PROMPT

    def execute(self, context: str, **kwargs: Any) -> ExperimentResult:
        title: str = kwargs["title"]
        contributions: str = kwargs.get("contributions_summary", "")
        architecture: str = kwargs.get("architecture_summary", "")
        experiment_design: ExperimentDesign | None = kwargs.get("experiment_design")
        retrieval_result: RetrievalResult = kwargs.get("retrieval_result", RetrievalResult())

        results_context = self._format_results_context(retrieval_result)

        design_str = ""
        if experiment_design:
            design_str = (
                f"Research Questions: {experiment_design.research_questions}\n"
                f"Datasets: {experiment_design.datasets}\n"
                f"Baselines: {experiment_design.baselines}\n"
                f"Metrics: {experiment_design.metrics}\n"
            )

        prompt = (
            f"Paper Title: {title}\n\n"
            f"Contributions:\n{contributions}\n\n"
            f"Architecture:\n{architecture}\n\n"
            f"Experiment Design:\n{design_str}\n\n"
            f"Results from Related Papers:\n{results_context}\n\n"
            f"Blackboard Context:\n{context}\n\n"
            f"Generate the experimental results, tables, and analysis.\n"
            f"REMEMBER: Mark all results for the proposed method as SIMULATED."
        )

        response = self.call_llm(prompt, temperature=0.3, max_tokens=4096)
        result = self.parse_json_response(response, ExperimentResult)

        # Enforce simulation flag
        result.is_simulated = True

        return result

    def _format_results_context(self, retrieval_result: RetrievalResult) -> str:
        parts = []
        for chunk in retrieval_result.chunks:
            if chunk.chunk_type.value in ("result_chunk", "table_chunk"):
                parts.append(
                    f"[{chunk.title}, {chunk.year}] {chunk.section}:\n"
                    f"{chunk.text[:500]}"
                )
                if chunk.metrics_mentioned:
                    parts.append(f"Metrics: {', '.join(chunk.metrics_mentioned)}")
                if chunk.datasets_mentioned:
                    parts.append(f"Datasets: {', '.join(chunk.datasets_mentioned)}")
        return "\n\n".join(parts[:10]) if parts else "No experimental results retrieved."

    def _summarize(self, result: Any) -> str:
        if isinstance(result, ExperimentResult):
            return (
                f"Results: {len(result.table_data)} table rows\n"
                f"Statistical tests: {len(result.statistical_tests)}\n"
                f"Simulated: {result.is_simulated}\n"
                f"Analysis length: {len(result.analysis)} chars"
            )
        return str(result)[:500]
