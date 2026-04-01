"""Architecture / Method Agent — designs model architecture and algorithm.

Implements §6.7 of plan_doc_v2.md:
- Model structure design
- Component specification
- Algorithm pseudocode
- Data flow description
- Complexity analysis
"""

from __future__ import annotations

import logging
from typing import Any

from article_generator.agents.base import BaseAgent
from article_generator.models import ArchitectureOutput, RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an AI systems architect designing the model/method for a research paper. \
You create clear, reproducible technical descriptions.

Output ONLY valid JSON:
{
  "model_structure": "High-level description of the model architecture",
  "components": [
    {
      "name": "Component A",
      "description": "What it does",
      "inputs": "Input specification",
      "outputs": "Output specification",
      "key_innovation": "What makes this component novel"
    }
  ],
  "data_flow": "Step-by-step data flow through the system",
  "algorithm_steps": [
    "Step 1: Initialize ...",
    "Step 2: For each ...",
    "Step 3: Compute ..."
  ],
  "complexity": {
    "time": "O(...)",
    "space": "O(...)",
    "parameters": "Approximate parameter count"
  }
}

Rules:
1. Be specific enough for reproduction — no hand-waving
2. Reference mathematical notation from the math formulation
3. Each component must have clear inputs/outputs
4. Algorithm steps should be implementable
5. Include at most 5-7 core components to keep it focused
"""


class ArchitectureAgent(BaseAgent):
    agent_id = "architecture"
    system_prompt = SYSTEM_PROMPT

    def execute(self, context: str, **kwargs: Any) -> ArchitectureOutput:
        title: str = kwargs["title"]
        contributions: str = kwargs.get("contributions_summary", "")
        math_summary: str = kwargs.get("math_summary", "")
        retrieval_result: RetrievalResult = kwargs.get("retrieval_result", RetrievalResult())

        method_context = self._format_method_context(retrieval_result)

        prompt = (
            f"Paper Title: {title}\n\n"
            f"Contributions:\n{contributions}\n\n"
            f"Mathematical Framework:\n{math_summary}\n\n"
            f"Related Methods from Literature:\n{method_context}\n\n"
            f"Blackboard Context:\n{context}\n\n"
            f"Design the complete model architecture and algorithm."
        )

        response = self.call_llm(prompt, temperature=0.3, max_tokens=4096)
        return self.parse_json_response(response, ArchitectureOutput)

    def _format_method_context(self, retrieval_result: RetrievalResult) -> str:
        parts = []
        for chunk in retrieval_result.chunks:
            if chunk.chunk_type.value in ("method_chunk", "algorithm_chunk"):
                parts.append(
                    f"[{chunk.title}] {chunk.section}:\n{chunk.text[:500]}"
                )
        return "\n\n".join(parts[:8]) if parts else "No method context available."

    def _summarize(self, result: Any) -> str:
        if isinstance(result, ArchitectureOutput):
            components = ", ".join(c.get("name", "?") for c in result.components[:5])
            return (
                f"Architecture: {result.model_structure[:100]}\n"
                f"Components: {components}\n"
                f"Algorithm steps: {len(result.algorithm_steps)}"
            )
        return str(result)[:500]
