"""Math Formulation Agent — generates formal mathematical framework.

Implements §6.6 of plan_doc_v2.md:
- Objective function formulation
- Constraint specification
- Optimization strategy
- Proof sketches
- Notation consistency enforcement via cross-references
"""

from __future__ import annotations

import logging
from typing import Any

from article_generator.agents.base import BaseAgent
from article_generator.models import MathFormulationOutput, RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a mathematical AI researcher formulating the formal framework for a \
research paper. You create rigorous, consistent mathematical notation and \
objective functions.

Output ONLY valid JSON:
{
  "objective_function": "LaTeX of the main objective (e.g., \\\\min_{\\\\theta} ...)",
  "constraints": ["Constraint 1 in LaTeX", "Constraint 2"],
  "optimization_strategy": "How to solve (SGD, EM, variational, etc.)",
  "equations": {
    "eq:objective": "\\\\mathcal{L} = ...",
    "eq:loss": "...",
    "eq:update": "..."
  },
  "proofs": ["Proof sketch 1", "Proof sketch 2"],
  "complexity_analysis": "Time: O(...), Space: O(...)",
  "verification_results": {
    "dimensional_consistency": true,
    "convergence_claim": true
  }
}

Rules:
1. Use consistent notation: \\theta for parameters, x for input, y for output
2. Reference existing notation from the cross-references if provided
3. Every equation must have a label (eq:name)
4. Include at least one theorem or proposition with proof sketch
5. All LaTeX must compile (double-escape backslashes for JSON)
6. complexity_analysis must cover both time and space
"""


class MathAgent(BaseAgent):
    agent_id = "math_formulation"
    system_prompt = SYSTEM_PROMPT

    def execute(self, context: str, **kwargs: Any) -> MathFormulationOutput:
        title: str = kwargs["title"]
        contributions: str = kwargs.get("contributions_summary", "")
        retrieval_result: RetrievalResult = kwargs.get("retrieval_result", RetrievalResult())

        # Get math-specific chunks
        math_context = self._format_math_context(retrieval_result)

        # Get existing notation from blackboard
        cross_refs = self.blackboard.get_cross_references()
        notation_str = ""
        if cross_refs.notation_table:
            notation_str = "Existing notation:\n" + "\n".join(
                f"  {sym}: {desc}" for sym, desc in cross_refs.notation_table.items()
            )

        prompt = (
            f"Paper Title: {title}\n\n"
            f"Contributions:\n{contributions}\n\n"
            f"Mathematical context from related papers:\n{math_context}\n\n"
            f"{notation_str}\n\n"
            f"Blackboard Context:\n{context}\n\n"
            f"Formulate the complete mathematical framework."
        )

        response = self.call_llm(
            prompt,
            model=self.config.models.reasoning_model,  # Use reasoning model for math
            temperature=0.2,
            max_tokens=4096,
        )

        result = self.parse_json_response(response, MathFormulationOutput)

        # Register equations and notation in cross-references
        self._register_cross_references(result)

        return result

    def _format_math_context(self, retrieval_result: RetrievalResult) -> str:
        parts = []
        for chunk in retrieval_result.chunks:
            if chunk.has_equation or chunk.chunk_type.value in ("math_chunk", "method_chunk"):
                parts.append(
                    f"From [{chunk.title}, {chunk.year}]:\n"
                    f"{chunk.text[:600]}"
                )
                if chunk.equations_latex:
                    parts.append("Equations: " + "; ".join(chunk.equations_latex[:3]))
        return "\n\n".join(parts[:10]) if parts else "No mathematical context available."

    def _register_cross_references(self, result: MathFormulationOutput) -> None:
        """Register equations and notation in the shared blackboard."""
        cross_refs = self.blackboard.get_cross_references()

        # Register equations
        for label, latex in result.equations.items():
            cross_refs.equation_registry[label] = latex

        # Register notation from the result
        for symbol, description in result.verification_results.items():
            if isinstance(description, str):
                cross_refs.notation_table[symbol] = description

        self.blackboard.update_cross_references(cross_refs)

    def _summarize(self, result: Any) -> str:
        if isinstance(result, MathFormulationOutput):
            return (
                f"Objective: {result.objective_function[:100]}\n"
                f"Equations: {len(result.equations)}\n"
                f"Proofs: {len(result.proofs)}\n"
                f"Complexity: {result.complexity_analysis[:100]}"
            )
        return str(result)[:500]
