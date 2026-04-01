"""Token budget accounting system (§4.2).

Every LLM call has an explicit token budget. This module ensures
no agent silently overflows the context window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import tiktoken

logger = logging.getLogger(__name__)

# Default tokenizer — cl100k_base covers Claude/GPT-4
_ENCODER: tiktoken.Encoding | None = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base)."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return len(_ENCODER.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    tokens = _ENCODER.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens])


@dataclass
class TokenBudget:
    """Manages token allocation for a single LLM call.

    Ensures the total context fits within the model's limit
    by tracking allocations per category.
    """

    model_context_limit: int = 128_000
    allocations: dict[str, int] = field(default_factory=lambda: {
        "system_prompt": 2000,
        "task_description": 1000,
        "working_memory": 4000,
        "retrieved_context": 60000,
        "conversation_history": 8000,
        "ontology_context": 2000,
        "output_budget": 16000,
        "safety_margin": 5000,
    })
    _used: dict[str, int] = field(default_factory=dict)

    @property
    def total_allocated(self) -> int:
        return sum(self.allocations.values())

    @property
    def total_used(self) -> int:
        return sum(self._used.values())

    @property
    def remaining(self) -> int:
        return self.model_context_limit - self.total_used

    def allocate(self, category: str, content: str) -> str:
        """Fit content within the budget for a given category.

        Returns truncated content if it exceeds the allocation.
        """
        if category not in self.allocations:
            logger.warning("Unknown budget category: %s", category)
            return content

        max_tokens = self.allocations[category]
        token_count = count_tokens(content)

        if token_count <= max_tokens:
            self._used[category] = token_count
            return content

        logger.info(
            "Truncating '%s': %d tokens → %d tokens",
            category, token_count, max_tokens,
        )
        truncated = truncate_to_tokens(content, max_tokens)
        self._used[category] = max_tokens
        return truncated

    def get_usage_report(self) -> dict[str, dict[str, int]]:
        """Return a summary of budget vs used per category."""
        return {
            cat: {
                "budget": self.allocations.get(cat, 0),
                "used": self._used.get(cat, 0),
            }
            for cat in self.allocations
        }


# ── Agent-specific budget presets ────────────────────────────────────

AGENT_BUDGETS: dict[str, dict[str, int]] = {
    "literature_review": {
        "system_prompt": 2000,
        "task_description": 1000,
        "working_memory": 4000,
        "retrieved_context": 60000,
        "conversation_history": 8000,
        "ontology_context": 2000,
        "output_budget": 16000,
        "safety_margin": 5000,
    },
    "math_formulation": {
        "system_prompt": 2000,
        "task_description": 1000,
        "working_memory": 4000,
        "retrieved_context": 20000,
        "conversation_history": 4000,
        "ontology_context": 1000,
        "output_budget": 8000,
        "safety_margin": 5000,
    },
    "experimentation": {
        "system_prompt": 2000,
        "task_description": 1000,
        "working_memory": 8000,
        "retrieved_context": 30000,
        "conversation_history": 8000,
        "ontology_context": 1000,
        "output_budget": 16000,
        "safety_margin": 5000,
    },
    "results_analysis": {
        "system_prompt": 2000,
        "task_description": 1000,
        "working_memory": 8000,
        "retrieved_context": 40000,
        "conversation_history": 12000,
        "ontology_context": 1000,
        "output_budget": 8000,
        "safety_margin": 5000,
    },
    "discussion": {
        "system_prompt": 2000,
        "task_description": 1000,
        "working_memory": 8000,
        "retrieved_context": 30000,
        "conversation_history": 16000,
        "ontology_context": 1000,
        "output_budget": 12000,
        "safety_margin": 5000,
    },
    "default": {
        "system_prompt": 2000,
        "task_description": 1000,
        "working_memory": 4000,
        "retrieved_context": 40000,
        "conversation_history": 8000,
        "ontology_context": 2000,
        "output_budget": 12000,
        "safety_margin": 5000,
    },
}


def get_budget_for_agent(agent_id: str) -> TokenBudget:
    """Get a pre-configured budget for a specific agent."""
    allocs = AGENT_BUDGETS.get(agent_id, AGENT_BUDGETS["default"])
    return TokenBudget(allocations=dict(allocs))
