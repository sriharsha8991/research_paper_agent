"""Base agent class that all sub-agents inherit from.

Handles: context injection, token budgets, blackboard I/O,
output validation, and structured logging.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from article_generator.config import AppConfig
from article_generator.context.memory import Blackboard
from article_generator.context.token_budget import TokenBudget, get_budget_for_agent
from article_generator.llm_client import LLMClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """Abstract base for all research paper agents.

    Each agent:
    1. Receives context from the blackboard
    2. Gets a task-specific token budget
    3. Calls the LLM with structured prompts
    4. Validates output against its typed contract
    5. Publishes results back to the blackboard
    """

    agent_id: str = "base"
    system_prompt: str = ""

    def __init__(
        self,
        llm: LLMClient,
        config: AppConfig,
        blackboard: Blackboard,
    ):
        self.llm = llm
        self.config = config
        self.blackboard = blackboard
        self.budget = get_budget_for_agent(self.agent_id)

    def run(self, **kwargs: Any) -> Any:
        """Execute the agent: build context → generate → validate → publish."""
        logger.info("Agent [%s] starting", self.agent_id)

        try:
            # Step 1: Build context from blackboard
            context = self._build_context(**kwargs)

            # Step 2: Execute agent-specific logic
            result = self.execute(context=context, **kwargs)

            # Step 3: Validate output
            validated = self._validate_output(result)

            # Step 4: Publish to blackboard
            summary = self._summarize(validated)
            self.blackboard.publish(self.agent_id, validated, summary)

            logger.info("Agent [%s] completed successfully", self.agent_id)
            return validated

        except Exception as e:
            logger.error("Agent [%s] failed: %s", self.agent_id, e)
            raise

    @abstractmethod
    def execute(self, context: str, **kwargs: Any) -> Any:
        """Agent-specific logic. Must be implemented by subclasses."""
        ...

    def _build_context(self, **kwargs: Any) -> str:
        """Assemble the context payload for this agent's LLM call."""
        # Get blackboard context (compressed for this agent)
        bb_context = self.blackboard.build_context_for_agent(
            self.agent_id, max_tokens=self.budget.allocations.get("conversation_history", 4000)
        )
        return self.budget.allocate("conversation_history", bb_context)

    def _validate_output(self, result: Any) -> Any:
        """Validate the agent's output. Override for typed validation."""
        return result

    def _summarize(self, result: Any) -> str:
        """Generate a concise summary for other agents. Override per agent."""
        if isinstance(result, BaseModel):
            return str(result.model_dump())[:500]
        return str(result)[:500]

    def call_llm(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Convenience wrapper that applies budget-managed system prompt."""
        sys = system if system is not None else self.system_prompt
        sys = self.budget.allocate("system_prompt", sys)
        prompt = self.budget.allocate("task_description", prompt)
        return self.llm.generate(
            prompt,
            system=sys,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def parse_json_response(self, response: str, model_class: type[T]) -> T:
        """Parse an LLM response as JSON into a Pydantic model.

        Handles common LLM quirks: markdown code fences, trailing text.
        """
        text = response.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        try:
            data = json.loads(text)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                "Failed to parse JSON response for %s: %s\nResponse: %s",
                model_class.__name__, e, response[:200],
            )
            raise
