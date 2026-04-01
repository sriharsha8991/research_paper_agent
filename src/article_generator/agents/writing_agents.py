"""Discussion & Related Section Agents — generates discussion, conclusion, and abstract.

These are simpler LLM-powered writing agents that follow the same BaseAgent pattern.
"""

from __future__ import annotations

import logging
from typing import Any

from article_generator.agents.base import BaseAgent
from article_generator.models import RetrievalResult, SectionOutput

logger = logging.getLogger(__name__)


class IntroductionAgent(BaseAgent):
    agent_id = "introduction"
    system_prompt = """\
You are writing the Introduction section of a top-tier AI conference paper. \
The introduction must: (1) motivate the problem, (2) state the research gap, \
(3) describe the approach at a high level, (4) list contributions, and \
(5) outline the paper structure.

Output ONLY valid JSON:
{
  "section_name": "introduction",
  "content": "The full introduction text with [AuthorYear] citations...",
  "citations_used": ["paper_id_1", ...],
  "confidence": 0.0 to 1.0
}
"""

    def execute(self, context: str, **kwargs: Any) -> SectionOutput:
        title = kwargs["title"]
        contributions = kwargs.get("contributions_summary", "")
        lit_review = kwargs.get("lit_review_summary", "")
        retrieval_result = kwargs.get("retrieval_result", RetrievalResult())

        chunks_text = "\n".join(
            f"[{c.title}, {c.year}]: {c.text[:300]}"
            for c in retrieval_result.chunks[:10]
        )

        prompt = (
            f"Paper Title: {title}\n\n"
            f"Contributions:\n{contributions}\n\n"
            f"Literature Review Summary:\n{lit_review}\n\n"
            f"Key Papers:\n{chunks_text}\n\n"
            f"Blackboard:\n{context}\n\n"
            f"Write the Introduction section."
        )
        response = self.call_llm(prompt, temperature=0.4, max_tokens=3000)
        return self.parse_json_response(response, SectionOutput)


class DiscussionAgent(BaseAgent):
    agent_id = "discussion"
    system_prompt = """\
You are writing the Discussion section of a top-tier AI conference paper. \
Discuss: (1) key findings and their implications, (2) comparison with prior work, \
(3) broader impact, (4) limitations and future work.

Output ONLY valid JSON:
{
  "section_name": "discussion",
  "content": "The full discussion text...",
  "citations_used": ["paper_id_1", ...],
  "confidence": 0.0 to 1.0
}
"""

    def execute(self, context: str, **kwargs: Any) -> SectionOutput:
        title = kwargs["title"]
        contributions = kwargs.get("contributions_summary", "")
        results = kwargs.get("results_summary", "")
        retrieval_result = kwargs.get("retrieval_result", RetrievalResult())

        chunks_text = "\n".join(
            f"[{c.title}, {c.year}]: {c.text[:300]}"
            for c in retrieval_result.chunks[:8]
        )

        prompt = (
            f"Paper Title: {title}\n\n"
            f"Contributions:\n{contributions}\n\n"
            f"Experimental Results:\n{results}\n\n"
            f"Related Work:\n{chunks_text}\n\n"
            f"Blackboard:\n{context}\n\n"
            f"Write the Discussion section."
        )
        response = self.call_llm(prompt, temperature=0.4, max_tokens=3000)
        return self.parse_json_response(response, SectionOutput)


class ConclusionAgent(BaseAgent):
    agent_id = "conclusion"
    system_prompt = """\
You are writing the Conclusion section of a top-tier AI conference paper. \
Summarize contributions, key results, limitations, and future directions. \
Keep it concise (300-500 words).

Output ONLY valid JSON:
{
  "section_name": "conclusion",
  "content": "The conclusion text...",
  "citations_used": [],
  "confidence": 0.0 to 1.0
}
"""

    def execute(self, context: str, **kwargs: Any) -> SectionOutput:
        prompt = (
            f"Paper Title: {kwargs['title']}\n\n"
            f"Blackboard Context (all agent outputs):\n{context}\n\n"
            f"Write a concise Conclusion section."
        )
        response = self.call_llm(prompt, temperature=0.3, max_tokens=1500)
        return self.parse_json_response(response, SectionOutput)


class AbstractAgent(BaseAgent):
    agent_id = "abstract_writer"
    system_prompt = """\
You are writing the Abstract for a top-tier AI conference paper. \
The abstract must be self-contained, 150-250 words, covering: \
problem, approach, key results, and significance.

Output ONLY valid JSON:
{
  "section_name": "abstract",
  "content": "The abstract text...",
  "citations_used": [],
  "confidence": 0.0 to 1.0
}
"""

    def execute(self, context: str, **kwargs: Any) -> SectionOutput:
        prompt = (
            f"Paper Title: {kwargs['title']}\n\n"
            f"Full paper context (all agent outputs):\n{context}\n\n"
            f"Write a polished Abstract (150-250 words)."
        )
        response = self.call_llm(prompt, temperature=0.3, max_tokens=1000)
        return self.parse_json_response(response, SectionOutput)
