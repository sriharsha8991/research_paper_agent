"""Three-tier memory system (§4.3).

Tier 1: Working Memory — per-call scratchpad (in-context)
Tier 2: Episodic Memory — shared blackboard across agents
Tier 3: Long-Term Memory — cross-run persistent storage
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from article_generator.context.token_budget import count_tokens, truncate_to_tokens
from article_generator.models import (
    AgentOutputRecord,
    CrossReferences,
    EpisodicMemory,
    GlobalDecision,
    TaskStatus,
    WorkingMemory,
)

logger = logging.getLogger(__name__)


class Blackboard:
    """Tier 2: Episodic Memory — the shared state across all agents in a run.

    Implements the Shared Blackboard Protocol (§5.2).
    All agents read/write through this interface.
    """

    def __init__(self, persist_path: Path | None = None):
        self._memory = EpisodicMemory()
        self._persist_path = persist_path

    @property
    def memory(self) -> EpisodicMemory:
        return self._memory

    def set_paper_identity(
        self,
        title: str,
        thesis: str = "",
        contributions: list[str] | None = None,
        venue: str = "",
    ) -> None:
        self._memory.paper_identity = {
            "title": title,
            "thesis_statement": thesis,
            "contribution_bullets": contributions or [],
            "target_venue": venue,
        }
        self._auto_save()

    def publish(
        self,
        agent_id: str,
        output: Any,
        summary: str,
    ) -> None:
        """Agent writes its completed output to the blackboard."""
        self._memory.agent_outputs[agent_id] = AgentOutputRecord(
            status=TaskStatus.COMPLETED,
            summary=summary,
            full_output=output,
            timestamp=datetime.now().isoformat(),
            token_count=count_tokens(str(output)),
        )
        logger.info("Blackboard: %s published output (%d tokens)", agent_id, count_tokens(str(output)))
        self._auto_save()

    def read(self, keys: list[str]) -> dict[str, Any]:
        """Read specific keys from the blackboard."""
        result = {}
        mem_dict = self._memory.model_dump()
        for k in keys:
            if k in mem_dict:
                result[k] = mem_dict[k]
        return result

    def get_agent_summary(self, agent_id: str) -> str:
        """Get the compressed summary of an agent's output."""
        record = self._memory.agent_outputs.get(agent_id)
        if record and record.status == TaskStatus.COMPLETED:
            return record.summary
        return ""

    def get_all_summaries(self, max_tokens: int = 4000) -> str:
        """Get concatenated summaries from all completed agents."""
        parts = []
        for agent_id, record in self._memory.agent_outputs.items():
            if record.status == TaskStatus.COMPLETED and record.summary:
                parts.append(f"[{agent_id}]: {record.summary}")

        combined = "\n\n".join(parts)
        if count_tokens(combined) > max_tokens:
            combined = truncate_to_tokens(combined, max_tokens)
        return combined

    def publish_decision(self, agent_id: str, decision: str, reason: str) -> None:
        """Record a global decision that constrains future agents."""
        self._memory.global_decisions.append(
            GlobalDecision(decision=decision, reason=reason, agent=agent_id)
        )
        logger.info("Blackboard: Decision by %s: %s", agent_id, decision[:80])
        self._auto_save()

    def get_decisions(self) -> list[GlobalDecision]:
        return self._memory.global_decisions

    def register_equation(self, label: str, latex: str) -> None:
        self._memory.cross_references.equation_registry[label] = latex
        self._auto_save()

    def register_notation(self, symbol: str, meaning: str) -> None:
        self._memory.cross_references.notation_table[symbol] = meaning
        self._auto_save()

    def register_figure(self, label: str, caption: str) -> None:
        self._memory.cross_references.figure_registry[label] = caption
        self._auto_save()

    def register_table(self, label: str, caption: str) -> None:
        self._memory.cross_references.table_registry[label] = caption
        self._auto_save()

    def get_cross_references(self) -> CrossReferences:
        return self._memory.cross_references

    def update_cross_references(self, cross_refs: CrossReferences) -> None:
        """Update the shared cross-references."""
        self._memory.cross_references = cross_refs
        self._auto_save()

    def add_invariant(self, invariant: str) -> None:
        self._memory.consistency_invariants.append(invariant)
        self._auto_save()

    def get_invariants(self) -> list[str]:
        return self._memory.consistency_invariants

    def build_context_for_agent(self, agent_id: str, max_tokens: int = 4000) -> str:
        """Build a compressed context payload for a specific agent.

        Implements the Context Router (§4.5) — each agent gets
        relevant context, not the entire blackboard.
        """
        parts = []

        # Paper identity (always included)
        identity = self._memory.paper_identity
        if identity:
            parts.append(f"## Paper Identity\n{json.dumps(identity, indent=2)}")

        # Relevant decisions
        decisions = self._memory.global_decisions
        if decisions:
            dec_text = "\n".join(
                f"- [{d.agent}] {d.decision}" for d in decisions[-10:]
            )
            parts.append(f"## Key Decisions\n{dec_text}")

        # Cross-references (notation, equations)
        xref = self._memory.cross_references
        if xref.notation_table:
            notation = "\n".join(f"  {k}: {v}" for k, v in xref.notation_table.items())
            parts.append(f"## Notation\n{notation}")

        # Summaries from completed agents
        summaries = self.get_all_summaries(max_tokens=max_tokens // 2)
        if summaries:
            parts.append(f"## Completed Agent Outputs\n{summaries}")

        # Invariants
        invs = self._memory.consistency_invariants
        if invs:
            inv_text = "\n".join(f"- {i}" for i in invs)
            parts.append(f"## Consistency Rules\n{inv_text}")

        combined = "\n\n".join(parts)
        if count_tokens(combined) > max_tokens:
            combined = truncate_to_tokens(combined, max_tokens)

        return combined

    # ── Persistence ──────────────────────────────────────────────────

    def _auto_save(self) -> None:
        if self._persist_path:
            self.save(self._persist_path)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._memory.model_dump(mode="json")
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Blackboard:
        if not path.exists():
            return cls(persist_path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        bb = cls(persist_path=path)
        bb._memory = EpisodicMemory.model_validate(data)
        return bb


class LongTermMemory:
    """Tier 3: Cross-run persistent memory (§4.3, §10.4).

    Stores: successful patterns, failure patterns,
    domain calibration data, and effective queries.
    """

    def __init__(self, persist_path: Path | None = None):
        self._path = persist_path or Path("./data/memory/long_term")
        self._data: dict[str, Any] = {
            "successful_patterns": [],
            "failure_patterns": [],
            "domain_calibration": {},
            "run_history": [],
        }
        self._load()

    def _load(self) -> None:
        path = self._path / "long_term_memory.json"
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        path = self._path / "long_term_memory.json"
        path.write_text(
            json.dumps(self._data, indent=2, default=str),
            encoding="utf-8",
        )

    def record_success(self, domain: str, pattern: dict[str, Any]) -> None:
        self._data["successful_patterns"].append({
            "domain": domain,
            "pattern": pattern,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def record_failure(self, issue: str, root_cause: str, fix: str) -> None:
        self._data["failure_patterns"].append({
            "issue": issue,
            "root_cause": root_cause,
            "fix": fix,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def update_calibration(self, domain: str, stats: dict[str, Any]) -> None:
        self._data["domain_calibration"][domain] = stats
        self.save()

    def record_run(self, run_summary: dict[str, Any]) -> None:
        self._data["run_history"].append(run_summary)
        # Keep last 50 runs
        self._data["run_history"] = self._data["run_history"][-50:]
        self.save()

    def get_patterns_for_domain(self, domain: str) -> list[dict[str, Any]]:
        return [
            p for p in self._data["successful_patterns"]
            if p.get("domain") == domain
        ]

    def get_failure_patterns(self) -> list[dict[str, Any]]:
        return self._data["failure_patterns"]

    def get_calibration(self, domain: str) -> dict[str, Any]:
        return self._data["domain_calibration"].get(domain, {})
