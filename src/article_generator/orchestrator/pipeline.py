"""DAG Orchestrator — manages the research paper generation pipeline.

Implements §9 of plan_doc_v2.md:
- Static DAG definition with dependencies
- Sequential task execution with dependency resolution
- Error handling and retry logic
- Pipeline state persistence
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from article_generator.agents.base import BaseAgent
from article_generator.agents.architecture_agent import ArchitectureAgent
from article_generator.agents.contributions_agent import ContributionsAgent
from article_generator.agents.experimentation_agent import ExperimentationAgent
from article_generator.agents.literature_review_agent import LiteratureReviewAgent
from article_generator.agents.math_agent import MathAgent
from article_generator.agents.requirement_agent import RequirementAgent
from article_generator.agents.writing_agents import (
    AbstractAgent,
    ConclusionAgent,
    DiscussionAgent,
    IntroductionAgent,
)
from article_generator.config import AppConfig
from article_generator.context.memory import Blackboard, LongTermMemory
from article_generator.discovery.discovery_agent import DiscoveryAgent
from article_generator.ingestion.pipeline import IngestionPipeline
from article_generator.knowledge.knowledge_graph import KnowledgeGraph
from article_generator.knowledge.vector_store import VectorStore
from article_generator.llm_client import LLMClient
from article_generator.models import (
    ContributionsOutput,
    DAGState,
    DiscoveryOutput,
    GeneratedPaper,
    GenerationInput,
    RequirementOutput,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    TaskNode,
    TaskStatus,
)
from article_generator.quality.humanizer import HumanizedWriter
from article_generator.quality.novelty import ConfidenceScorer, NoveltyScorer
from article_generator.quality.verification import ChainOfVerification, ProvenanceTracker
from article_generator.retrieval.orchestrator import RetrievalOrchestrator

logger = logging.getLogger(__name__)

# Pipeline stage definitions with dependencies
PIPELINE_STAGES = [
    ("requirement_gathering", []),
    ("paper_discovery", ["requirement_gathering"]),
    ("ingestion", ["paper_discovery"]),
    ("indexing", ["ingestion"]),
    ("contributions", ["indexing"]),
    ("literature_review", ["contributions"]),
    ("math_formulation", ["contributions"]),
    ("architecture", ["math_formulation"]),
    ("experimentation", ["architecture"]),
    ("introduction", ["literature_review", "contributions"]),
    ("discussion", ["experimentation"]),
    ("conclusion", ["discussion"]),
    ("abstract_writer", ["conclusion"]),
    ("quality_control", ["abstract_writer"]),
]


class PipelineOrchestrator:
    """Orchestrates the full research paper generation pipeline."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = LLMClient(config)
        self.blackboard = Blackboard()
        self.ltm = LongTermMemory()

        # Infrastructure
        self.vector_store = VectorStore(config)
        self.kg = KnowledgeGraph(config)
        self.ingestion = IngestionPipeline(config)
        self.retrieval = RetrievalOrchestrator(
            self.vector_store, self.kg, self.llm, config
        )

        # Quality
        self.cove = ChainOfVerification(self.llm)
        self.provenance = ProvenanceTracker(self.llm)
        self.novelty_scorer = NoveltyScorer()
        self.confidence_scorer = ConfidenceScorer()
        self.humanizer = HumanizedWriter(self.llm)

        # Agents
        self.agents: dict[str, BaseAgent] = {
            "requirement_gathering": RequirementAgent(self.llm, config, self.blackboard),
            "paper_discovery": DiscoveryAgent(self.llm, config, self.blackboard),
            "contributions": ContributionsAgent(self.llm, config, self.blackboard),
            "literature_review": LiteratureReviewAgent(self.llm, config, self.blackboard),
            "math_formulation": MathAgent(self.llm, config, self.blackboard),
            "architecture": ArchitectureAgent(self.llm, config, self.blackboard),
            "experimentation": ExperimentationAgent(self.llm, config, self.blackboard),
            "introduction": IntroductionAgent(self.llm, config, self.blackboard),
            "discussion": DiscussionAgent(self.llm, config, self.blackboard),
            "conclusion": ConclusionAgent(self.llm, config, self.blackboard),
            "abstract_writer": AbstractAgent(self.llm, config, self.blackboard),
        }

        # State
        self.dag = self._build_dag()
        self._results: dict[str, Any] = {}

    def _build_dag(self) -> DAGState:
        """Build the initial DAG state."""
        tasks = []
        for stage_id, deps in PIPELINE_STAGES:
            tasks.append(
                TaskNode(
                    task_id=stage_id,
                    agent_id=stage_id,
                    dependencies=deps,
                )
            )
        return DAGState(tasks=tasks, current_phase="initialization")

    def run(self, gen_input: GenerationInput) -> GeneratedPaper:
        """Execute the full pipeline."""
        logger.info("Starting pipeline for: %s", gen_input.title)

        # Store identity in blackboard
        self.blackboard.publish(
            "system",
            {"title": gen_input.title, "venue": gen_input.target_venue.value},
            f"Paper: {gen_input.title}",
        )

        # Execute stages in dependency order
        for task in self.dag.tasks:
            if not self._dependencies_met(task):
                logger.error("Dependencies not met for %s — skipping", task.task_id)
                task.status = TaskStatus.SKIPPED
                continue

            try:
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now().isoformat()
                self.dag.current_phase = task.task_id

                logger.info("=== Stage: %s ===", task.task_id)
                self._execute_stage(task.task_id, gen_input)

                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()

            except Exception as e:
                logger.error("Stage %s failed: %s", task.task_id, e)
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.retries += 1

                if task.retries < task.max_retries:
                    logger.info("Retrying stage %s (%d/%d)", task.task_id, task.retries, task.max_retries)
                    task.status = TaskStatus.RECOVERING
                    try:
                        self._execute_stage(task.task_id, gen_input)
                        task.status = TaskStatus.COMPLETED
                    except Exception as retry_err:
                        logger.error("Retry failed for %s: %s", task.task_id, retry_err)
                        task.status = TaskStatus.FAILED

        # Assemble final paper
        paper = self._assemble_paper(gen_input)
        logger.info("Pipeline complete. Confidence: %.2f", paper.confidence_report.overall_confidence)

        # Record in long-term memory
        self.ltm.record_success(gen_input.title, {"confidence": paper.confidence_report.overall_confidence})

        return paper

    def _dependencies_met(self, task: TaskNode) -> bool:
        """Check if all dependencies have completed."""
        for dep_id in task.dependencies:
            dep_task = next((t for t in self.dag.tasks if t.task_id == dep_id), None)
            if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    def _execute_stage(self, stage_id: str, gen_input: GenerationInput) -> None:
        """Execute a single pipeline stage."""
        if stage_id == "requirement_gathering":
            result = self.agents["requirement_gathering"].run(generation_input=gen_input)
            self._results["requirement"] = result

        elif stage_id == "paper_discovery":
            result = self.agents["paper_discovery"].run(requirement=self._results["requirement"])
            self._results["discovery"] = result

        elif stage_id == "ingestion":
            discovery: DiscoveryOutput = self._results["discovery"]
            chunks = self.ingestion.ingest_corpus(discovery.corpus)
            self._results["chunks"] = chunks

        elif stage_id == "indexing":
            chunks = self._results["chunks"]
            self.vector_store.index_chunks(chunks)
            self.kg.build_from_corpus(self._results["discovery"].corpus, chunks)
            self.kg.save()

        elif stage_id == "contributions":
            retrieval = self._retrieve_for("contributions", gen_input.title)
            result = self.agents["contributions"].run(
                title=gen_input.title,
                abstract=gen_input.abstract or "",
                retrieval_result=retrieval,
            )
            self._results["contributions"] = result

        elif stage_id == "literature_review":
            retrieval = self._retrieve_for("literature_review", gen_input.title)
            contributions: ContributionsOutput = self._results["contributions"]
            result = self.agents["literature_review"].run(
                title=gen_input.title,
                contributions_summary=contributions.positioning_statement,
                retrieval_result=retrieval,
            )
            self._results["literature_review"] = result

        elif stage_id == "math_formulation":
            retrieval = self._retrieve_for("math", gen_input.title)
            contributions = self._results["contributions"]
            result = self.agents["math_formulation"].run(
                title=gen_input.title,
                contributions_summary=contributions.positioning_statement,
                retrieval_result=retrieval,
            )
            self._results["math"] = result

        elif stage_id == "architecture":
            retrieval = self._retrieve_for("method", gen_input.title)
            result = self.agents["architecture"].run(
                title=gen_input.title,
                contributions_summary=self._results["contributions"].positioning_statement,
                math_summary=self._results["math"].objective_function,
                retrieval_result=retrieval,
            )
            self._results["architecture"] = result

        elif stage_id == "experimentation":
            retrieval = self._retrieve_for("experiments", gen_input.title)
            result = self.agents["experimentation"].run(
                title=gen_input.title,
                contributions_summary=self._results["contributions"].positioning_statement,
                architecture_summary=self._results["architecture"].model_structure,
                retrieval_result=retrieval,
            )
            self._results["experimentation"] = result

        elif stage_id == "introduction":
            retrieval = self._retrieve_for("introduction", gen_input.title)
            result = self.agents["introduction"].run(
                title=gen_input.title,
                contributions_summary=self._results["contributions"].positioning_statement,
                lit_review_summary=self._results["literature_review"].narrative[:500],
                retrieval_result=retrieval,
            )
            self._results["introduction"] = result

        elif stage_id == "discussion":
            retrieval = self._retrieve_for("discussion", gen_input.title)
            result = self.agents["discussion"].run(
                title=gen_input.title,
                contributions_summary=self._results["contributions"].positioning_statement,
                results_summary=self._results["experimentation"].analysis[:500],
                retrieval_result=retrieval,
            )
            self._results["discussion"] = result

        elif stage_id == "conclusion":
            result = self.agents["conclusion"].run(title=gen_input.title)
            self._results["conclusion"] = result

        elif stage_id == "abstract_writer":
            result = self.agents["abstract_writer"].run(title=gen_input.title)
            self._results["abstract"] = result

        elif stage_id == "quality_control":
            self._run_quality_control(gen_input)

    def _retrieve_for(self, section: str, title: str) -> RetrievalResult:
        """Retrieve context for a specific section."""
        queries = [title, f"{title} {section}"]
        return self.retrieval.retrieve_for_section(section, queries, top_k=15)

    def _run_quality_control(self, gen_input: GenerationInput) -> None:
        """Run quality control: CoVe, provenance, novelty, humanization."""
        sections = self._collect_sections()
        all_source = self.retrieval.retrieve(
            RetrievalQuery(query=gen_input.title, top_k=30, strategy=RetrievalStrategy.HYBRID)
        )

        # CoVe verification
        verified_sections = {}
        for name, content in sections.items():
            verified = self.cove.verify_section(content, all_source, name)
            verified_sections[name] = verified.text
            logger.info(
                "CoVe [%s]: %d corrections, confidence=%.2f",
                name, verified.corrections_made, verified.confidence,
            )

        # Humanization pass
        humanized_sections = {}
        for name, content in verified_sections.items():
            humanized_sections[name] = self.humanizer.humanize(content, name)

        # Update results with humanized text
        self._results["final_sections"] = humanized_sections

        # Provenance tracking
        provenance_records = []
        for name, content in humanized_sections.items():
            records = self.provenance.track_section(content, all_source)
            provenance_records.extend(records)
        self._results["provenance"] = provenance_records

        # Novelty scoring
        novelty = self.novelty_scorer.score(humanized_sections, self._results.get("chunks", []))
        self._results["novelty"] = novelty

        # Confidence scoring
        confidence = self.confidence_scorer.score(
            humanized_sections, provenance_records
        )
        self._results["confidence"] = confidence

    def _collect_sections(self) -> dict[str, str]:
        """Collect all generated section texts."""
        sections = {}

        for key in ["introduction", "literature_review", "discussion", "conclusion", "abstract"]:
            result = self._results.get(key)
            if result and hasattr(result, "content"):
                sections[key] = result.content
            elif result and hasattr(result, "narrative"):
                sections[key] = result.narrative

        # Math and architecture
        math = self._results.get("math")
        if math:
            sections["method_math"] = math.objective_function + "\n" + "\n".join(
                f"{k}: {v}" for k, v in math.equations.items()
            )

        arch = self._results.get("architecture")
        if arch:
            sections["method_arch"] = arch.model_structure + "\n" + arch.data_flow

        exp = self._results.get("experimentation")
        if exp:
            sections["experiments"] = exp.analysis

        return sections

    def _assemble_paper(self, gen_input: GenerationInput) -> GeneratedPaper:
        """Assemble the final paper from all generated sections."""
        final_sections = self._results.get("final_sections", self._collect_sections())

        # Build BibTeX from discovered papers
        bibtex = self._build_bibtex()

        return GeneratedPaper(
            title=gen_input.title,
            sections=final_sections,
            bibtex=bibtex,
            confidence_report=self._results.get("confidence", {}),
            provenance_records=self._results.get("provenance", []),
            novelty_report=self._results.get("novelty", {}),
            generation_log={
                "stages_completed": [
                    t.task_id for t in self.dag.tasks if t.status == TaskStatus.COMPLETED
                ],
                "stages_failed": [
                    t.task_id for t in self.dag.tasks if t.status == TaskStatus.FAILED
                ],
                "llm_stats": self.llm.stats,
            },
        )

    def _build_bibtex(self) -> str:
        """Generate BibTeX entries from discovered papers."""
        discovery: DiscoveryOutput | None = self._results.get("discovery")
        if not discovery:
            return ""

        entries = []
        for paper in discovery.corpus[:50]:
            safe_id = paper.paper_id.replace(":", "_").replace("/", "_")
            authors_str = " and ".join(paper.authors[:5])
            entry = (
                f"@article{{{safe_id},\n"
                f"  title = {{{paper.title}}},\n"
                f"  author = {{{authors_str}}},\n"
                f"  year = {{{paper.year}}},\n"
                f"  venue = {{{paper.venue}}},\n"
                f"}}\n"
            )
            entries.append(entry)

        return "\n".join(entries)
