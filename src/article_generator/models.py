"""Core data models for the article generator system.

All agent inputs/outputs are typed via these Pydantic models,
enforcing the Typed Agent Contracts from §5.4 of the plan.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class ContributionType(str, Enum):
    THEORETICAL = "theoretical"
    EMPIRICAL = "empirical"
    HYBRID = "hybrid"


class VenueStyle(str, Enum):
    NEURIPS = "neurips"
    ICLR = "iclr"
    ICML = "icml"
    ACL = "acl"
    CVPR = "cvpr"
    AAAI = "aaai"
    JOURNAL = "journal"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"
    SKIPPED = "skipped"


class PaperRole(str, Enum):
    SEED = "seed"
    HUB = "hub"
    RECENT = "recent"
    CONTRASTIVE = "contrastive"


class ChunkType(str, Enum):
    ABSTRACT = "abstract_chunk"
    INTRODUCTION = "introduction_chunk"
    METHOD = "method_chunk"
    MATH = "math_chunk"
    RESULT = "result_chunk"
    TABLE = "table_chunk"
    DISCUSSION = "discussion_chunk"
    ALGORITHM = "algorithm_chunk"
    RELATED_WORK = "related_work_chunk"


class ProvenanceType(str, Enum):
    DIRECT_CITE = "direct_cite"
    PARAPHRASED = "paraphrased"
    SYNTHESIZED = "synthesized"
    INFERRED = "inferred"
    ORIGINAL = "original"
    SIMULATED = "simulated"


class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KG_GUIDED = "kg_guided"
    MULTI_HOP = "multi_hop"


# ── Input Models ─────────────────────────────────────────────────────

class GenerationConstraints(BaseModel):
    max_pages: int = 10
    must_cite: list[str] = Field(default_factory=list)
    exclude_methods: list[str] = Field(default_factory=list)
    time_horizon_start: int = 2020
    time_horizon_end: int = 2026


class GenerationInput(BaseModel):
    """Top-level input to the system."""
    title: str
    abstract: str | None = None
    research_direction: str | None = None
    target_venue: VenueStyle = VenueStyle.NEURIPS
    constraints: GenerationConstraints = Field(default_factory=GenerationConstraints)


# ── Requirement Agent Models ─────────────────────────────────────────

class DomainLabels(BaseModel):
    primary: list[str] = Field(default_factory=list)
    method: list[str] = Field(default_factory=list)
    task: list[str] = Field(default_factory=list)


class RequirementOutput(BaseModel):
    primary_question: str
    secondary_questions: list[str] = Field(default_factory=list)
    positive_queries: list[str] = Field(min_length=5)
    negative_queries: list[str] = Field(default_factory=list)
    domain_labels: DomainLabels = Field(default_factory=DomainLabels)
    contribution_type: ContributionType = ContributionType.HYBRID
    target_papers_min: int = 30
    target_papers_max: int = 80
    time_horizon_start: int = 2020
    time_horizon_end: int = 2026
    venue_style: VenueStyle = VenueStyle.NEURIPS


# ── Paper Discovery Models ───────────────────────────────────────────

class PaperMetadata(BaseModel):
    paper_id: str
    source: str  # "arxiv", "semantic_scholar", "openreview"
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    venue: str = ""
    abstract: str = ""
    citation_count: int = 0
    relevance_score: float = 0.0
    role: PaperRole = PaperRole.SEED
    pdf_path: str = ""
    pdf_url: str = ""
    has_code: bool = False
    code_url: str = ""
    doi: str = ""
    arxiv_id: str = ""


class CoverageReport(BaseModel):
    method_coverage: float = 0.0
    temporal_coverage: float = 0.0
    venue_diversity: float = 0.0


class DiscoveryOutput(BaseModel):
    corpus: list[PaperMetadata] = Field(default_factory=list)
    total_papers: int = 0
    coverage_report: CoverageReport = Field(default_factory=CoverageReport)


# ── Ingestion / Chunking Models ──────────────────────────────────────

class Claim(BaseModel):
    claim_text: str
    claim_type: str = "general"  # methodological, empirical, theoretical
    confidence: str = "medium"


class PaperChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    venue: str = ""
    section: str = ""
    subsection: str = ""
    chunk_type: ChunkType = ChunkType.METHOD
    text: str
    tags: list[str] = Field(default_factory=list)
    has_equation: bool = False
    equations_latex: list[str] = Field(default_factory=list)
    datasets_mentioned: list[str] = Field(default_factory=list)
    metrics_mentioned: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    references_in_chunk: list[str] = Field(default_factory=list)
    position_in_paper: float = 0.0  # 0.0 to 1.0
    ontology_nodes: list[str] = Field(default_factory=list)


# ── Knowledge Graph Models ───────────────────────────────────────────

class KGNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: str  # Paper, Method, Dataset, Metric, Concept, Claim, Result
    properties: dict[str, Any] = Field(default_factory=dict)


class KGEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: str  # CITES, PROPOSES, USES, REPORTS, EXTENDS, COMPETES_WITH
    properties: dict[str, Any] = Field(default_factory=dict)


# ── Context Engineering Models ───────────────────────────────────────

class TokenBudgetAllocation(BaseModel):
    system_prompt: int = 2000
    task_description: int = 1000
    working_memory: int = 4000
    retrieved_context: int = 60000
    conversation_history: int = 8000
    ontology_context: int = 2000
    output_budget: int = 16000
    safety_margin: int = 5000


class WorkingMemory(BaseModel):
    current_task: str = ""
    key_decisions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class GlobalDecision(BaseModel):
    decision: str
    reason: str
    agent: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class CrossReferences(BaseModel):
    method_name: str = ""
    equation_registry: dict[str, str] = Field(default_factory=dict)
    notation_table: dict[str, str] = Field(default_factory=dict)
    figure_registry: dict[str, str] = Field(default_factory=dict)
    table_registry: dict[str, str] = Field(default_factory=dict)


class AgentOutputRecord(BaseModel):
    status: TaskStatus = TaskStatus.PENDING
    summary: str = ""
    full_output: Any = None
    timestamp: str = ""
    token_count: int = 0


class EpisodicMemory(BaseModel):
    """The Shared Blackboard — persists across all agent calls in a run."""
    paper_identity: dict[str, Any] = Field(default_factory=dict)
    agent_outputs: dict[str, AgentOutputRecord] = Field(default_factory=dict)
    global_decisions: list[GlobalDecision] = Field(default_factory=list)
    cross_references: CrossReferences = Field(default_factory=CrossReferences)
    consistency_invariants: list[str] = Field(default_factory=list)


# ── Retrieval Models ─────────────────────────────────────────────────

class RetrievalQuery(BaseModel):
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 10
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID


class RetrievalResult(BaseModel):
    chunks: list[PaperChunk] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    strategy_used: RetrievalStrategy = RetrievalStrategy.HYBRID
    confidence: str = "medium"  # high, medium, low


# ── Agent Output Models ──────────────────────────────────────────────

class ContributionClaim(BaseModel):
    claim: str
    contribution_type: ContributionType = ContributionType.HYBRID
    novelty_justification: str = ""
    supporting_evidence_needed: str = ""
    ontology_position: str = ""
    confidence: float = 0.0


class ContributionsOutput(BaseModel):
    contributions: list[ContributionClaim] = Field(default_factory=list)
    positioning_statement: str = ""
    gap_analysis: list[str] = Field(default_factory=list)


class ProblemFormulationOutput(BaseModel):
    formal_definition: str = ""
    input_space: str = ""
    output_space: str = ""
    hypothesis_space: str = ""
    loss_function: str = ""
    assumptions: list[str] = Field(default_factory=list)
    objective: str = ""
    notation_additions: dict[str, str] = Field(default_factory=dict)


class LiteratureReviewOutput(BaseModel):
    narrative: str = ""
    taxonomy: dict[str, list[str]] = Field(default_factory=dict)
    citations_used: list[str] = Field(default_factory=list)
    gaps_identified: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class MathFormulationOutput(BaseModel):
    objective_function: str = ""
    constraints: list[str] = Field(default_factory=list)
    optimization_strategy: str = ""
    equations: dict[str, str] = Field(default_factory=dict)  # label → latex
    proofs: list[str] = Field(default_factory=list)
    complexity_analysis: str = ""
    verification_results: dict[str, bool] = Field(default_factory=dict)


class ArchitectureOutput(BaseModel):
    model_structure: str = ""
    components: list[dict[str, Any]] = Field(default_factory=list)
    data_flow: str = ""
    algorithm_steps: list[str] = Field(default_factory=list)
    complexity: dict[str, str] = Field(default_factory=dict)


class ExperimentDesign(BaseModel):
    research_questions: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    setup_description: str = ""


class ExperimentResult(BaseModel):
    table_data: list[dict[str, Any]] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)  # paths to generated figures
    analysis: str = ""
    statistical_tests: list[dict[str, Any]] = Field(default_factory=list)
    is_simulated: bool = True


class SectionOutput(BaseModel):
    """Generic output for any text section."""
    section_name: str
    content: str
    citations_used: list[str] = Field(default_factory=list)
    confidence: float = 0.0


# ── Quality Control Models ───────────────────────────────────────────

class ProvenanceRecord(BaseModel):
    sentence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    provenance_type: ProvenanceType = ProvenanceType.ORIGINAL
    source_paper: str = ""
    source_section: str = ""
    source_sentence: str = ""
    similarity_score: float = 0.0
    transformation: str = ""
    cove_verified: bool = False
    verification_question: str = ""
    verification_answer: str = ""
    confidence: float = 0.0
    citation_key: str = ""


class VerificationResult(BaseModel):
    question: str
    draft_says: str
    source_says: str
    source_id: str = ""
    consistent: bool = True
    action: str = ""  # CORRECT, OK


class VerifiedSection(BaseModel):
    text: str
    corrections_made: int = 0
    verification_log: list[VerificationResult] = Field(default_factory=list)
    confidence: float = 1.0


class NoveltyReport(BaseModel):
    textual_novelty: float = 0.0
    structural_novelty: float = 0.0
    contribution_novelty: float = 0.0
    overall: float = 0.0
    overlap_regions: list[dict[str, Any]] = Field(default_factory=list)


class ConsistencyIssue(BaseModel):
    rule: str
    details: str


class SectionConfidence(BaseModel):
    confidence: float = 0.0
    source_coverage: float = 0.0
    cove_score: float = 0.0
    note: str = ""


class ConfidenceReport(BaseModel):
    section_scores: dict[str, SectionConfidence] = Field(default_factory=dict)
    overall_confidence: float = 0.0
    hallucination_risk: str = "unknown"
    recommendation: str = ""


# ── Orchestrator Models ──────────────────────────────────────────────

class TaskNode(BaseModel):
    task_id: str
    agent_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = Field(default_factory=list)
    output: Any = None
    error: str = ""
    retries: int = 0
    max_retries: int = 3
    started_at: str = ""
    completed_at: str = ""


class DAGState(BaseModel):
    tasks: list[TaskNode] = Field(default_factory=list)
    current_phase: str = ""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Output Models ────────────────────────────────────────────────────

class GeneratedPaper(BaseModel):
    title: str
    sections: dict[str, str] = Field(default_factory=dict)  # section_name → content
    bibtex: str = ""
    latex_source: str = ""
    pdf_path: str = ""
    confidence_report: ConfidenceReport = Field(default_factory=ConfidenceReport)
    provenance_records: list[ProvenanceRecord] = Field(default_factory=list)
    novelty_report: NoveltyReport = Field(default_factory=NoveltyReport)
    generation_log: dict[str, Any] = Field(default_factory=dict)
