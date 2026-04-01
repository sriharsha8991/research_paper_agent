"""Configuration management for the article generator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """LLM model routing configuration (MoA — §5.3)."""
    primary: str = "gemini-2.5-pro"
    reasoning: str = "gemini-2.5-pro"
    fast: str = "gemini-2.5-flash"


@dataclass
class PathConfig:
    """File system paths (§10.1)."""
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    config_dir: Path = field(default_factory=lambda: Path("./config"))

    @property
    def papers_raw(self) -> Path:
        return self.data_dir / "papers_raw"

    @property
    def parsed_chunks(self) -> Path:
        return self.data_dir / "parsed" / "chunks"

    @property
    def knowledge_store(self) -> Path:
        return self.data_dir / "knowledge_store"

    @property
    def memory_working(self) -> Path:
        return self.data_dir / "memory" / "working"

    @property
    def memory_episodic(self) -> Path:
        return self.data_dir / "memory" / "episodic"

    @property
    def memory_long_term(self) -> Path:
        return self.data_dir / "memory" / "long_term"

    @property
    def agent_outputs(self) -> Path:
        return self.data_dir / "intermediate" / "agent_outputs"

    @property
    def verification_logs(self) -> Path:
        return self.data_dir / "intermediate" / "verification_logs"

    @property
    def provenance(self) -> Path:
        return self.data_dir / "intermediate" / "provenance"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def latex_dir(self) -> Path:
        return self.data_dir / "latex"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def ontology_dir(self) -> Path:
        return self.config_dir / "ontology"

    def ensure_all(self) -> None:
        """Create all directories if they don't exist."""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            val = getattr(self, attr_name)
            if isinstance(val, Path):
                val.mkdir(parents=True, exist_ok=True)


@dataclass
class RetrievalConfig:
    """Retrieval settings (§7)."""
    default_top_k: int = 10
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3
    relevance_threshold_high: float = 0.7
    relevance_threshold_low: float = 0.4
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "research_chunks"
    embedding_model: str = "all-MiniLM-L6-v2"  # Fallback; prefer SPECTER2
    embedding_dim: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class AgentConfig:
    """Per-agent settings (§5.4)."""
    max_retries: int = 3
    timeout_seconds: int = 300
    min_confidence: float = 0.7


@dataclass
class QualityConfig:
    """Quality control thresholds (§8, §9)."""
    cove_enabled: bool = True
    adversarial_probing_enabled: bool = True
    reflexion_max_iterations: int = 3
    min_novelty_score: float = 0.5
    min_cove_pass_rate: float = 0.9
    max_ai_detection_score: float = 0.3
    min_burstiness: float = 25.0


@dataclass
class HITLConfig:
    """Human-in-the-loop gate configuration (§5.5)."""
    after_requirements: bool = False
    after_literature_review: bool = False
    after_experiment_design: bool = False
    after_full_draft: bool = False


@dataclass
class AppConfig:
    """Root configuration object."""
    models: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    hitl: HITLConfig = field(default_factory=HITLConfig)

    # API keys loaded from environment
    google_api_key: str = ""
    semantic_scholar_api_key: str = ""

    def __post_init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY", self.google_api_key)
        self.semantic_scholar_api_key = os.getenv(
            "SEMANTIC_SCHOLAR_API_KEY", self.semantic_scholar_api_key
        )

        primary = os.getenv("PRIMARY_MODEL")
        if primary:
            self.models.primary = primary
        reasoning = os.getenv("REASONING_MODEL")
        if reasoning:
            self.models.reasoning = reasoning
        fast = os.getenv("FAST_MODEL")
        if fast:
            self.models.fast = fast

        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            self.retrieval.qdrant_url = qdrant_url

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> AppConfig:
        """Load config from YAML file, with env overrides."""
        config = cls()

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            if "models" in data:
                config.models = ModelConfig(**data["models"])
            if "retrieval" in data:
                config.retrieval = RetrievalConfig(**data["retrieval"])
            if "quality" in data:
                config.quality = QualityConfig(**data["quality"])
            if "hitl" in data:
                config.hitl = HITLConfig(**data["hitl"])
            if "paths" in data:
                p = data["paths"]
                config.paths = PathConfig(
                    data_dir=Path(p.get("data_dir", "./data")),
                    config_dir=Path(p.get("config_dir", "./config")),
                )

        config.paths.ensure_all()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Alias for load()."""
        return cls.load(path)

    def model_dump_json(self, indent: int = 2) -> str:
        """Serialize config to JSON string."""
        import json
        import dataclasses

        def _convert(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        return json.dumps(_convert(self), indent=indent)
