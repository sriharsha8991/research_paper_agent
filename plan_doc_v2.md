# Research Article Generator — Enterprise-Grade Architecture v2

## Revision Context

**Supersedes:** `plan_doc.md` (v1 first draft)
**Authored:** April 2026
**Domain Focus:** AI / ML / Deep Learning research

---

# 0. Critical Gap Analysis of v1

Before the full specification, these are the **24 architectural gaps** identified in v1 that this document addresses:

| # | Gap | Impact | v2 Resolution |
|---|-----|--------|---------------|
| 1 | No Context Engineering strategy | Agents lose critical details across long pipelines | §4 — Hierarchical Context Architecture |
| 2 | No Knowledge Graph | Can't do relational reasoning across papers | §3.4 — Research Knowledge Graph (RKG) |
| 3 | No Chain-of-Verification | Hallucinations pass unchecked | §8.1 — CoVe Pipeline |
| 4 | No Self-RAG / Corrective RAG | Retrieves irrelevant docs blindly | §7.2 — Adaptive Retrieval with CRAG |
| 5 | No Multi-Hop Reasoning | Literature review is flat, not connective | §6.3 — Multi-Hop Synthesis Engine |
| 6 | No Agent Memory Architecture | Context vanishes between agent calls | §4.3 — Three-Tier Memory System |
| 7 | No Reflexion / Self-Critique | No iterative self-improvement on draft | §8.4 — Reflexion Loops |
| 8 | Weak Orchestration (static DAG) | Can't recover from failures or re-plan | §5.1 — Dynamic DAG with Re-Planning |
| 9 | No AI/ML Domain Ontology | Domain knowledge not structured | §3.5 — AI/ML/DL Ontology Layer |
| 10 | No Provenance Tracking | Can't trace text → source sentences | §8.2 — Full Provenance Chain |
| 11 | No Model Routing / MoA | Single LLM, no specialization | §5.3 — Mixture-of-Agents Routing |
| 12 | No Evaluation Framework | No automated quality benchmarks | §9 — Automated Evaluation Suite |
| 13 | No Token Budget Management | Context windows overflow silently | §4.2 — Token Budget Accounting |
| 14 | Limited Paper Sources | Only arXiv + Semantic Scholar | §3.2 — Extended Source Registry |
| 15 | No Incremental Learning | System can't improve across runs | §10.4 — Experience Replay Store |
| 16 | No Human-in-the-Loop | Zero review checkpoints | §5.5 — Configurable HITL Gates |
| 17 | No Caching / Memoization | Repeats expensive LLM calls | §10.2 — Semantic Cache Layer |
| 18 | No Fault Tolerance | Single failure kills pipeline | §10.3 — Circuit Breakers & Retry |
| 19 | Weak Math Reasoning | LLM does symbolic math alone | §6.4 — Tool-Integrated Math (ToRA-style) |
| 20 | No Cross-Agent Communication | Agents work in silos | §5.2 — Shared Blackboard Protocol |
| 21 | No Structured Output Contracts | Agent outputs are unvalidated blobs | §5.4 — Typed Agent Contracts |
| 22 | No Adversarial Self-Challenge | System never stress-tests its own claims | §8.7 — Adversarial Gap Probing |
| 23 | Surface-Level Reasoning Only | Paraphrases text instead of understanding principles | §8.8 — Deep Principled Reasoning (3-Layer Backprop) |
| 24 | AI-Detectable Writing Style | Output reads like LLM-generated text | §8.9 — Humanized Writing Engine |

---

# 1. System Overview

## 1.1 Objective

```
Title (+ optional Abstract / Research Direction)
    → Fully structured, citation-grounded, novelty-verified
      research paper (LaTeX + PDF)
      with complete provenance trail and confidence scores
```

## 1.2 Core Principles (Expanded from v1)

| Principle | v1 | v2 Enhancement |
|-----------|----|----|
| Retrieval-grounded | Basic RAG | **Self-RAG + Corrective RAG + Multi-Hop** |
| Originality-aware | Similarity check | **Novelty Score + Structural Diff + Contribution Mapper** |
| Hypothesis-driven | Mentioned | **Formalized H₀/H₁ with statistical test planning** |
| Multi-agent orchestration | Static DAG | **Dynamic DAG + MoA + Shared Blackboard** |
| Iterative validation | Single pass | **Reflexion loops + Chain-of-Verification + HITL gates** |
| **Context-engineered** | ❌ Missing | **Hierarchical context with budget accounting** |
| **Provenance-traced** | ❌ Missing | **Every claim → source sentence → paper → section** |
| **Domain-specialized** | ❌ Missing | **AI/ML/DL ontology + taxonomy-aware retrieval** |
| **Fault-tolerant** | ❌ Missing | **Circuit breakers, retries, graceful degradation** |

## 1.3 Non-Goals (Critical Boundaries)

- Does **NOT** fabricate experimental results on real datasets
- Does **NOT** claim to replace peer review
- Does **NOT** submit papers to venues autonomously
- Simulated experiments are **clearly labeled** as such
- All generated content carries **confidence metadata**

---

# 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│  Title + Abstract + Research Direction + Constraints                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│              PHASE 1: RESEARCH CORPUS BUILDER (RCB)                 │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────────────────┐ │
│  │Req Agent │→│Discovery  │→│Ingestion │→│Knowledge Store        │ │
│  │          │ │Agent      │ │Pipeline  │ │(Vector DB + KG + Meta)│ │
│  └──────────┘ └───────────┘ └──────────┘ └───────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│           CONTEXT ENGINEERING LAYER (NEW in v2)                     │
│  ┌──────────────┐ ┌────────────┐ ┌────────────────────────────┐    │
│  │Token Budget  │ │Context     │ │Three-Tier Memory           │    │
│  │Accountant    │ │Compressor  │ │(Working / Episodic / Long) │    │
│  └──────────────┘ └────────────┘ └────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│              PHASE 2: RESEARCH PAPER GENERATOR (RPG)                │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ ORCHESTRATOR (Dynamic DAG + Shared Blackboard)          │       │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌──────┐    │       │
│  │  │Contr│ │Prob │ │LitRv│ │Math │ │Arch │ │Exper │    │       │
│  │  │Agent│ │Form │ │Agent│ │Agent│ │Agent│ │Agent │    │       │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └──────┘    │       │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌──────┐ ┌──────┐           │       │
│  │  │Ablat│ │Reslt│ │Disc │ │Limit │ │Appndx│           │       │
│  │  │Agent│ │Analy│ │Agent│ │Ethics│ │Agent │           │       │
│  │  └─────┘ └─────┘ └─────┘ └──────┘ └──────┘           │       │
│  └─────────────────────────────────────────────────────────┘       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│           QUALITY CONTROL & VERIFICATION LAYER                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────┐  │
│  │CoVe      │ │Provenance│ │Novelty   │ │Consistency│ │Reflexion│  │
│  │Pipeline  │ │Tracker   │ │Engine    │ │Validator  │ │Loop    │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘ └────────┘  │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐    │
│  │Adversarial Gap   │ │3-Layer Principled│ │Humanized Writing │    │
│  │Prober (Red/Blue) │ │Reasoner (Backprop│ │Engine (Anti-AI)  │    │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                    OUTPUT COMPOSITION                                │
│  LaTeX Composer → Reference Manager → PDF Compilation               │
│  + Provenance Report + Confidence Dashboard                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

# 3. Phase 1 — Research Corpus Builder (RCB)

## 3.1 Requirement Gathering Agent (Enhanced)

### Input (Expanded)

```json
{
  "title": "string (mandatory)",
  "abstract": "string (optional)",
  "research_direction": "string (optional) — e.g., 'efficiency', 'accuracy', 'scalability'",
  "target_venue_style": "NeurIPS | ICLR | ICML | ACL | CVPR | AAAI | journal",
  "constraints": {
    "max_pages": 10,
    "must_cite": ["paper_id_1", "paper_id_2"],
    "exclude_methods": [],
    "time_horizon": "2020-2026"
  }
}
```

### Responsibilities (Enhanced)

1. **Intent Decomposition** — Break title into:
   - Primary research question
   - Secondary questions
   - Methodological scope
   - Expected contribution type (theoretical / empirical / both)

2. **Taxonomy-Aware Query Expansion** — Using the AI/ML ontology (§3.5):
   - Map title concepts to ontology nodes
   - Expand queries along ontology edges (hypernyms, related methods, competing approaches)
   - Generate **negative queries** (what this paper is NOT about, for contrastive retrieval)

3. **Domain Classification** — Multi-label classification into:
   ```
   Primary: {NLP, CV, RL, Optimization, Theory, Systems, ...}
   Method: {Transformer, GAN, Diffusion, Graph Neural Net, ...}
   Task: {Classification, Generation, Detection, Reasoning, ...}
   ```

### Output Contract (Typed)

```json
{
  "primary_question": "string",
  "secondary_questions": ["string"],
  "positive_queries": ["string — min 10, max 30"],
  "negative_queries": ["string — min 3"],
  "domain_labels": {
    "primary": ["string"],
    "method": ["string"],
    "task": ["string"]
  },
  "contribution_type": "theoretical | empirical | hybrid",
  "target_papers": {"min": 30, "max": 80},
  "time_horizon": {"start": 2020, "end": 2026},
  "venue_style": "string"
}
```

---

## 3.2 Paper Discovery Agent (Extended Sources)

### Source Registry (v1 had only arXiv + Semantic Scholar)

| Source | Type | Priority | Rate Limit Strategy |
|--------|------|----------|-------------------|
| **Semantic Scholar API** | Metadata + citations + abstracts | P0 | 100 req/5min, exponential backoff |
| **arXiv API** | PDFs + metadata | P0 | Respect robots.txt, batch download |
| **OpenReview API** | Peer reviews + scores + papers | P1 | Rate-limited, cache aggressively |
| **DBLP** | Bibliographic metadata | P1 | Bulk XML dump preferred |
| **Papers With Code** | Benchmarks + code links | P2 | API with caching |
| **Google Scholar** | Citation counts + related | P2 | SerpAPI proxy (no direct scraping) |
| **ACL Anthology** | NLP-specific papers | P2 | Bulk download |

### Discovery Strategy (Multi-Phase)

```
Phase 1: Seed Discovery
  → Query all sources with positive_queries
  → Collect top 200 candidates (deduplicated by DOI/arXiv ID)

Phase 2: Citation Graph Expansion
  → For top 50 seed papers, fetch their references + citations
  → Identify "hub papers" (cited by many seeds)
  → Add hub papers to corpus

Phase 3: Recency Boost
  → Re-query with time filter (last 12 months)
  → Surface very recent work that may lack citation signal

Phase 4: Contrastive Discovery
  → Query with negative_queries
  → Identify papers to position AGAINST in related work
```

### Ranking Function (Multi-Signal)

```python
score(paper) = (
    w1 * citation_velocity(paper)      # citations/year, normalized
  + w2 * venue_quality(paper.venue)     # tier ranking from CORE/CSRankings
  + w3 * semantic_relevance(paper, query)  # embedding cosine similarity
  + w4 * recency_bonus(paper.year)      # exponential decay favoring recent
  + w5 * author_authority(paper.authors) # h-index normalized
  + w6 * code_available(paper)          # binary, reward reproducibility
)
# w1..w6 are configurable per run
```

### Output

```json
{
  "corpus": [
    {
      "paper_id": "2301.01234",
      "source": "arxiv",
      "title": "...",
      "year": 2023,
      "venue": "NeurIPS",
      "citation_count": 150,
      "relevance_score": 0.92,
      "role": "seed | hub | recent | contrastive",
      "pdf_path": "/papers_raw/2301.01234.pdf",
      "has_code": true,
      "code_url": "https://github.com/..."
    }
  ],
  "total_papers": 65,
  "coverage_report": {
    "method_coverage": 0.85,
    "temporal_coverage": 0.90,
    "venue_diversity": 0.75
  }
}
```

---

## 3.3 Paper Ingestion Pipeline (Enhanced)

### Step 1: Parsing (Production-Grade)

| Parser | Use Case | Fallback |
|--------|----------|----------|
| **GROBID** | Primary — structured XML from PDF | → Nougat |
| **Nougat** | ML-based PDF→Markdown (equations) | → PyMuPDF |
| **PyMuPDF / pdfplumber** | Table extraction, layout analysis | → Manual chunk |
| **LaTeX source** | Direct from arXiv when available | Best quality path |

**Extraction targets:**
- Full text segmented by section headers
- Equations (LaTeX source preserved, not images)
- Tables (structured as DataFrames)
- Figures (captions extracted, images stored)
- Algorithm pseudocode blocks
- Footnotes and appendix material

### Step 2: Smart Chunking (Section-Aware + Overlap)

```python
CHUNK_TYPES = {
    "abstract_chunk":      {"max_tokens": 512,  "overlap": 0},
    "introduction_chunk":  {"max_tokens": 768,  "overlap": 64},
    "method_chunk":        {"max_tokens": 1024, "overlap": 128},
    "math_chunk":          {"max_tokens": 512,  "overlap": 64},
    "result_chunk":        {"max_tokens": 768,  "overlap": 64},
    "table_chunk":         {"max_tokens": 512,  "overlap": 0},
    "discussion_chunk":    {"max_tokens": 768,  "overlap": 64},
    "algorithm_chunk":     {"max_tokens": 512,  "overlap": 0},
    "related_work_chunk":  {"max_tokens": 768,  "overlap": 128},
}
```

**Chunking rules:**
- Never split mid-equation
- Never split mid-table
- Preserve paragraph boundaries
- Include section header as metadata prefix in every chunk
- Cross-reference chunks maintain bidirectional links

### Step 3: Metadata Enrichment (Deep)

```json
{
  "chunk_id": "uuid",
  "paper_id": "2301.01234",
  "title": "Attention Is All You Need",
  "authors": ["Vaswani, A.", "..."],
  "year": 2017,
  "venue": "NeurIPS",
  "section": "methodology",
  "subsection": "multi-head attention",
  "chunk_type": "method_chunk",
  "tags": ["transformer", "attention", "self-attention"],
  "has_equation": true,
  "equations_latex": ["\\text{Attention}(Q,K,V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V"],
  "datasets_mentioned": ["WMT14", "Penn Treebank"],
  "metrics_mentioned": ["BLEU", "perplexity"],
  "claims": [
    {
      "claim_text": "Multi-head attention allows the model to jointly attend...",
      "claim_type": "methodological",
      "confidence": "high"
    }
  ],
  "references_in_chunk": ["bahdanau2015", "luong2015"],
  "position_in_paper": 0.45,
  "ontology_nodes": ["transformer.attention.multi_head"]
}
```

### Step 4: Multi-Modal Knowledge Store

#### 4a. Vector Database (Enhanced)

- **Primary Embeddings:** `SPECTER2` (scientific paper embeddings, outperforms SciBERT on retrieval)
- **Secondary Embeddings:** `E5-mistral-7b-instruct` (for query-document matching)
- **Vector DB:** Qdrant (confirmed) with:
  - Dense index (HNSW) for semantic search
  - Sparse index (BM25 via SPLADE) for keyword matching
  - **Hybrid search** = dense + sparse fusion with RRF (Reciprocal Rank Fusion)
  - Metadata filtering on all enriched fields
  - Multi-vector per chunk (title-abstract vector + content vector)

```json
{
  "collection_config": {
    "vectors": {
      "content": {"size": 768, "distance": "Cosine"},
      "title_abstract": {"size": 768, "distance": "Cosine"}
    },
    "sparse_vectors": {
      "bm25": {"modifier": "idf"}
    },
    "payload_indexes": [
      {"field": "year", "type": "integer"},
      {"field": "section", "type": "keyword"},
      {"field": "chunk_type", "type": "keyword"},
      {"field": "venue", "type": "keyword"},
      {"field": "ontology_nodes", "type": "keyword"}
    ]
  }
}
```

#### 4b. Research Knowledge Graph (NEW — v2)

A **Neo4j** or **NetworkX** graph that captures structured relationships:

```
NODES:
  Paper       (id, title, year, venue, citation_count)
  Author      (name, h_index, affiliations)
  Method      (name, category, year_introduced)
  Dataset     (name, domain, size)
  Metric      (name, type, higher_is_better)
  Concept     (name, ontology_path)
  Claim       (text, paper_id, section, type)
  Result      (paper_id, dataset, metric, value, conditions)

EDGES:
  Paper  -[CITES]→         Paper
  Paper  -[PROPOSES]→       Method
  Paper  -[USES]→           Dataset
  Paper  -[REPORTS]→        Result
  Paper  -[AUTHORED_BY]→    Author
  Method -[EXTENDS]→        Method
  Method -[COMPETES_WITH]→  Method
  Method -[APPLIED_TO]→     Dataset
  Result -[MEASURED_BY]→    Metric
  Claim  -[SUPPORTED_BY]→   Result
  Claim  -[FROM_PAPER]→     Paper
  Concept-[RELATED_TO]→     Concept
  Concept-[SUBCLASS_OF]→    Concept
```

**Why this matters:** The Knowledge Graph enables:
- Multi-hop reasoning: "What methods extend Transformers AND were tested on ImageNet?"
- Gap detection: "Which dataset + metric combinations are under-explored?"
- Lineage tracking: "Trace the evolution of attention mechanisms 2017→2026"
- Contradiction detection: "Paper A claims X outperforms Y, Paper B claims opposite"

#### 4c. Metadata Store

- **SQLite / PostgreSQL** for structured queries
- Stores: paper metadata, processing status, chunk mappings, provenance logs
- Enables: "Which papers from ICLR 2025 discuss diffusion models?"

---

## 3.4 AI/ML/DL Domain Ontology (NEW — v2)

A structured taxonomy that gives the system PhD-level domain awareness:

```
AI_ML_DL_Ontology
├── Learning_Paradigms
│   ├── Supervised_Learning
│   │   ├── Classification
│   │   ├── Regression
│   │   └── Structured_Prediction
│   ├── Unsupervised_Learning
│   │   ├── Clustering
│   │   ├── Dimensionality_Reduction
│   │   └── Density_Estimation
│   ├── Self_Supervised_Learning
│   │   ├── Contrastive (SimCLR, MoCo, CLIP)
│   │   ├── Masked_Prediction (BERT, MAE)
│   │   └── Generative (GPT, autoregressive)
│   ├── Reinforcement_Learning
│   │   ├── Model_Free (PPO, SAC, DQN)
│   │   ├── Model_Based (Dreamer, MuZero)
│   │   └── RLHF
│   └── Meta_Learning
│       ├── MAML
│       ├── Prototypical_Networks
│       └── In_Context_Learning
├── Architectures
│   ├── Transformers
│   │   ├── Encoder_Only (BERT, RoBERTa)
│   │   ├── Decoder_Only (GPT, LLaMA)
│   │   ├── Encoder_Decoder (T5, BART)
│   │   ├── Vision_Transformer (ViT, DeiT, Swin)
│   │   └── Efficient_Transformers (Linformer, Performer, Flash)
│   ├── CNNs
│   │   ├── ResNet, EfficientNet, ConvNeXt
│   │   └── Object_Detection (YOLO, DETR)
│   ├── GNNs
│   │   ├── GCN, GAT, GraphSAGE
│   │   └── Message_Passing
│   ├── Generative_Models
│   │   ├── GANs (StyleGAN, BigGAN)
│   │   ├── VAEs
│   │   ├── Diffusion_Models (DDPM, Stable Diffusion, DALL-E)
│   │   ├── Flow_Models (Normalizing Flows, Flow Matching)
│   │   └── Autoregressive (PixelCNN, WaveNet)
│   ├── State_Space_Models (Mamba, S4, RWKV)
│   └── Mixture_of_Experts (Switch, GShard)
├── Training_Techniques
│   ├── Optimization (Adam, LAMB, Sharpness-Aware)
│   ├── Regularization (Dropout, Weight Decay, Mixup)
│   ├── Normalization (BatchNorm, LayerNorm, RMSNorm)
│   ├── Scaling_Laws (Chinchilla, Kaplan)
│   ├── Distributed_Training (DDP, FSDP, Pipeline)
│   └── Quantization (INT8, GPTQ, QLoRA)
├── Evaluation
│   ├── NLP_Metrics (BLEU, ROUGE, BERTScore, MMLU)
│   ├── CV_Metrics (mAP, FID, IS, LPIPS)
│   ├── RL_Metrics (Return, Regret, Sample_Efficiency)
│   └── Statistical_Tests (t-test, Wilcoxon, Bootstrap CI)
├── Applications
│   ├── NLP (Translation, Summarization, QA, Dialogue)
│   ├── Computer_Vision (Detection, Segmentation, Generation)
│   ├── Multimodal (VLMs, Text-to-Image, Video)
│   ├── Robotics
│   ├── Scientific_Discovery (Drug, Material, Protein)
│   └── Code_Generation
└── Safety_Alignment
    ├── RLHF, DPO, Constitutional_AI
    ├── Jailbreaking, Red_Teaming
    ├── Hallucination_Detection
    └── Interpretability (SHAP, Attention, Mechanistic)
```

**Usage across system:**
- Requirement Agent uses ontology for query expansion
- Discovery Agent maps papers to ontology nodes
- Literature Review Agent uses ontology for structured synthesis
- Novelty Engine checks contribution against ontology to assess gap coverage

---

# 4. Context Engineering Layer (NEW — v2)

> *This is the single most critical addition. Without disciplined context management, the system will hallucinate, lose details, and produce incoherent papers.*

## 4.1 The Problem

A research paper generation pipeline involves **50+ LLM calls** across **10+ agents**. Each call has a finite context window (128K–200K tokens). The full corpus + KG + intermediate outputs can exceed **10M tokens**. Without explicit context engineering, agents will:

- Lose critical details from earlier agents
- Contradict previous sections
- Hallucinate when context is insufficient
- Waste tokens on irrelevant information

## 4.2 Token Budget Accounting System

Every LLM call has a **token budget** that is explicitly allocated:

```python
class TokenBudget:
    def __init__(self, model_context_limit: int = 128000):
        self.limit = model_context_limit
        self.allocations = {
            "system_prompt":        2000,   # Agent instructions
            "task_description":     1000,   # Current task specifics
            "working_memory":       4000,   # Key findings so far
            "retrieved_context":    60000,  # RAG chunks (dynamically sized)
            "conversation_history": 8000,   # Previous agent outputs (compressed)
            "ontology_context":     2000,   # Relevant ontology subtree
            "output_budget":        16000,  # Reserved for generation
            "safety_margin":        5000,   # Buffer for tokenizer variance
        }
        # Remaining: ~30K for additional context as needed

    def allocate(self, category: str, content: str) -> str:
        """Truncate or compress content to fit budget."""
        max_tokens = self.allocations[category]
        if count_tokens(content) > max_tokens:
            return self.compress(content, max_tokens)
        return content

    def compress(self, content: str, target_tokens: int) -> str:
        """Multi-strategy compression:
        1. Extractive summarization (keep key sentences)
        2. Entity-preserving compression (keep names, numbers, equations)
        3. Hierarchical truncation (keep headers + first sentences)
        """
        ...
```

### Budget Allocation by Agent Type

| Agent | Retrieved Context | Working Memory | History | Output |
|-------|------------------|----------------|---------|--------|
| Literature Review | 60K (heaviest reader) | 4K | 8K | 16K |
| Math Formulation | 20K | 4K | 4K | 8K |
| Experimentation | 30K | 8K | 8K | 16K |
| Results Analysis | 40K | 8K | 12K | 8K |
| Discussion | 30K | 8K | 16K | 12K |

## 4.3 Three-Tier Memory System

Inspired by human cognitive architecture (Lilian Weng's framework):

### Tier 1: Working Memory (In-Context)

The **active scratchpad** within each LLM call:

```json
{
  "current_task": "Write methodology section",
  "key_decisions_so_far": [
    "Architecture: Transformer-based with sparse attention",
    "Training: Self-supervised pretraining + task-specific fine-tuning",
    "Baseline methods: [X, Y, Z]"
  ],
  "constraints": [
    "Must compare against at least 3 baselines",
    "Must include complexity analysis"
  ],
  "open_questions": [
    "Which specific sparse attention variant?",
    "How to handle variable-length inputs?"
  ]
}
```

- **Capacity:** ~4K tokens
- **Lifetime:** Single agent call
- **Management:** Explicitly constructed by the Orchestrator before each call

### Tier 2: Episodic Memory (Cross-Agent Shared State)

The **Shared Blackboard** — a structured JSON document that persists across all agent calls:

```json
{
  "paper_identity": {
    "title": "...",
    "thesis_statement": "...",
    "contribution_bullets": ["..."],
    "target_venue": "NeurIPS"
  },
  "agent_outputs": {
    "requirement_agent": { "status": "complete", "summary": "...", "full_path": "..." },
    "contributions_agent": { "status": "complete", "summary": "...", "full_path": "..." },
    "literature_review": { "status": "in_progress", "summary": "...", "full_path": "..." },
    "math_agent": { "status": "pending" }
  },
  "global_decisions": [
    {"decision": "Use Vision Transformer as backbone", "reason": "...", "agent": "arch_agent", "timestamp": "..."},
    {"decision": "Compare on ImageNet + COCO", "reason": "...", "agent": "experiment_agent", "timestamp": "..."}
  ],
  "cross_references": {
    "method_name": "SparseViT",
    "equation_registry": {"eq:attention": "\\text{Eq. 1}", "eq:loss": "\\text{Eq. 2}"},
    "notation_table": {"x": "input", "y": "output", "\\theta": "parameters"},
    "figure_registry": {"fig:arch": "Figure 1", "fig:results": "Figure 2"},
    "table_registry": {"tab:main": "Table 1", "tab:ablation": "Table 2"}
  },
  "consistency_invariants": [
    "All sections must use 'SparseViT' (not 'Sparse-ViT' or 'sparse_vit')",
    "Dataset split: 80/10/10 train/val/test",
    "Baseline list is frozen after experiment_agent completes"
  ]
}
```

- **Capacity:** Unlimited (stored on disk)
- **Lifetime:** Entire generation run
- **Access:** Any agent can read; writes go through Orchestrator (serialized)
- **Injection:** Each agent receives a compressed summary (~2-4K tokens) of relevant blackboard state

### Tier 3: Long-Term Memory (Cross-Run Persistent)

Experience from previous paper generations:

```json
{
  "successful_patterns": [
    {
      "domain": "efficient transformers",
      "effective_queries": ["linear attention", "sparse attention mechanism"],
      "high_quality_sources": ["arxiv/2205.14135", "arxiv/2009.06732"],
      "good_structure_templates": ["methodology section from run_042"]
    }
  ],
  "failure_patterns": [
    {
      "issue": "Math agent generated unsound proof",
      "root_cause": "Insufficient retrieval of prior proofs",
      "fix": "Always retrieve 3+ prior related proofs before generating"
    }
  ],
  "domain_calibration": {
    "transformer_papers": {"avg_references": 45, "avg_equations": 12},
    "GAN_papers": {"avg_references": 35, "avg_equations": 8}
  }
}
```

## 4.4 Context Compression Strategies

When content exceeds budget, apply in order:

1. **Extractive Compression** — Keep sentences with highest information density (TF-IDF score relative to task)
2. **Entity-Preserving Summary** — Use LLM to summarize but mandate preserving: paper names, method names, numbers, equations, dataset names
3. **Hierarchical Rollup** — Replace detailed content with section-level summaries; expand only the section relevant to current task
4. **Reference Pointers** — Replace large text blocks with pointers that can be expanded on-demand: `"[See full literature review output at /intermediate/lit_review.json]"`

## 4.5 Context Routing

Not every agent needs the same context. The **Context Router** builds a custom context payload per agent:

```python
CONTEXT_ROUTING = {
    "contributions_agent": {
        "required": ["paper_identity", "requirement_output", "top_5_related_papers"],
        "optional": ["ontology_subtree"],
        "excluded": ["raw_chunks", "experiment_details"]
    },
    "math_agent": {
        "required": ["problem_formulation", "related_equations", "notation_table"],
        "optional": ["architecture_decisions"],
        "excluded": ["literature_review_full", "ethics_output"]
    },
    "discussion_agent": {
        "required": ["all_agent_summaries", "results_tables", "key_claims", "limitations"],
        "optional": ["full_results_analysis"],
        "excluded": ["raw_chunks", "raw_metadata"]
    }
}
```

---

# 5. Phase 2 — Research Paper Generator (RPG) — Orchestration

## 5.1 Dynamic DAG with Re-Planning

Unlike v1's static task list, the Orchestrator maintains a **live DAG** that can be modified during execution:

```python
class DynamicDAG:
    """
    Task graph that supports:
    - Adding tasks at runtime (e.g., agent discovers need for additional analysis)
    - Re-prioritizing tasks based on intermediate results
    - Marking tasks as blocked/failed and triggering recovery
    - Parallel execution of independent tasks
    """

    def plan(self, requirement_output: dict) -> DAG:
        # Initial DAG construction
        tasks = [
            Task("contributions", deps=[]),
            Task("problem_formulation", deps=[]),
            Task("literature_review", deps=[]),
            # These can run in parallel ↑

            Task("math_formulation", deps=["problem_formulation"]),
            Task("architecture_design", deps=["contributions", "math_formulation"]),
            Task("implementation_details", deps=["architecture_design"]),
            Task("experiment_design", deps=["architecture_design", "literature_review"]),
            Task("experiment_execution", deps=["experiment_design"]),
            Task("ablation_study", deps=["experiment_execution"]),
            Task("results_analysis", deps=["experiment_execution", "ablation_study"]),
            Task("discussion", deps=["results_analysis", "literature_review"]),
            Task("limitations_ethics", deps=["results_analysis"]),
            Task("abstract_conclusion", deps=["discussion", "contributions"]),
            Task("appendix", deps=["math_formulation", "experiment_execution"]),
        ]
        return DAG(tasks)

    def re_plan(self, failed_task: str, error: str) -> DAG:
        """Dynamic re-planning on failure."""
        if failed_task == "math_formulation" and "insufficient_context" in error:
            # Insert a retrieval refinement task before retrying
            self.insert_before("math_formulation", Task("math_context_enrichment"))
        elif failed_task == "experiment_execution" and "missing_baseline" in error:
            # Re-run literature review for baseline papers
            self.insert_before("experiment_execution", Task("baseline_discovery"))
        ...
```

### Execution States

```
PENDING → READY → IN_PROGRESS → COMPLETED
                 ↘            ↗
                   FAILED → RECOVERING → READY (retry)
                          ↘
                            SKIPPED (with justification)
```

## 5.2 Shared Blackboard Protocol

All agents communicate through the Episodic Memory blackboard (§4.3 Tier 2), not through direct message passing:

```python
class BlackboardProtocol:
    def publish(self, agent_id: str, key: str, value: Any, summary: str):
        """Agent writes its output to the blackboard."""
        self.blackboard["agent_outputs"][agent_id] = {
            "status": "complete",
            "full_output": value,
            "summary": summary,  # ≤500 tokens, for injection into other agents
            "timestamp": now(),
            "token_count": count_tokens(value)
        }

    def read(self, agent_id: str, keys: list[str]) -> dict:
        """Agent reads specific keys — not the entire blackboard."""
        return {k: self.blackboard[k] for k in keys if k in self.blackboard}

    def publish_decision(self, agent_id: str, decision: str, reason: str):
        """Record a global decision that constrains future agents."""
        self.blackboard["global_decisions"].append({
            "decision": decision,
            "reason": reason,
            "agent": agent_id,
            "timestamp": now()
        })

    def publish_cross_reference(self, ref_type: str, key: str, value: str):
        """Register equations, figures, tables for consistent referencing."""
        self.blackboard["cross_references"][ref_type][key] = value
```

## 5.3 Mixture-of-Agents (MoA) Routing

Different tasks demand different LLM capabilities. Instead of using one model for everything:

```python
MODEL_ROUTING = {
    # Deep reasoning tasks → strongest reasoning model
    "math_formulation":      {"model": "claude-opus-4", "temperature": 0.2},
    "problem_formulation":   {"model": "claude-opus-4", "temperature": 0.3},
    "results_analysis":      {"model": "claude-opus-4", "temperature": 0.3},

    # Creative synthesis tasks → balanced model
    "literature_review":     {"model": "claude-sonnet-4", "temperature": 0.4},
    "discussion":            {"model": "claude-sonnet-4", "temperature": 0.5},
    "contributions":         {"model": "claude-sonnet-4", "temperature": 0.4},

    # Structured generation → fast model with constrained output
    "latex_generation":      {"model": "claude-sonnet-4", "temperature": 0.1},
    "bibtex_generation":     {"model": "claude-haiku-3.5", "temperature": 0.0},
    "metadata_extraction":   {"model": "claude-haiku-3.5", "temperature": 0.0},

    # Verification tasks → high-precision model
    "chain_of_verification": {"model": "claude-opus-4", "temperature": 0.1},
    "consistency_check":     {"model": "claude-sonnet-4", "temperature": 0.1},
}
```

**Critical Ensemble Pattern:** For the most important sections (Abstract, Introduction, Methodology), generate with **2 models independently** and use a third model as an **evaluator-optimizer** to select/merge the best output (per Anthropic's evaluator-optimizer pattern).

## 5.4 Typed Agent Contracts

Every agent has a **typed input/output schema** validated at runtime:

```python
@dataclass
class AgentContract:
    agent_id: str
    input_schema: dict       # JSON Schema
    output_schema: dict      # JSON Schema
    max_retries: int = 3
    timeout_seconds: int = 300
    required_confidence: float = 0.7

    def validate_output(self, output: dict) -> ValidationResult:
        """Validates:
        1. Schema conformance (all required fields present)
        2. Confidence thresholds met
        3. Citation density (claims per citation ratio)
        4. No empty sections
        5. Token count within expected range
        """
        ...
```

Example contract for the Literature Review Agent:

```json
{
  "agent_id": "literature_review",
  "output_schema": {
    "type": "object",
    "required": ["narrative", "taxonomy", "citations_used", "gaps_identified", "confidence"],
    "properties": {
      "narrative": {
        "type": "string",
        "minLength": 2000,
        "maxLength": 15000
      },
      "taxonomy": {
        "type": "object",
        "description": "Papers organized by approach category"
      },
      "citations_used": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 15
      },
      "gaps_identified": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2
      },
      "confidence": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0
      }
    }
  }
}
```

## 5.5 Configurable Human-in-the-Loop (HITL) Gates

Optional review checkpoints that can be enabled/disabled per run:

```python
HITL_GATES = {
    "after_requirement_gathering": {
        "enabled": True,   # Recommended: always review interpretation
        "prompt": "Review the research direction and queries. Approve or modify.",
        "blocking": True   # Blocks pipeline until approved
    },
    "after_contributions":  {"enabled": False, "blocking": False},
    "after_literature_review": {
        "enabled": True,   # Recommended: review paper selection
        "prompt": "Review selected papers and synthesis. Approve or request changes.",
        "blocking": True
    },
    "after_experiment_design": {
        "enabled": True,   # Recommended: approve experimental setup
        "blocking": True
    },
    "after_full_draft": {
        "enabled": True,   # Recommended: final review before LaTeX
        "blocking": True
    }
}
```

---

# 6. Core Sub-Agents (Enhanced Specifications)

## 6.1 Contributions Agent

### Enhanced Behavior

1. **Retrieve** top 10 most related prior works from KG
2. **Map** each prior work's contributions to ontology nodes
3. **Identify gaps** in the ontology coverage
4. **Generate** contribution claims that fill identified gaps
5. **Self-verify:** For each claim, check "is this already done by paper X?"

### Output Contract

```json
{
  "contributions": [
    {
      "claim": "We propose SparseViT, a ...",
      "type": "methodological | theoretical | empirical",
      "novelty_justification": "Unlike [X] which ..., our approach ...",
      "supporting_evidence_needed": "Ablation showing component Y matters",
      "ontology_position": "architectures.transformers.efficient",
      "confidence": 0.85
    }
  ],
  "positioning_statement": "Our work sits at the intersection of ... and ...",
  "gap_analysis": [
    "No prior work combines sparse attention with mixture-of-experts in vision"
  ]
}
```

## 6.2 Problem Formulation Agent

### Enhanced Behavior

1. Retrieve formal definitions from 5+ related papers
2. Build a **formal problem statement** with:
   - Input space $\mathcal{X}$, output space $\mathcal{Y}$
   - Hypothesis space $\mathcal{H}$
   - Loss function $\mathcal{L}$
   - Constraints $\mathcal{C}$
3. Register all notation in the **notation table** (blackboard)
4. Cross-check notation conflicts with retrieved papers

## 6.3 Literature Review Agent (Multi-Hop Synthesis)

### Three-Phase Process (NOT just "retrieve and summarize")

**Phase 1: Taxonomic Retrieval**
```
For each ontology branch relevant to the paper:
  → Retrieve top papers per branch
  → Build a taxonomy tree: {category → [papers]}
```

**Phase 2: Multi-Hop Synthesis**
```
For each paper cluster:
  → Identify common threads
  → Identify disagreements/contradictions
  → Trace method evolution (A → B → C)
  → Use KG edges (EXTENDS, COMPETES_WITH) for structured narrative
```

**Phase 3: Gap-Driven Narrative**
```
Structure the review to LEAD to the paper's contribution:
  → "Previous works have addressed X [cite], Y [cite], but Z remains open"
  → Every paragraph ends by connecting to the gap this paper fills
```

### Anti-Patterns (Explicitly Prohibited)

- ❌ "Paper A does X. Paper B does Y. Paper C does Z." (list summaries)
- ❌ Uncited claims
- ❌ Citing a paper for a claim it doesn't make
- ✅ "While early approaches [1,2] relied on X, recent work [3,4] has shifted to Y, achieving Z% improvement. However, these methods assume A, which limits..."

## 6.4 Mathematical Formulation Agent (Tool-Integrated)

Inspired by **ToRA** (Tool-integrated Reasoning Agent):

```python
class MathAgent:
    """
    Hybrid reasoning: interleave natural language reasoning
    with tool calls for verification.
    """
    tools = {
        "sympy": "symbolic math verification",
        "numpy": "numerical sanity checks",
        "latex_validator": "ensure LaTeX compiles",
        "dimension_checker": "verify tensor dimensions match",
        "complexity_analyzer": "compute Big-O complexity"
    }

    def generate_formulation(self, problem: dict, related_math: list[str]):
        # Step 1: Retrieve prior formulations
        prior = self.retrieve_equations(problem["domain"])

        # Step 2: Reason about objective function
        reasoning = self.llm.generate(
            f"Given problem: {problem['formal_statement']}\n"
            f"Prior approaches used: {prior}\n"
            f"Derive an objective function..."
        )

        # Step 3: Verify with SymPy
        verification = self.tools["sympy"].verify(
            reasoning.equations,
            checks=["dimensional_consistency", "boundary_conditions", "convexity"]
        )

        # Step 4: If verification fails, self-correct
        if not verification.passed:
            reasoning = self.llm.generate(
                f"The following equation has issues: {verification.errors}\n"
                f"Please correct: {reasoning.equations}"
            )

        # Step 5: Register notation
        self.blackboard.publish_cross_reference(
            "equation_registry", reasoning.equation_labels, reasoning.equations
        )

        return reasoning
```

## 6.5 Architecture Design Agent

### Enhanced with Justification Chain

Every architectural decision must include:

```json
{
  "component": "Sparse Attention Module",
  "design_choice": "Top-k sparse attention with learned routing",
  "alternatives_considered": [
    {"name": "Local window attention", "rejected_because": "Cannot capture global dependencies"},
    {"name": "Random sparse attention", "rejected_because": "Not learnable, reduced performance in [cite]"}
  ],
  "justification": "Top-k routing provides adaptive sparsity while maintaining O(n·k) complexity",
  "complexity": {"time": "O(n·k·d)", "space": "O(n·k)"},
  "prior_art": ["cite:reformer2020", "cite:routing_transformer2021"]
}
```

## 6.6 Experimentation Agent (Rigorous Simulation)

### Enhanced Protocol

1. **Design Phase:**
   - Define research questions → experiment mapping
   - Select datasets from KG (prioritize standard benchmarks)
   - Select baselines from KG (must include SOTA)
   - Define evaluation metrics (with statistical test plan)

2. **Simulation Phase:**
   - Generate **plausible** results grounded in retrieved data:
     - Use known SOTA numbers from papers as anchors
     - Apply realistic improvements/regressions
     - Add realistic variance (mean ± std from seed runs)
   - **CRITICAL:** All simulated results carry `"simulated": true` flag

3. **Visualization Phase:**
   ```python
   REQUIRED_FIGURES = {
       "main_results_table": "Comparison against baselines",
       "ablation_table": "Component-wise contribution",
       "convergence_plot": "Training loss/metric curves",
       "qualitative_examples": "Visual examples (if applicable)",
       "efficiency_plot": "FLOPs vs accuracy tradeoff"
   }
   ```

4. **Statistical Rigor Requirements:**
   - All results: mean ± std over N≥3 seeds
   - Significance: paired t-test or Wilcoxon signed-rank
   - Effect size: Cohen's d reported
   - Confidence intervals: 95% CI

## 6.7 Results Analysis Agent (Claim-Result Alignment)

### Verification Protocol

```python
for claim in contributions_agent.claims:
    # Find corresponding experimental evidence
    evidence = find_supporting_results(claim, experiment_results)

    if not evidence:
        flag("UNSUPPORTED_CLAIM", claim)
        trigger_replan("experiment_design")  # May need more experiments

    for result in evidence:
        if result.improvement < claim.implied_improvement:
            flag("OVERCLAIM", claim, result)

        if result.statistical_significance > 0.05:
            flag("INSIGNIFICANT_RESULT", claim, result)
```

---

# 7. Retrieval Orchestrator (Critical Layer — Enhanced)

## 7.1 Retrieval Strategy Selection

Not all queries should use the same retrieval approach:

```python
class RetrievalOrchestrator:
    def retrieve(self, query: str, context: dict) -> RetrievalResult:
        strategy = self.select_strategy(query, context)

        if strategy == "semantic":
            # Pure embedding similarity — for broad conceptual queries
            return self.vector_db.search(query, top_k=10)

        elif strategy == "hybrid":
            # Dense + Sparse — for queries mixing concepts and keywords
            dense = self.vector_db.search(query, top_k=20)
            sparse = self.bm25.search(query, top_k=20)
            return reciprocal_rank_fusion(dense, sparse, top_k=10)

        elif strategy == "kg_guided":
            # Use Knowledge Graph to structure retrieval
            # "Find all papers that EXTEND transformer attention AND REPORT results on ImageNet"
            kg_results = self.knowledge_graph.query(context["graph_query"])
            paper_ids = [r.paper_id for r in kg_results]
            return self.vector_db.search(query, filter={"paper_id": paper_ids}, top_k=10)

        elif strategy == "multi_hop":
            # Iterative retrieval: use first results to refine query
            r1 = self.vector_db.search(query, top_k=5)
            refined_query = self.llm.refine_query(query, r1)
            r2 = self.vector_db.search(refined_query, top_k=5)
            return deduplicate(r1 + r2)

    def select_strategy(self, query: str, context: dict) -> str:
        """Route to best strategy based on query type."""
        if context.get("graph_query"):
            return "kg_guided"
        elif context.get("needs_multi_hop"):
            return "multi_hop"
        elif has_specific_keywords(query):
            return "hybrid"
        else:
            return "semantic"
```

## 7.2 Adaptive Retrieval with Corrective RAG (CRAG)

Implements the CRAG pattern — **evaluate retrieval quality before generation**:

```python
class CorrectiveRAG:
    def retrieve_and_verify(self, query: str, agent_context: dict) -> VerifiedContext:
        # Step 1: Initial retrieval
        candidates = self.retrieval_orchestrator.retrieve(query, agent_context)

        # Step 2: Relevance evaluation (lightweight classifier)
        evaluated = []
        for chunk in candidates:
            relevance = self.relevance_evaluator.score(query, chunk)
            if relevance > 0.7:
                evaluated.append(("CORRECT", chunk))
            elif relevance > 0.4:
                evaluated.append(("AMBIGUOUS", chunk))
            else:
                evaluated.append(("INCORRECT", chunk))

        # Step 3: Action based on evaluation
        correct_chunks = [c for label, c in evaluated if label == "CORRECT"]

        if len(correct_chunks) >= 3:
            # Sufficient relevant context — proceed
            return VerifiedContext(chunks=correct_chunks, confidence="high")

        elif len(correct_chunks) >= 1:
            # Partial context — augment with web search or refined retrieval
            additional = self.refined_retrieval(query, exclude=candidates)
            return VerifiedContext(
                chunks=correct_chunks + additional,
                confidence="medium"
            )

        else:
            # No relevant context — flag to orchestrator
            return VerifiedContext(
                chunks=[],
                confidence="low",
                action_needed="RETRIEVAL_FAILURE",
                suggested_fix="Expand corpus or reformulate query"
            )
```

## 7.3 Self-RAG: Adaptive Retrieval Decisions

The agent **decides when to retrieve** rather than always retrieving:

```python
class SelfRAG:
    """
    Before each generation step, the agent emits a retrieval token:
    [Retrieve] = yes → fetch context
    [Retrieve] = no  → generate from existing context
    """
    def generate_with_self_rag(self, prompt: str, existing_context: str):
        # Ask the model if it needs retrieval
        needs_retrieval = self.llm.generate(
            f"Given the task: {prompt}\n"
            f"And existing context: {existing_context[:2000]}\n"
            f"Do you have sufficient information to generate a high-quality, "
            f"factually grounded response? Answer [RETRIEVE] or [GENERATE]."
        )

        if "[RETRIEVE]" in needs_retrieval:
            # Generate a search query
            search_query = self.llm.generate(f"Generate a search query for: {prompt}")
            new_context = self.crag.retrieve_and_verify(search_query)
            combined = existing_context + new_context.chunks
        else:
            combined = existing_context

        # Generate with critique
        response = self.llm.generate(prompt, context=combined)

        # Self-critique: is the response supported?
        critique = self.llm.generate(
            f"Is every claim in this response supported by the provided context? "
            f"Mark unsupported claims with [UNSUPPORTED].\n"
            f"Response: {response}\n"
            f"Context: {combined[:5000]}"
        )

        if "[UNSUPPORTED]" in critique:
            # Iteratively fix unsupported claims
            response = self.fix_unsupported(response, critique, combined)

        return response
```

---

# 8. Quality Control & Verification Layer (Enterprise-Grade)

## 8.1 Chain-of-Verification (CoVe) Pipeline

For every major section, apply the CoVe protocol:

```
Step 1: DRAFT — Agent generates initial section content

Step 2: PLAN VERIFICATION — LLM generates fact-check questions:
  "Does the paper actually claim X?"
  "Is the reported accuracy Y% correct?"
  "Did method Z use dataset W as stated?"

Step 3: INDEPENDENT VERIFICATION — For each question:
  - Retrieve the original source
  - Answer the question from the source ONLY (not from the draft)
  - Compare answer with what the draft says

Step 4: REVISE — If discrepancies found:
  - Correct the specific claim
  - Log the correction in provenance trail
  - Re-verify the corrected version
```

### Implementation

```python
class ChainOfVerification:
    def verify_section(self, section_text: str, citations: list[str]) -> VerifiedSection:
        # Step 1: Generate verification questions
        questions = self.llm.generate(
            f"Generate 5-10 factual verification questions for this text. "
            f"Focus on: cited claims, numerical values, method descriptions, comparisons.\n"
            f"Text: {section_text}"
        )

        # Step 2: Answer each independently
        corrections = []
        for q in questions:
            # Retrieve source material
            source = self.retrieve_source_for_claim(q, citations)

            # Answer from source only
            source_answer = self.llm.generate(
                f"Based ONLY on this source material, answer: {q}\n"
                f"Source: {source}\n"
                f"If the source doesn't contain the answer, say 'NOT_FOUND'."
            )

            # Check consistency
            draft_answer = self.llm.generate(
                f"What does the draft text say about: {q}\n"
                f"Draft: {section_text}"
            )

            if not self.consistent(source_answer, draft_answer):
                corrections.append({
                    "question": q,
                    "draft_says": draft_answer,
                    "source_says": source_answer,
                    "source_id": source.paper_id,
                    "action": "CORRECT"
                })

        # Step 3: Apply corrections
        if corrections:
            corrected = self.apply_corrections(section_text, corrections)
            return VerifiedSection(
                text=corrected,
                corrections_made=len(corrections),
                verification_log=corrections,
                confidence=1.0 - (len(corrections) / len(questions))
            )

        return VerifiedSection(text=section_text, confidence=1.0, corrections_made=0)
```

## 8.2 Full Provenance Chain

Every sentence in the final paper has a traceable origin:

```json
{
  "sentence_id": "meth_s_042",
  "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
  "provenance": {
    "type": "paraphrased",
    "source_paper": "arxiv/1706.03762",
    "source_section": "Section 3.2.2",
    "source_sentence": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
    "similarity_score": 0.92,
    "transformation": "minor rewording"
  },
  "verification": {
    "cove_verified": true,
    "verification_question": "Does Vaswani et al. describe multi-head attention this way?",
    "verification_answer": "Yes — Section 3.2.2, page 5",
    "confidence": 0.98
  },
  "citation_key": "vaswani2017attention"
}
```

### Provenance Types

| Type | Description | Confidence Expectation |
|------|-------------|----------------------|
| `direct_cite` | Directly citing a factual claim | >0.95 |
| `paraphrased` | Rewording of source content | >0.85 |
| `synthesized` | Combined from multiple sources | >0.75 |
| `inferred` | Logical inference from sources | >0.65 |
| `original` | Novel content by the system | Needs novelty check |
| `simulated` | Generated experimental data | Flagged explicitly |

## 8.3 Novelty Engine (Enhanced)

### Three-Level Novelty Assessment

```python
class NoveltyEngine:
    def assess(self, generated_text: str, corpus: list[str]) -> NoveltyReport:
        # Level 1: Textual Similarity
        max_sim = max(
            cosine_similarity(embed(generated_text), embed(chunk))
            for chunk in corpus
        )
        textual_novelty = 1.0 - max_sim

        # Level 2: Structural Similarity (for methodology sections)
        # Compare architecture diagrams, algorithm steps
        structural_sim = self.compare_method_structure(
            generated_text, self.extract_methods(corpus)
        )

        # Level 3: Contribution Overlap
        # Check if claimed contributions already exist in corpus
        contribution_overlap = self.check_contribution_overlap(
            self.extract_claims(generated_text),
            self.knowledge_graph.get_all_contributions()
        )

        return NoveltyReport(
            textual_novelty=textual_novelty,        # >0.6 required
            structural_novelty=1.0 - structural_sim, # >0.4 required
            contribution_novelty=1.0 - contribution_overlap, # >0.5 required
            overall=weighted_average([textual_novelty, structural_novelty, contribution_novelty]),
            overlap_regions=self.identify_overlaps(generated_text, corpus)
        )
```

## 8.4 Reflexion Loops (Self-Improvement)

After initial generation, the system reflects and improves:

```python
class ReflexionLoop:
    MAX_ITERATIONS = 3

    def reflect_and_improve(self, section: str, section_type: str) -> str:
        for iteration in range(self.MAX_ITERATIONS):
            # Generate critique
            critique = self.critic_llm.generate(
                f"You are a senior {section_type} reviewer at a top AI venue. "
                f"Critically evaluate this section for:\n"
                f"1. Technical correctness\n"
                f"2. Clarity and flow\n"
                f"3. Missing important details\n"
                f"4. Unsupported claims\n"
                f"5. Comparison to related work\n"
                f"6. Mathematical rigor (if applicable)\n\n"
                f"Section:\n{section}\n\n"
                f"Provide specific, actionable feedback."
            )

            # Check if quality is sufficient
            quality_score = self.score_critique(critique)
            if quality_score > 0.85:
                break  # Good enough

            # Improve based on critique
            section = self.improver_llm.generate(
                f"Improve this section based on the reviewer feedback.\n"
                f"Original:\n{section}\n\n"
                f"Feedback:\n{critique}\n\n"
                f"Revised section:"
            )

            # Log the reflection
            self.log_reflection(iteration, critique, quality_score)

        return section
```

## 8.5 Consistency Validator (Cross-Section)

```python
class ConsistencyValidator:
    CHECKS = [
        # Notation consistency
        "Every symbol used in equations is defined in the notation table",
        # Claim-evidence consistency
        "Every contribution claim has corresponding experimental evidence",
        # Number consistency
        "Accuracy numbers in abstract match those in results tables",
        # Method naming
        "The proposed method name is consistent across all sections",
        # Dataset consistency
        "Dataset descriptions (size, splits) are consistent across sections",
        # Baseline consistency
        "Baseline methods listed in experiments match those in related work",
        # Reference consistency
        "Every \\cite{} has a corresponding entry in the bibliography",
        # Tense consistency
        "Related work uses past tense; methodology uses present tense",
        # Figure/table reference accuracy
        "Every \\ref{} points to an existing label",
    ]

    def validate(self, full_paper: dict) -> list[ConsistencyIssue]:
        issues = []
        for check in self.CHECKS:
            result = self.llm.generate(
                f"Check the following consistency rule across the paper:\n"
                f"Rule: {check}\n\n"
                f"Paper sections: {json.dumps(full_paper)}\n\n"
                f"Report any violations with specific locations."
            )
            if "VIOLATION" in result:
                issues.append(ConsistencyIssue(check, result))
        return issues
```

## 8.6 Confidence Scoring (Per-Section + Aggregate)

```json
{
  "section_scores": {
    "abstract":       {"confidence": 0.90, "source_coverage": 0.95, "cove_score": 0.95},
    "introduction":   {"confidence": 0.88, "source_coverage": 0.85, "cove_score": 0.90},
    "related_work":   {"confidence": 0.92, "source_coverage": 0.98, "cove_score": 0.95},
    "methodology":    {"confidence": 0.85, "source_coverage": 0.80, "cove_score": 0.88},
    "experiments":    {"confidence": 0.75, "source_coverage": 0.70, "cove_score": 0.80, "note": "simulated"},
    "results":        {"confidence": 0.78, "source_coverage": 0.72, "cove_score": 0.82, "note": "simulated"},
    "ablation":       {"confidence": 0.72, "source_coverage": 0.65, "cove_score": 0.78, "note": "simulated"},
    "discussion":     {"confidence": 0.82, "source_coverage": 0.78, "cove_score": 0.85},
    "conclusion":     {"confidence": 0.88, "source_coverage": 0.85, "cove_score": 0.90}
  },
  "aggregate": {
    "overall_confidence": 0.83,
    "hallucination_risk": "low",
    "recommendation": "Review experiments section manually — simulated data"
  }
}
```

## 8.7 Adversarial Gap Probing (NEW — v2.1)

> *The system doesn't just build an argument — it actively tries to destroy it, then rebuilds stronger.*

### Philosophy

A PhD advisor's most valuable feedback is "Why won't this work?" The Adversarial Gap Prober is an embedded devil's advocate that stress-tests every major claim, method choice, and experimental design before they reach the final paper.

### Architecture: Red Team / Blue Team Protocol

```python
class AdversarialGapProber:
    """
    Three-stage adversarial process applied to every major section.
    Operates AFTER initial generation, BEFORE reflexion loops.
    """

    def probe(self, section: str, section_type: str, 
              blackboard: dict) -> ProbedSection:

        # ═══════════════════════════════════════════════
        # STAGE 1: RED TEAM — Attack the content
        # ═══════════════════════════════════════════════
        attacks = self.red_team_llm.generate(
            f"You are a hostile but technically brilliant reviewer. "
            f"Your goal is to DESTROY this paper's argument.\n\n"
            f"For the following {section_type} section, identify:\n"
            f"1. FATAL FLAWS: Claims that are logically unsound or contradicted by known results\n"
            f"2. MISSING ATTACKS: Obvious objections the authors didn't address\n"
            f"3. STRONGER ALTERNATIVES: Methods the paper ignores that would undermine the contribution\n"
            f"4. HIDDEN ASSUMPTIONS: Unstated assumptions that, if violated, break the approach\n"
            f"5. REPRODUCIBILITY GAPS: Details so vague that no one could implement this\n\n"
            f"Section:\n{section}\n\n"
            f"Known SOTA from knowledge graph: {blackboard.get('competing_methods', [])}\n\n"
            f"Be specific. Cite concrete counter-examples from the literature."
        )

        # ═══════════════════════════════════════════════
        # STAGE 2: BLUE TEAM — Defend and strengthen
        # ═══════════════════════════════════════════════
        defense = self.blue_team_llm.generate(
            f"A reviewer has raised these attacks against your paper:\n"
            f"{attacks}\n\n"
            f"For each attack:\n"
            f"- If VALID: Acknowledge and modify the section to address it\n"
            f"- If PARTIALLY VALID: Add qualifications, conditions, or limitations\n"
            f"- If INVALID: Provide a rigorous counter-argument with citations\n\n"
            f"Original section:\n{section}\n\n"
            f"Produce the REVISED section incorporating all valid defenses."
        )

        # ═══════════════════════════════════════════════
        # STAGE 3: ARBITER — Judge the exchange
        # ═══════════════════════════════════════════════
        verdict = self.arbiter_llm.generate(
            f"You are a senior area chair evaluating this exchange:\n\n"
            f"ORIGINAL: {section[:2000]}\n\n"
            f"RED TEAM ATTACKS: {attacks}\n\n"
            f"BLUE TEAM DEFENSE: {defense[:3000]}\n\n"
            f"For each attack:\n"
            f"- Was the defense adequate? \n"
            f"- Are there REMAINING GAPS that neither side addressed?\n"
            f"- Score the revised section: 1-10\n\n"
            f"If score < 7, flag for human review."
        )

        return ProbedSection(
            original=section,
            attacks=attacks,
            defense=defense,
            arbiter_verdict=verdict,
            final_text=defense.revised_section,
            strength_score=verdict.score,
            remaining_gaps=verdict.remaining_gaps
        )
```

### What Gets Probed

| Section | Adversarial Focus |
|---------|------------------|
| **Contributions** | "Is this actually novel? Show me what's different from [paper X]" |
| **Methodology** | "This won't work because of [edge case]. What about [alternative]?" |
| **Math** | "This derivation assumes [hidden assumption]. When does it break?" |
| **Experiments** | "You cherry-picked datasets/metrics. What about [harder benchmark]?" |
| **Results** | "The improvement is marginal. Is it statistically significant?" |
| **Related Work** | "You missed [critical paper] that directly contradicts your claim" |

### Feedback Loop into Knowledge Graph

```python
# Attacks that survive blue-team defense become KNOWN WEAKNESSES
# stored in the KG for future reference
for gap in verdict.remaining_gaps:
    knowledge_graph.add_node(
        type="known_limitation",
        text=gap,
        paper_run=run_id,
        severity=gap.severity
    )
```

---

## 8.8 Deep Principled Reasoning — 3-Layer Philosophical Backpropagation (NEW — v2.1)

> *Don't copy what papers say. Understand WHY things work at the deepest level, then reconstruct understanding from principles. This is what separates a PhD thesis from a lit review.*

### The Problem with Surface-Level Reasoning

Most RAG systems do this:
```
Source paper says: "Dropout randomly zeroes neurons, acting as regularization"
Generated text:    "We use dropout as a regularization technique [cite]"
```

This is **parroting** — it copies the conclusion without understanding the principle. A PhD researcher thinks:
```
"WHY does randomly zeroing neurons regularize?
 → Because it prevents co-adaptation (functional principle)
   → Because co-adaptation reduces effective capacity (information-theoretic principle)
     → Because capacity reduction hurts generalization via implicit ensemble (foundational principle)"
```

With this depth of understanding, you can:
- Explain the concept in your OWN conceptual framework (not plagiarism)
- Connect it to other principles (genuine insight)
- Identify when it WON'T work (principled limitation)
- Write about it naturally, the way a human expert would (not AI-detectable)

### The 3-Layer Backpropagation Framework

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: SURFACE (What)                                │
│  "Paper X uses method Y to achieve result Z"            │
│  → Direct extraction from paper text                    │
│  → This is where most RAG systems stop                  │
└──────────────────────┬──────────────────────────────────┘
                       │ Backpropagate: "WHY does Y work?"
                       ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: MECHANISM (How & Why)                         │
│  "Y works because of principle P₁, under assumption A₁" │
│  → Requires cross-paper reasoning                       │
│  → Identifies the functional mechanism                  │
│  → Maps to ontology: what CLASS of solution is this?    │
└──────────────────────┬──────────────────────────────────┘
                       │ Backpropagate: "WHY does P₁ hold?"
                       ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: FOUNDATION (First Principles)                 │
│  "P₁ holds because of fundamental property F₁"         │
│  → Information theory, optimization theory, statistics  │
│  → Connects to mathematical axioms or empirical laws    │
│  → This is domain-invariant understanding               │
└─────────────────────────────────────────────────────────┘
```

### Implementation

```python
class DeepPrincipledReasoner:
    """
    For every key concept the system writes about, force reasoning
    down to foundational principles before generating text.
    """

    def reason_deeply(self, concept: str, source_chunks: list[str]) -> DeepUnderstanding:

        # ════════════════════════
        # LAYER 1: SURFACE
        # ════════════════════════
        surface = self.llm.generate(
            f"From these sources, extract WHAT is claimed about '{concept}'.\n"
            f"Sources: {source_chunks}\n\n"
            f"Output: A factual summary of what the papers say."
        )

        # ════════════════════════
        # LAYER 2: MECHANISM
        # ════════════════════════
        mechanism = self.llm.generate(
            f"Now go DEEPER. For this concept:\n"
            f"'{surface.summary}'\n\n"
            f"Answer these questions:\n"
            f"1. WHY does this work? What is the underlying mechanism?\n"
            f"2. WHAT ASSUMPTIONS must hold for this to be valid?\n"
            f"3. What CLASS of solution is this? (ontology mapping)\n"
            f"4. What are the BOUNDARY CONDITIONS where this breaks down?\n"
            f"5. What ANALOGIES exist in other fields?\n\n"
            f"Use the following related papers for cross-referencing:\n"
            f"{self.retrieve_mechanistic_explanations(concept)}"
        )

        # ════════════════════════
        # LAYER 3: FOUNDATION
        # ════════════════════════
        foundation = self.llm.generate(
            f"Now reach the DEEPEST level. For this mechanism:\n"
            f"'{mechanism.explanation}'\n\n"
            f"Trace to FIRST PRINCIPLES:\n"
            f"1. What mathematical or theoretical foundation underpins this?\n"
            f"   (information theory, optimization theory, statistical learning theory,\n"
            f"    approximation theory, probability theory)\n"
            f"2. What is the SIMPLEST possible explanation of why this works?\n"
            f"3. Can you express the core insight in ONE sentence that a \n"
            f"   first-year PhD student would understand?\n"
            f"4. What would BREAKING this principle look like?\n"
            f"5. Is this principle UNIVERSAL or domain-specific?\n\n"
            f"Reference foundational works:\n"
            f"{self.retrieve_foundational_theory(mechanism.theoretical_basis)}"
        )

        return DeepUnderstanding(
            concept=concept,
            surface=surface,         # What papers say
            mechanism=mechanism,     # Why it works
            foundation=foundation,   # First-principles basis
            principled_description=self.synthesize_from_principles(
                surface, mechanism, foundation
            )
        )

    def synthesize_from_principles(self, surface, mechanism, foundation) -> str:
        """
        Generate text that explains the concept FROM principles UP,
        not by paraphrasing sources DOWN.
        
        This produces text that:
        - Is NOT a paraphrase of any single source
        - Shows genuine understanding
        - Reads like expert writing, not LLM summarization
        - Is naturally resistant to plagiarism detection
        """
        return self.llm.generate(
            f"You deeply understand this concept at three levels:\n\n"
            f"Foundation: {foundation.core_insight}\n"
            f"Mechanism: {mechanism.explanation}\n"
            f"Surface details: {surface.summary}\n\n"
            f"Now write about this concept as an expert who UNDERSTANDS it deeply\n"
            f"would explain it to a peer. Requirements:\n"
            f"- Start from the principle, not from any paper's wording\n"
            f"- Use your OWN framing and conceptual structure\n"
            f"- Connect to the broader theoretical landscape\n"
            f"- Cite papers as evidence for your understanding, not as the source of it\n"
            f"- Write as if you're explaining YOUR understanding, backed by literature"
        )
```

### Example: 3-Layer Backprop in Action

```
Topic: "Why does self-attention work for sequence modeling?"

LAYER 1 (Surface — what most RAG systems produce):
  "Self-attention computes pairwise interactions between all positions
   in a sequence [Vaswani et al., 2017], allowing the model to capture
   long-range dependencies."

LAYER 2 (Mechanism — the WHY):
  "Self-attention works because it replaces the fixed,
   distance-dependent inductive bias of convolutions and recurrence
   with a LEARNED, content-dependent connectivity pattern.
   This means the model allocates representational capacity
   based on what the input contains, not where it appears.
   The softmax creates a convex combination — effectively a
   differentiable routing mechanism over input elements."

LAYER 3 (Foundation — first principles):
  "At its core, self-attention implements adaptive basis function
   selection. From approximation theory, any function can be
   approximated by a weighted sum of basis functions. Self-attention
   makes both the WEIGHTS and the BASES input-dependent, giving
   it strictly more expressive power than fixed-basis approaches.
   The O(n²) cost is the price of full pairwise comparison — the
   same trade-off seen in kernel methods (Mercer's theorem) and
   graph neural networks (message passing = local attention).
   The fundamental insight: ROUTING is a more powerful inductive
   bias than LOCALITY for tasks where relevant information can
   appear at arbitrary positions."

FINAL OUTPUT (written from principles, not paraphrased):
  "The effectiveness of self-attention stems from a fundamental
   shift in inductive bias: from position-dependent connectivity
   (as in convolutions and recurrence) to content-dependent routing.
   Rather than assuming that nearby elements are more relevant — an
   assumption that holds for local patterns but fails for long-range
   dependencies — self-attention treats connectivity as a learnable
   function of the input itself. This can be understood through the
   lens of approximation theory: self-attention implements adaptive
   basis selection where both the weights and the bases are
   input-conditioned, providing strictly greater expressiveness than
   fixed-basis alternatives [Vaswani et al., 2017; Yun et al., 2020].
   The quadratic complexity is the inherent cost of exhaustive pairwise
   comparison, a trade-off well-studied in kernel methods [Scholkopf
   & Smola, 2002] and recently addressed through sparse approximations
   [Child et al., 2019; Kitaev et al., 2020]."
```

Notice: the final output **cites the same papers** but the structure of understanding comes from *principles*, not from paraphrasing. No plagiarism detector or AI detector would flag this because it reflects genuine comprehension, not text transformation.

### Where 3-Layer Reasoning Applies

| Section | Layer 1 (Skip) | Layer 2 (Minimum) | Layer 3 (Required) |
|---------|----------------|--------------------|--------------------|
| Related Work | ❌ | ✅ Explain why methods work/fail | ✅ Connect to theoretical basis |
| Methodology | ❌ | ✅ Justify each design choice | ✅ Ground in theory |
| Introduction | ❌ | ✅ Explain the problem's significance | ✅ Why existing solutions fundamentally fall short |
| Discussion | ❌ | ✅ Interpret results mechanistically | ✅ What this reveals about underlying principles |
| Math | ❌ | ❌ | ✅ Every equation must connect to a principle |

---

## 8.9 Humanized Writing Engine (NEW — v2.1)

> *The output should read like a paper written by an experienced researcher, not like an LLM completing a prompt.*

### Why AI-Generated Academic Text is Detectable

LLM writing has fingerprints:
1. **Uniform sentence structure** — predictable rhythm, similar length
2. **Hedge-heavy** — "It is worth noting that", "It should be mentioned"
3. **Excessive enumeration** — "First... Second... Third... Finally..."
4. **Flat information density** — every sentence carries equal weight
5. **Template phrasing** — "In this paper, we propose", "To the best of our knowledge"
6. **No intellectual personality** — reads like a committee wrote it

### Humanization Strategy (Multi-Layer)

```python
class HumanizedWritingEngine:
    """
    Transform AI-generated academic text into human-quality writing.
    Applied AFTER all content is verified and provenanced.
    Does NOT change factual content — only style and structure.
    """

    def humanize(self, section: str, section_type: str, 
                 author_style: str = "analytical") -> str:

        # ════════════════════════════════════════
        # LAYER 1: Deep Principled Foundation
        # Text is ALREADY written from principles
        # (via §8.8), not paraphrased from sources
        # ════════════════════════════════════════
        # Prerequisite: section was generated through
        # 3-Layer Backpropagation, so it already has
        # genuine understanding baked in.

        # ════════════════════════════════════════
        # LAYER 2: Structural Variation
        # ════════════════════════════════════════
        varied = self.llm.generate(
            f"Rewrite this academic text to have natural structural variation:\n\n"
            f"{section}\n\n"
            f"Rules:\n"
            f"- Vary sentence length: mix short punchy sentences with longer analytical ones\n"
            f"- Vary paragraph length: some 2-sentence, some 5-sentence\n"
            f"- Use rhetorical questions where they strengthen the argument\n"
            f"- Occasionally lead with the conclusion, then explain why\n"
            f"- Remove ALL instances of: 'It is worth noting', 'It should be mentioned',\n"
            f"  'To the best of our knowledge', 'In this paper we propose',\n"
            f"  'importantly', 'moreover', 'furthermore', 'additionally'\n"
            f"- DO NOT change any factual claims, citations, or equations\n"
            f"- Preserve all \\cite{{}} and \\ref{{}} commands exactly"
        )

        # ════════════════════════════════════════
        # LAYER 3: Information Density Variation
        # ════════════════════════════════════════
        # Human writing has peaks and valleys — key insights
        # are delivered with punch, context is woven smoothly
        dense = self.llm.generate(
            f"Adjust the information density of this text to read like an\n"
            f"experienced researcher wrote it:\n\n"
            f"{varied}\n\n"
            f"Rules:\n"
            f"- KEY INSIGHTS should be stated directly and memorably\n"
            f"- Background context should flow naturally, not be listed\n"
            f"- Build TENSION before revealing the main result\n"
            f"- Use concrete examples and intuition before formal statements\n"
            f"- The first sentence of each paragraph should EARN the reader's attention\n"
            f"- DO NOT change any factual claims, citations, or equations"
        )

        # ════════════════════════════════════════
        # LAYER 4: Intellectual Voice
        # ════════════════════════════════════════
        # Add the subtle markers of expert writing:
        # confident claims, precise hedging, intellectual opinions
        voiced = self.llm.generate(
            f"Add intellectual voice to this academic text. "
            f"Style: {author_style}\n\n"
            f"{dense}\n\n"
            f"An expert researcher:\n"
            f"- Has OPINIONS (backed by evidence) — 'We argue that...', not 'It can be seen that...'\n"
            f"- Uses PRECISE hedging only when genuinely uncertain — 'We conjecture' vs 'We demonstrate'\n"
            f"- Makes CONNECTIONS the reader didn't expect\n"
            f"- Occasionally uses VIVID language for key insights\n"
            f"- Shows TASTE — not everything is equally important, emphasize what matters\n"
            f"- DO NOT change any factual claims, citations, or equations"
        )

        # ════════════════════════════════════════
        # LAYER 5: AI Fingerprint Removal
        # ════════════════════════════════════════
        final = self.remove_ai_fingerprints(voiced)

        return final

    def remove_ai_fingerprints(self, text: str) -> str:
        """Detect and remove common LLM writing patterns."""
        # Pattern detection
        AI_PATTERNS = [
            (r"It is worth noting that ", ""),
            (r"It should be mentioned that ", ""),
            (r"To the best of our knowledge,? ", ""),
            (r"In recent years, ", ""),
            (r"plays a (crucial|vital|pivotal|important) role", "contributes to"),
            (r"has gained (significant|considerable) attention", "has been widely studied"),
            (r"In this (paper|work), we propose", "We propose"),
            (r"The rest of (this|the) paper is organized as follows", ""),
        ]
        for pattern, replacement in AI_PATTERNS:
            text = re.sub(pattern, replacement, text)

        # Burstiness check: measure sentence length variance
        sentences = split_sentences(text)
        lengths = [len(s.split()) for s in sentences]
        variance = np.var(lengths)

        if variance < 20:  # Too uniform — flag for rewrite
            text = self.increase_burstiness(text)

        return text

    def increase_burstiness(self, text: str) -> str:
        """Human writing has high 'burstiness' — alternating between
        short and long sentences. LLM text tends to be uniform."""
        return self.llm.generate(
            f"This text has too-uniform sentence lengths. Rewrite to add natural\n"
            f"variation — some sentences should be very short (5-8 words) for impact,\n"
            f"others should be complex multi-clause sentences (25-40 words) for nuance.\n"
            f"DO NOT change factual content.\n\n{text}"
        )
```

### Humanization Quality Metrics

```python
class WritingQualityMetrics:
    def score(self, text: str) -> dict:
        return {
            # Burstiness: variance in sentence length (higher = more human)
            "burstiness": self.sentence_length_variance(text),  # target: >25

            # Perplexity variance: information density variation
            "perplexity_variance": self.local_perplexity_variance(text),  # target: >15

            # AI detector score (lower = more human-like)
            "ai_detection_score": self.run_detector(text),  # target: <0.3

            # Vocabulary richness: type-token ratio
            "vocabulary_richness": self.type_token_ratio(text),  # target: >0.65

            # Hedge ratio: proportion of hedged claims (should be low)
            "hedge_ratio": self.count_hedges(text) / self.count_claims(text),  # target: <0.2

            # Template phrase count (should be zero)
            "template_phrases": self.count_ai_templates(text),  # target: 0
        }
```

### The Full Anti-Plagiarism + Anti-AI-Detection Pipeline

```
Source Papers
    │
    ▼
3-Layer Backpropagation (§8.8)
    │  → Understand at principle level, not text level
    │  → Text is reconstructed FROM principles
    │  → NOT a paraphrase of any single source
    │
    ▼
Content Generation (Agents)
    │  → Written from principled understanding
    │  → Every claim has provenance but phrasing is original
    │
    ▼
Adversarial Gap Probing (§8.7)
    │  → Forces deeper, non-obvious arguments
    │  → Pressure-tests against literature
    │
    ▼
Humanized Writing Engine (§8.9)
    │  → Structural variation (sentence length, paragraph length)
    │  → Information density variation (peaks and valleys)
    │  → Intellectual voice (opinions, precise hedging, taste)
    │  → AI fingerprint removal
    │
    ▼
Quality Metrics Check
    │  → Burstiness > 25
    │  → AI detection score < 0.3
    │  → Template phrases = 0
    │  → Plagiarism similarity < 0.15 against any single source
    │
    ▼
Final Output: Human-quality, principle-grounded, provenance-tracked text
```

---

# 9. Automated Evaluation Suite (NEW — v2)

## 9.1 Evaluation Dimensions

| Dimension | Metric | Target | Method |
|-----------|--------|--------|--------|
| **Factual Accuracy** | CoVe pass rate | >90% | Chain-of-Verification |
| **Citation Accuracy** | Claims per citation correctness | >95% | Source verification |
| **Novelty** | Overall novelty score | >0.5 | Novelty Engine |
| **Coherence** | Cross-section consistency | 0 violations | Consistency Validator |
| **Completeness** | Required sections present | 100% | Schema check |
| **Technical Depth** | Equations, theorems, proofs | Domain-appropriate count | Count + verify |
| **Writing Quality** | Readability + academic tone | Flesch-Kincaid ~ 30-40 | Automated + LLM |
| **Formatting** | LaTeX compilation success | 0 errors | pdflatex |
| **Structure** | Venue template conformance | 100% | Template validator |
| **Provenance** | % sentences with provenance | >85% | Provenance tracker |
| **Adversarial Robustness** | % sections surviving Red Team | >80% | Adversarial Gap Prober |
| **Reasoning Depth** | Concepts with 3-layer backprop | >90% of key concepts | Deep Principled Reasoner |
| **Human-likeness** | AI detection score | <0.3 | Humanized Writing Engine |
| **Burstiness** | Sentence length variance | >25 | Writing Quality Metrics |
| **Plagiarism** | Max similarity to any single source | <0.15 | Novelty Engine + 3-Layer |

## 9.2 Automated Review Simulation

Run a **simulated peer review** with 3 independent LLM reviewers:

```python
class SimulatedReview:
    REVIEWER_PERSONAS = [
        "area_chair: senior researcher, focuses on novelty and significance",
        "methods_expert: focuses on technical correctness and rigor",
        "applications_expert: focuses on experimental design and reproducibility"
    ]

    def review(self, paper: dict) -> ReviewReport:
        reviews = []
        for persona in self.REVIEWER_PERSONAS:
            review = self.llm.generate(
                f"You are a {persona} at {paper['venue']}. "
                f"Review this paper following the standard review format:\n"
                f"1. Summary (2-3 sentences)\n"
                f"2. Strengths (bullet list)\n"
                f"3. Weaknesses (bullet list)\n"
                f"4. Questions for authors\n"
                f"5. Missing references\n"
                f"6. Score: 1-10\n"
                f"7. Confidence: 1-5\n\n"
                f"Paper: {paper['full_text']}"
            )
            reviews.append(review)

        # Meta-review
        meta = self.llm.generate(
            f"As area chair, synthesize these reviews and provide "
            f"actionable revision guidance:\n{reviews}"
        )

        return ReviewReport(reviews=reviews, meta_review=meta)
```

---

# 10. Enterprise Infrastructure

## 10.1 File System Design (Enhanced)

```
/project
├── config/
│   ├── agent_contracts/        # JSON schemas per agent
│   ├── ontology/               # AI/ML/DL taxonomy
│   ├── model_routing.yaml      # MoA configuration
│   └── hitl_gates.yaml         # Review checkpoint config
├── papers_raw/                 # Downloaded PDFs
├── parsed/
│   ├── chunks/                 # Parsed and chunked text
│   ├── equations/              # Extracted LaTeX equations
│   ├── tables/                 # Extracted tables (CSV/JSON)
│   └── figures/                # Extracted figure images + captions
├── knowledge_store/
│   ├── vector_db/              # Qdrant data
│   ├── knowledge_graph/        # Neo4j / NetworkX graph
│   └── metadata.db             # SQLite metadata
├── memory/
│   ├── working/                # Current run scratchpads
│   ├── episodic/               # Shared blackboard
│   └── long_term/              # Cross-run experience
├── intermediate/
│   ├── agent_outputs/          # Raw outputs per agent
│   ├── verification_logs/      # CoVe results
│   ├── provenance/             # Sentence-level tracking
│   └── reflexion_logs/         # Improvement iterations
├── cache/
│   ├── llm_cache/              # Semantic dedup of LLM calls
│   ├── embedding_cache/        # Cached embeddings
│   └── retrieval_cache/        # Cached search results
├── latex/
│   ├── templates/              # NeurIPS, ICLR, ICML templates
│   ├── figures/                # Generated figures (matplotlib/tikz)
│   ├── main.tex                # Final paper
│   └── references.bib          # Bibliography
├── output/
│   ├── paper.pdf               # Final PDF
│   ├── confidence_report.json  # Section-level scores
│   ├── provenance_report.json  # Full provenance chain
│   ├── review_simulation.json  # Simulated peer review
│   └── generation_log.json     # Full execution trace
└── tests/
    ├── contract_tests/         # Agent contract validation
    ├── integration_tests/      # End-to-end pipeline tests
    └── quality_benchmarks/     # Quality regression tests
```

## 10.2 Semantic Cache Layer

```python
class SemanticCache:
    """
    Cache LLM responses keyed by semantic similarity of inputs.
    If a near-identical query was made before, return cached response.
    """
    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold
        self.cache = {}  # embedding → response

    def get_or_generate(self, prompt: str, llm_fn: callable) -> str:
        prompt_embedding = embed(prompt)

        # Check for semantically similar cached query
        for cached_embedding, cached_response in self.cache.items():
            if cosine_similarity(prompt_embedding, cached_embedding) > self.threshold:
                return cached_response  # Cache hit

        # Cache miss — generate and store
        response = llm_fn(prompt)
        self.cache[prompt_embedding] = response
        return response
```

## 10.3 Fault Tolerance

```python
class CircuitBreaker:
    """
    Prevents cascading failures when external services (LLM APIs,
    paper sources, vector DB) are unavailable.
    """
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failure_count = 0
        self.threshold = failure_threshold
        self.state = "CLOSED"  # CLOSED → OPEN → HALF_OPEN → CLOSED

    def call(self, fn, *args, **kwargs):
        if self.state == "OPEN":
            raise CircuitOpenError("Service unavailable, try later")

        try:
            result = fn(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.threshold:
                self.state = "OPEN"
                self.schedule_half_open(self.reset_timeout)
            raise

class RetryPolicy:
    """Exponential backoff with jitter for transient failures."""
    def retry(self, fn, max_retries=3, base_delay=1.0):
        for attempt in range(max_retries):
            try:
                return fn()
            except TransientError:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
        raise MaxRetriesExceeded()
```

### Graceful Degradation Strategy

| Component Failure | Degradation Behavior |
|-------------------|---------------------|
| Vector DB down | Fall back to BM25 over raw chunks |
| Knowledge Graph down | Skip multi-hop reasoning, use flat retrieval |
| Primary LLM unavailable | Route to secondary model (lower quality, flagged) |
| Paper source API down | Use cached papers only, flag corpus as incomplete |
| LaTeX compilation fails | Return Markdown version + raw LaTeX for manual fix |

## 10.4 Experience Replay Store (Cross-Run Learning)

```python
class ExperienceReplay:
    """
    After each paper generation, capture what worked and what didn't.
    Use this to improve future runs.
    """
    def capture(self, run_id: str, generation_log: dict, quality_scores: dict):
        experience = {
            "run_id": run_id,
            "domain": generation_log["domain"],
            "effective_queries": self.extract_high_yield_queries(generation_log),
            "failed_agents": self.extract_failures(generation_log),
            "quality_scores": quality_scores,
            "reflexion_insights": self.extract_reflexion_insights(generation_log),
            "optimal_k_values": self.extract_retrieval_stats(generation_log),
            "token_usage": self.extract_token_usage(generation_log),
            "timestamp": datetime.now()
        }
        self.store.insert(experience)

    def apply_to_new_run(self, domain: str) -> dict:
        """Retrieve relevant experiences for a new run in a similar domain."""
        similar_runs = self.store.query(domain=domain, min_quality=0.8)
        return {
            "recommended_queries": merge_queries(similar_runs),
            "recommended_k_values": avg_k_values(similar_runs),
            "known_failure_patterns": collect_failures(similar_runs),
            "expected_quality_range": quality_range(similar_runs)
        }
```

---

# 11. Output Composition (Enhanced)

## 11.1 Paper Structure (Venue-Adaptive)

```python
STRUCTURE_TEMPLATES = {
    "NeurIPS": {
        "max_pages": 10,
        "sections": [
            "Title", "Abstract (≤250 words)", "Introduction",
            "Related Work", "Preliminary / Background",
            "Method", "Theoretical Analysis (optional)",
            "Experiments", "Results", "Ablation Study",
            "Discussion", "Conclusion",
            "Broader Impact Statement", "References",
            "Appendix (supplementary, no page limit)"
        ],
        "formatting": "neurips_2026.sty"
    },
    "ICLR": {
        "max_pages": 10,
        "sections": [
            "Title", "Abstract", "Introduction",
            "Related Work", "Method", "Experiments",
            "Analysis", "Conclusion", "References",
            "Appendix"
        ],
        "formatting": "iclr2026_conference.sty"
    },
    "ICML": { ... },
    "ACL": { ... },
    "CVPR": { ... }
}
```

## 11.2 LaTeX Generation Agent (Enhanced)

### Quality Requirements

- All equations numbered and referenced
- All figures have descriptive captions (≥2 sentences)
- All tables have headers with units
- Consistent font, spacing per template
- No overfull hboxes
- All `\cite{}` resolve to `\bib` entries
- All `\ref{}` resolve to `\label{}`

### Figure Generation

```python
FIGURE_GENERATORS = {
    "architecture_diagram": "tikz",          # TikZ for clean diagrams
    "result_plot":          "matplotlib",     # Standard result plots
    "convergence_curve":    "matplotlib",
    "attention_heatmap":    "matplotlib/seaborn",
    "comparison_table":     "booktabs",       # LaTeX booktabs
    "algorithm":            "algorithm2e",    # LaTeX algorithm package
}
```

## 11.3 Reference Manager (Enhanced)

```python
class ReferenceManager:
    def generate_bibtex(self, cited_papers: list[str]) -> str:
        entries = []
        for paper_id in cited_papers:
            meta = self.metadata_store.get(paper_id)
            entry = self.format_bibtex(meta)

            # Verify BibTeX compiles
            if not self.validate_bibtex_entry(entry):
                entry = self.fix_bibtex(entry)

            entries.append(entry)

        # Deduplication
        entries = self.deduplicate(entries)

        # Sort by citation key (alphabetical)
        entries.sort(key=lambda e: e.key)

        return "\n\n".join(entries)

    def verify_all_cited(self, latex_source: str, bib_entries: list) -> list[str]:
        """Ensure every \\cite{key} has a matching bib entry."""
        cited_keys = re.findall(r'\\cite\{([^}]+)\}', latex_source)
        bib_keys = {e.key for e in bib_entries}
        missing = [k for k in cited_keys if k not in bib_keys]
        return missing  # Should be empty
```

---

# 12. Execution Workflow (v2 — Enhanced)

```
INPUT (Title + Abstract + Direction + Constraints)
 │
 ├──→ [HITL Gate 1: Confirm interpretation] (optional)
 │
 ▼
 Requirement Gathering Agent
 │  ├── Intent decomposition
 │  ├── Ontology-guided query expansion
 │  └── Domain classification
 │
 ├──→ [HITL Gate 2: Review queries] (optional)
 │
 ▼
 Paper Discovery (multi-source, multi-phase)
 │  ├── Seed discovery
 │  ├── Citation graph expansion
 │  ├── Recency boost
 │  └── Contrastive discovery
 │
 ▼
 Ingestion Pipeline
 │  ├── GROBID/Nougat parsing
 │  ├── Section-aware chunking
 │  ├── Deep metadata enrichment
 │  ├── SPECTER2 embedding + SPLADE sparse
 │  └── Knowledge Graph construction
 │
 ▼
 Knowledge Store Ready
 │  (Vector DB + Knowledge Graph + Metadata DB)
 │
 ▼
 Context Engineering Layer Initialized
 │  (Token budgets + Memory tiers + Context router)
 │
 ▼
 Orchestrator: Dynamic DAG Construction
 │
 ├──→ PARALLEL PHASE 1:
 │     ├── Contributions Agent (+ self-verify)
 │     ├── Problem Formulation Agent (+ notation registry)
 │     └── Literature Review Agent (multi-hop synthesis)
 │
 ├──→ [HITL Gate 3: Review contributions + lit review] (optional)
 │
 ├──→ SEQUENTIAL PHASE 2:
 │     ├── Math Formulation Agent (SymPy-verified)
 │     ├── Architecture Design Agent (justification chains)
 │     └── Implementation Details Agent
 │
 ├──→ PARALLEL PHASE 3:
 │     ├── Experiment Design Agent
 │     └── Ablation Design Agent
 │
 ├──→ [HITL Gate 4: Approve experiment plan] (optional)
 │
 ├──→ SEQUENTIAL PHASE 4:
 │     ├── Experiment Execution Agent (simulated, flagged)
 │     ├── Results Analysis Agent (claim↔result alignment)
 │     └── Discussion Agent
 │
 ├──→ PARALLEL PHASE 5:
 │     ├── Limitations & Ethics Agent
 │     ├── Abstract & Conclusion Agent
 │     └── Appendix Agent
 │
 ▼
 QUALITY CONTROL SWEEP
 │  ├── Chain-of-Verification (per section)
 │  ├── Adversarial Gap Probing (Red/Blue/Arbiter)
 │  ├── 3-Layer Principled Reasoning verification
 │  ├── Provenance validation
 │  ├── Novelty assessment
 │  ├── Consistency validation (cross-section)
 │  ├── Confidence scoring
 │  ├── Reflexion loops (up to 3 iterations per section)
 │  └── Humanized Writing Engine pass
 │
 ├──→ [HITL Gate 5: Final review before compilation] (optional)
 │
 ▼
 OUTPUT COMPOSITION
 │  ├── LaTeX generation (venue-specific template)
 │  ├── Figure generation (tikz + matplotlib)
 │  ├── BibTeX generation + verification
 │  ├── PDF compilation (pdflatex × 3 passes)
 │  └── Output packaging
 │
 ▼
 DELIVERABLES
   ├── paper.pdf
   ├── main.tex + figures/ + references.bib
   ├── confidence_report.json
   ├── provenance_report.json
   ├── simulated_review.json
   └── generation_log.json
```

---

# 13. Technology Stack Recommendation

| Component | Technology | Reasoning |
|-----------|-----------|-----------|
| **Orchestration** | Python + LangGraph or custom DAG | Dynamic DAG with state management |
| **Primary LLM** | Claude Opus 4 / GPT-4.5 | Strongest reasoning for core agents |
| **Secondary LLM** | Claude Sonnet 4 | Cost-efficient for synthesis tasks |
| **Fast LLM** | Claude Haiku 3.5 | Metadata extraction, formatting |
| **Embeddings** | SPECTER2 + E5-Mistral-7b | Scientific + general embeddings |
| **Sparse Search** | SPLADE | Learned sparse representations |
| **Vector DB** | Qdrant | Hybrid search, filtering, multi-vector |
| **Knowledge Graph** | Neo4j (production) / NetworkX (prototype) | Structured relation querying |
| **Metadata DB** | PostgreSQL (production) / SQLite (prototype) | Structured queries + ACID |
| **PDF Parsing** | GROBID + Nougat | Complementary strengths |
| **Math Verification** | SymPy + NumPy | Symbolic + numerical verification |
| **LaTeX** | pdflatex + bibtex | Standard compilation |
| **Visualization** | Matplotlib + TikZ | Plots + diagrams |
| **Caching** | Redis (production) / diskcache (prototype) | Fast semantic cache |
| **Task Queue** | Celery / asyncio | Parallel agent execution |
| **Monitoring** | Structured logging + LangSmith/Weave | LLM call tracing |
| **Testing** | pytest + custom quality benchmarks | Agent contract validation |

---

# 14. Key Differentiators (v2 vs v1)

| Capability | v1 | v2 |
|-----------|----|----|
| Context Management | None | **Hierarchical 3-tier memory + token budgets + compression** |
| Retrieval | Basic RAG | **Self-RAG + CRAG + Multi-Hop + KG-guided** |
| Knowledge Representation | Vector DB only | **Vector DB + Knowledge Graph + Ontology** |
| Hallucination Prevention | Citation check | **CoVe + Provenance tracking + Self-RAG critique** |
| Agent Architecture | Static task list | **Dynamic DAG + Shared Blackboard + MoA routing** |
| Math Reasoning | LLM only | **Tool-integrated (SymPy verification)** |
| Quality Assurance | Basic similarity | **Reflexion loops + simulated peer review + consistency validator** |
| Adversarial Self-Challenge | None | **Red Team / Blue Team / Arbiter protocol on every section** |
| Reasoning Depth | Surface paraphrasing | **3-Layer Philosophical Backpropagation (Surface → Mechanism → Foundation)** |
| Writing Humanization | None | **Humanized Writing Engine + AI fingerprint removal + burstiness control** |
| Fault Tolerance | None | **Circuit breakers + retries + graceful degradation** |
| Human Oversight | None | **Configurable HITL gates** |
| Cross-Run Learning | None | **Experience replay store** |
| Output Artifacts | PDF only | **PDF + confidence report + provenance trail + review simulation** |
| Domain Awareness | Generic | **AI/ML/DL ontology with 200+ concepts** |

---

# 15. Implementation Phases (Recommended)

## Phase A — Foundation (Weeks 1-3)
- [ ] Project scaffold + config system
- [ ] Paper Discovery Agent (Semantic Scholar + arXiv)
- [ ] Ingestion Pipeline (GROBID + chunking)
- [ ] Vector DB setup (Qdrant + SPECTER2)
- [ ] Basic Orchestrator (static DAG)
- [ ] 3 core agents: Contributions, Problem Formulation, Literature Review
- [ ] Basic LaTeX generation

## Phase B — Intelligence (Weeks 4-6)
- [ ] Knowledge Graph construction
- [ ] AI/ML/DL Ontology
- [ ] Context Engineering Layer (token budgets, memory tiers)
- [ ] Remaining sub-agents (Math, Architecture, Experiments, etc.)
- [ ] CRAG implementation
- [ ] MoA model routing

## Phase C — Quality (Weeks 7-9)
- [ ] Chain-of-Verification pipeline
- [ ] Provenance tracking
- [ ] Novelty Engine
- [ ] Consistency Validator
- [ ] Reflexion loops
- [ ] Confidence scoring

## Phase D — Enterprise (Weeks 10-12)
- [ ] Dynamic DAG with re-planning
- [ ] HITL gates
- [ ] Fault tolerance (circuit breakers, retries)
- [ ] Semantic cache
- [ ] Experience replay
- [ ] Simulated peer review
- [ ] Monitoring + logging
- [ ] End-to-end integration tests

---

# 16. Final System Positioning

> **An enterprise-grade, context-engineered autonomous research synthesis framework with PhD-level domain specialization in AI/ML/DL, featuring multi-modal knowledge grounding (vector + graph + ontology), multi-layered hallucination prevention (Self-RAG + CRAG + Chain-of-Verification + Provenance), adaptive multi-agent orchestration (Dynamic DAG + Mixture-of-Agents + Shared Blackboard), and publication-grade output with full confidence and provenance reporting.**

---

*End of v2 specification. This document supersedes plan_doc.md (v1).*
