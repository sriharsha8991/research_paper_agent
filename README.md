# Article Generator — Enterprise-Grade Research Paper System

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env  # Add your API keys
article-gen --title "Your Research Title" --venue neurips
```

## Architecture

See `plan_doc_v2.md` for full specification.

```
src/article_generator/
├── models.py          # Core data models (Pydantic)
├── config.py          # Configuration management
├── main.py            # CLI entrypoint
├── agents/            # All sub-agents
├── context/           # Context engineering (memory, budgets)
├── discovery/         # Paper discovery
├── ingestion/         # PDF parsing & chunking
├── knowledge/         # Vector DB + Knowledge Graph
├── orchestrator/      # DAG execution engine
├── quality/           # CoVe, provenance, novelty, adversarial
├── retrieval/         # RAG orchestrator (CRAG, Self-RAG)
└── output/            # LaTeX generation, PDF compilation
```
