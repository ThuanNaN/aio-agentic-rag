# AIO Agentic RAG — Vietnamese Legal Benchmark

Benchmarking **5 RAG strategies** on Vietnamese legal documents using Deep Agents orchestration.

## Dataset
[th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) — ~153k docs, ~897k cross-document relationships.

## Strategies Compared

| Strategy | Retrieval | Notes |
|----------|-----------|-------|
| **Naive RAG** | Dense (Chroma) | Baseline |
| **Hybrid RAG** | BM25 + Dense, RRF fusion | Better keyword matching |
| **Reranker RAG** | Hybrid + CrossEncoder | Higher precision |
| **GraphRAG** | Hybrid + NetworkX multi-hop | Amendment chains |
| **Agentic RAG** | Query classification → route → retry | Best quality |

## Stack

| Component | Options |
|-----------|---------|
| **Orchestration** | [Deep Agents](https://github.com/langchain-ai/deepagents) (`create_deep_agent`) |
| **LLM** | OpenAI · OpenRouter · Ollama · Gemini (configurable) |
| **Embeddings** | `google/embeddinggemma-300m` — local or vLLM server |
| **Vector Store** | Chroma (persistent local or Docker) |
| **BM25** | `rank-bm25` |
| **Reranker** | `BAAI/bge-m3` |
| **Graph** | NetworkX (amendment/reference chains) |
| **Evaluation** | RAGAS + Recall@k + nDCG@k |
| **API** | FastAPI |

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

For optional providers:
```bash
pip install -e ".[ollama]"           # Ollama
pip install -e ".[gemini]"           # Google Gemini
pip install -e ".[all-providers]"    # everything
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in the relevant block for your chosen LLM and embedding provider (see [Configuration](#configuration) below).

### 3. (Optional) Start Chroma via Docker

```bash
docker compose up -d
# verify: curl http://localhost:8001/api/v2/heartbeat
```

### 4. Ingest documents

```bash
# Quick smoke-test: 1000 docs
python scripts/ingest.py --sample 1000

# Full dataset (~153k docs, hours on CPU):
# python scripts/ingest.py
```

### 5. Build the evaluation gold set

```bash
python scripts/build_eval_set.py
```

Then open `data/eval/gold_set.json` and fill in `expected_doc_ids` for each query — use `doc_id` values from `data/processed/raw_docs.json`. Without them Recall@k and nDCG@k will be 0.

### 6. Run the benchmark

```bash
# All 5 strategies, 8 queries
python scripts/run_benchmark.py --strategy all --sample 8

# Single strategy
python scripts/run_benchmark.py --strategy agentic --sample 8

# Full gold set
python scripts/run_benchmark.py --strategy all
```

Expected output:
```
=================================================================
Strategy             recall@5          ndcg@10    avg_latency_ms
=================================================================
naive                  0.XXXX           0.XXXX          XX.XX
hybrid                 0.XXXX           0.XXXX          XX.XX
reranker               0.XXXX           0.XXXX          XX.XX
graph                  0.XXXX           0.XXXX          XX.XX
agentic                0.XXXX           0.XXXX          XX.XX
=================================================================
Results saved to results/benchmark_YYYYMMDD_HHMMSS.csv
```

### 7. Start the API

```bash
uvicorn src.api.app:app --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Index readiness check |
| `/query` | POST | Answer a single legal question |
| `/benchmark` | POST | Run strategies against the gold set |

```bash
# Health check
curl http://localhost:8000/health

# Single query
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Luật Đất đai số 31/2024/QH15 có hiệu lực từ ngày nào?", "strategy": "hybrid", "k": 5}'

# Benchmark via API
curl -s -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{"strategies": ["naive", "hybrid", "reranker", "graph", "agentic"], "sample_n": 8}'
```

---

## Configuration

All configuration is done through environment variables in `.env`.

### LLM Provider (`LLM_PROVIDER`)

| Provider | `LLM_PROVIDER` | Required keys | Example models |
|----------|---------------|---------------|----------------|
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini`, `gpt-4o`, `gpt-4.1` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini`, `meta-llama/llama-3.1-70b-instruct` |
| Ollama (local) | `ollama` | — | `llama3.1`, `qwen2.5`, `mistral` |
| Google Gemini | `gemini` | `GOOGLE_API_KEY` | `gemini-1.5-flash`, `gemini-2.0-flash` |

```bash
# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# OpenRouter
LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=sk-or-...

# Ollama (run: ollama pull llama3.1 first)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1

# Gemini
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
GOOGLE_API_KEY=AIza...
```

### Embedding Provider (`EMBEDDING_PROVIDER`)

| Provider | `EMBEDDING_PROVIDER` | Notes |
|----------|---------------------|-------|
| Local SentenceTransformer | `local` | Default, no server needed |
| vLLM server | `vllm` | Faster for large datasets, GPU-accelerated |

```bash
# Local (default)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=google/embeddinggemma-300m
EMBEDDING_DEVICE=cpu     # or cuda

# vLLM server
# Start: vllm serve google/embeddinggemma-300m --task embed --port 8080
EMBEDDING_PROVIDER=vllm
EMBEDDING_MODEL=google/embeddinggemma-300m
EMBEDDING_BASE_URL=http://localhost:8080/v1
EMBEDDING_API_KEY=empty
```

---

## Project Structure

```
src/
├── llm.py          # LLM factory (OpenAI / OpenRouter / Ollama / Gemini)
├── ingestion/      # loader, cleaner, chunker
├── indexing/       # embeddings (local + vLLM), chroma_store, bm25_index
├── retrieval/      # dense, bm25, hybrid, reranker, graph
├── tools/          # @tool wrappers for subagents
├── agents/         # Deep Agent orchestrator + subagents
├── evaluation/     # gold_set, metrics, benchmark runner
└── api/            # FastAPI service

skills/             # SKILL.md for each RAG strategy
notebooks/          # Exploration + results visualization
scripts/            # CLI: ingest, benchmark, build eval set
configs/            # config.yaml
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration` | Dataset structure, metadata analysis |
| `02_indexing_pipeline` | Build indexes on a sample |
| `03_pipeline_comparison` | Side-by-side strategy outputs |
| `04_benchmark_results` | Metrics table + charts |

## Additional Documentation

- [Architecture Guide](docs/architecture.md) - codebase walkthrough, data flow, retrieval strategy implementation details, and extension points
