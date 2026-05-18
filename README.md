# Vietnamese Legal Agentic RAG

An Agentic RAG system for answering questions about Vietnamese legal documents, built with Deep Agents and LangGraph.

```
User question
     │
     ▼
┌────────────────────────────────────────────┐
│             Orchestrator Agent             │
│          (Deep Agents + LangGraph)         │
└──────────┬──────────────────┬──────────────┘
           │                  │
           ▼                  ▼
  ┌─────────────────┐  ┌───────────────────────────────┐
  │ Ingestion Agent │  │       Legal-RAG Agent         │
  │                 │  │                               │
  │  HF download    │  │  1. Classify query intent     │
  │  HTML clean     │  │  2. Route to best retrieval   │
  │  Chunk text     │  │  3. Grade → retry if poor     │
  │  Chroma index   │  │  4. Generate cited answer     │
  │  BM25 index     │  │                               │
  │  Graph index    │  │  dense · bm25 · hybrid        │
  └─────────────────┘  │  reranker · graph             │
                       └───────────────────────────────┘
```

Instead of always using the same retrieval pipeline, the agent classifies each query and picks the right strategy:

| Query type | Signal keywords | Retrieval |
|------------|----------------|-----------|
| `multi_hop` | "sửa đổi", "thay thế", "tham chiếu" | Graph traversal (amendment chains) |
| `temporal` | "còn hiệu lực", "sau năm", "từ ngày" | Hybrid + date metadata filter |
| `factual` / `reasoning` | everything else | Hybrid → CrossEncoder rerank |

If fewer than 2 relevant documents are found, the agent rewrites the query and retries with dense search.

## Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | [Deep Agents](https://github.com/langchain-ai/deepagents) (`create_deep_agent`) |
| LLM | OpenAI · OpenRouter · vLLM · Gemini |
| Embeddings | `google/embeddinggemma-300m` via vLLM server |
| Vector store | ChromaDB (HTTP, Docker) |
| Sparse retrieval | `rank-bm25` (BM25Okapi) |
| Reranker | `BAAI/bge-reranker-v2-m3` CrossEncoder |
| Graph | NetworkX DiGraph (amended_by · references · replaces) |
| API | FastAPI |

---

## Prerequisites

- Python ≥ 3.11
- Docker + Docker Compose
- HuggingFace token (`HF_TOKEN`) — dataset is gated
- One LLM API key
- GPU machine (optional, for the vLLM embedding server)

---

## Quick start

### 1. Install

```bash
git clone https://github.com/ThuanNaN/aio-agentic-rag.git
cd aio-agentic-rag
pip install -e ".[all]"
```

### 2. Configure

```bash
cp .env.example .env
```

Minimum required in `.env`:

```bash
CHROMA_HOST=localhost
CHROMA_PORT=8000

LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

EMBEDDING_BASE_URL=http://localhost:8080/v1

HF_TOKEN=hf_...
```

### 3. Start ChromaDB

```bash
docker compose up -d
```

| Service | URL |
|---------|-----|
| ChromaDB | `http://localhost:8000` |
| Admin UI | `http://localhost:3001` |

> Admin UI connection string: `http://chroma:8000` (Docker-internal)

### 4. Start the embedding server

```bash
vllm serve google/embeddinggemma-300m --port 8080
```

### 5. Ingest documents *(run once)*

```bash
python scripts/ingest.py --sample 1000   # quick smoke test
python scripts/ingest.py                 # full ~153k docs
```

### 6. Start the API

```bash
uvicorn src.api.app:app --reload
# http://localhost:8000/docs
```

### 7. Ask a question

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Luật Đất đai số 31/2024/QH15 có hiệu lực từ ngày nào?",
    "strategy": "agentic",
    "k": 5
  }' | python -m json.tool
```

```json
{
  "question": "Luật Đất đai số 31/2024/QH15 có hiệu lực từ ngày nào?",
  "strategy": "agentic",
  "answer": "Theo Luật Đất đai số 31/2024/QH15, luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2025...",
  "sources": [{"doc_id": "31_2024_QH15", "title": "Luật Đất đai", ...}],
  "latency_ms": 412.3
}
```

---

## Optional: Benchmarking

Compare the agentic strategy against simpler baselines to measure quality gains:

```bash
python scripts/build_eval_set.py         # create gold QA template
# → fill in expected_doc_ids in data/eval/gold_set.json

python scripts/run_benchmark.py --strategy all --sample 8
```

```
=============================================================
Strategy          recall@5       ndcg@10  avg_latency_ms
=============================================================
naive               0.3125        0.2841           87.43
hybrid              0.4375        0.3912          142.17
reranker            0.5000        0.4530          389.62
graph               0.4375        0.4021          201.38
agentic             0.5625        0.5104          412.90
=============================================================
```

---

## Dataset

[`th1nhng0/vietnamese-legal-documents`](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents)

| Config | Rows | Contents |
|--------|------|---------|
| `metadata` | ~153k | Title, type, authority, dates, sector, status |
| `content` | ~153k | Full HTML body |
| `relationships` | ~897k | Cross-document edges (amended_by · references · replaces) |
