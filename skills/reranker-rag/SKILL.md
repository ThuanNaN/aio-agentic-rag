---
name: reranker-rag
description: Run Reranker RAG strategy — hybrid retrieval followed by CrossEncoder (ms-marco-MiniLM-L-6-v2) reranking, then generate with GPT-4.1.
---

# Reranker RAG Strategy

## Overview
Extends Hybrid RAG by adding a CrossEncoder reranking step. The CrossEncoder jointly scores each
(query, document) pair for more accurate relevance estimation than embedding similarity alone.

## When to Use
Use this skill when asked to run, evaluate, or explain the **reranker** RAG strategy.

## Instructions

### Step 1 — Retrieve candidates (hybrid)
Retrieve more candidates than needed (2×k) for the reranker to work with:

```
hybrid_search_tool(query="<user question>", k=10, bm25_k=15, dense_k=15)
```

### Step 2 — Rerank
Pass the candidates JSON to `rerank_tool` with the final target k:

```
rerank_tool(query="<user question>", docs_json="<JSON from step 1>", k=5)
```

The CrossEncoder re-scores all candidates and returns the top-k.

### Step 3 — Generate
```
generate_answer_tool(query="<user question>", docs_json="<JSON from step 2>")
```

### Step 4 — Return
Return the answer with source doc_ids.

## Notes
- Adds ~200-500ms latency vs hybrid due to CrossEncoder inference.
- Consistently improves precision over hybrid RAG, especially for nuanced legal questions.
- Reranker model: cross-encoder/ms-marco-MiniLM-L-6-v2 (loaded once, cached in memory).
