---
name: naive-rag
description: Run Naive RAG strategy — pure dense vector search via Chroma with multilingual-e5-large embeddings, then generate with GPT-4.1.
---

# Naive RAG Strategy

## Overview
The simplest baseline: embed the query, search Chroma for the top-k most similar chunks, pass them directly to GPT-4.1 for generation.

## When to Use
Use this skill when asked to run, evaluate, or explain the **naive** or **baseline** RAG strategy.

## Instructions

### Step 1 — Retrieve
Call `dense_search_tool` with the user's query and `k=5`.

```
dense_search_tool(query="<user question>", k=5)
```

The tool returns a JSON list of `{page_content, metadata}` dicts.

### Step 2 — Generate
Pass the retrieved docs JSON to `generate_answer_tool` along with the original query.

```
generate_answer_tool(query="<user question>", docs_json="<JSON from step 1>")
```

### Step 3 — Return
Return the generated answer along with source `doc_id` values from the retrieved docs metadata.

## Notes
- No BM25, no reranking, no graph traversal.
- Good for simple factual queries; poor on multi-hop or keyword-heavy queries.
- Latency: fastest of all strategies.
