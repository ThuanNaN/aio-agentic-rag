---
name: hybrid-rag
description: Run Hybrid RAG strategy — RRF fusion of BM25 keyword search and dense vector search, then generate with GPT-4.1.
---

# Hybrid RAG Strategy

## Overview
Combines BM25 (exact keyword matching) and dense vector search using Reciprocal Rank Fusion (RRF). 
Better than naive RAG for queries containing specific legal terms, document numbers, or article references.

## When to Use
Use this skill when asked to run, evaluate, or explain the **hybrid** RAG strategy.

## Instructions

### Step 1 — Retrieve (hybrid)
Call `hybrid_search_tool` with the user's query:

```
hybrid_search_tool(query="<user question>", k=5, bm25_k=10, dense_k=10)
```

For queries with known date ranges or authorities, add a metadata filter:
```
hybrid_search_tool(
    query="<user question>",
    k=5,
    metadata_filter_json='{"issue_date": {"$gte": "2020-01-01"}}'
)
```

The tool returns a JSON list of `{page_content, metadata}` dicts fused via RRF.

### Step 2 — Generate
```
generate_answer_tool(query="<user question>", docs_json="<JSON from step 1>")
```

### Step 3 — Return
Return the answer with source doc_ids cited.

## Notes
- RRF constant is 60 (default). Higher values reduce the influence of top-ranked results.
- Significantly better than naive RAG on Vietnamese legal queries with specific article/decree numbers.
