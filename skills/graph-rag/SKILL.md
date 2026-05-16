---
name: graph-rag
description: Run GraphRAG strategy — hybrid retrieval expanded via relationship graph traversal (amended_by, references, replaces edges) for multi-hop legal queries, then generate with GPT-4.1.
---

# GraphRAG Strategy

## Overview
Extends Hybrid RAG by traversing the Vietnamese legal document relationship graph after initial
retrieval. Critical for queries that require following amendment chains or cross-referenced decrees
(e.g., "What amended Decree X?", "Which documents reference Article Y?").

## When to Use
Use this skill when asked to run, evaluate, or explain the **graph** RAG strategy, OR when the
query involves:
- Finding amendments or replacements of a law/decree
- Following citation chains between legal documents
- Multi-hop: "the law referenced in X says..."

## Instructions

### Step 1 — Graph-guided retrieval
```
graph_traverse_tool(
    query="<user question>",
    k=5,
    initial_k=3,
    max_hops=2
)
```

This tool:
1. Dense-retrieves 3 seed documents
2. Expands via `amended_by`, `references`, `replaces` edges in the relationship graph (up to 2 hops)
3. Dense-retrieves chunks from all reachable doc_ids
4. Returns the top-5 most relevant merged results

### Step 2 — Generate
```
generate_answer_tool(query="<user question>", docs_json="<JSON from step 1>")
```

### Step 3 — Return
Return the answer. Highlight if any documents were found via graph traversal (different doc_id from
the seed documents).

## Notes
- Graph has ~897k edges covering ~153k documents.
- Edge types used: amended_by, references, replaces.
- max_hops=2 balances coverage vs noise. Use max_hops=1 for faster, more precise results.
- Best strategy for multi-hop queries; on par with reranker for simple factual queries.
