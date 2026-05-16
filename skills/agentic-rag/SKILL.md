---
name: agentic-rag
description: Run Agentic RAG strategy — classify the query type, apply metadata filters, choose the best retrieval path (dense/hybrid/graph), grade relevance, and retry with query rewriting if needed, then generate with GPT-4.1.
---

# Agentic RAG Strategy

## Overview
The most powerful strategy. Instead of always using the same retrieval pipeline, the agent:
1. Classifies the query type (factual / multi_hop / temporal / reasoning)
2. Extracts metadata filters (date ranges, authority, document type)
3. Routes to the optimal retrieval strategy
4. Grades retrieved document relevance and retries with a rewritten query if results are poor

## When to Use
Use this skill when asked to run, evaluate, or explain the **agentic** RAG strategy, OR when
handling difficult queries that other strategies may fail on.

## Instructions

### Step 1 — Classify the query
Analyze the user's question and assign one of these types:

| Type | Signals |
|------|---------|
| `factual` | Asks for a specific fact, date, or definition |
| `multi_hop` | Asks what amended/references/replaced something |
| `temporal` | Contains "còn hiệu lực", "sau năm X", "trước năm X" |
| `reasoning` | Requires applying rules to a scenario |

### Step 2 — Extract metadata filters
Look for:
- Date mentions → `issue_date` or `effective_date` filter
- Authority names → `authority` filter
- Document type → `doc_type` filter (luật, nghị định, thông tư, quyết định)

Build a JSON filter string, e.g.:
```json
{"doc_type": "Nghị định", "issue_date": {"$gte": "2020-01-01"}}
```

### Step 3 — Choose retrieval path

| Query type | Retrieval tool | Notes |
|------------|---------------|-------|
| `factual` | `hybrid_search_tool` | Apply metadata_filter if available |
| `multi_hop` | `graph_traverse_tool` | max_hops=2 |
| `temporal` | `hybrid_search_tool` | Must apply date range filter |
| `reasoning` | `hybrid_search_tool` then `rerank_tool` | Higher k, rerank for precision |

### Step 4 — Grade relevance
After retrieval, check: do the retrieved docs actually contain information relevant to the query?
- If YES (≥2 docs seem relevant): proceed to generation
- If NO: rewrite the query (simplify, use synonyms, remove specifics) and retry once with `dense_search_tool`

Maximum 2 retrieval attempts total.

### Step 5 — Generate
```
generate_answer_tool(query="<original user question>", docs_json="<best retrieved docs JSON>")
```

### Step 6 — Return
Return the answer with:
- Source doc_ids
- Query type classification
- Which retrieval strategy was used
- Whether a retry was needed

## Notes
- This strategy has the highest latency but the best answer quality.
- The classification and routing logic is the key differentiator vs. the simpler strategies.
- For temporal queries, ALWAYS apply a date filter — otherwise you'll get outdated regulations.
