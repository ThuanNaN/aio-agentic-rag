"""Retrieval and generation evaluation metrics."""
from __future__ import annotations

import math
from typing import Any


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Fraction of relevant docs found in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_set
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Average Precision for a single query."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            hits += 1
            total += hits / rank
    return total / len(relevant_ids)


def aggregate_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average per-query metric dicts into a single summary dict."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}


def compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, Any]:
    """Run RAGAS evaluation. Returns a dict of metric name → score."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_faithfulness, answer_relevancy, context_precision

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(dataset, metrics=[answer_faithfulness, answer_relevancy, context_precision])
        return dict(result)
    except Exception as e:
        return {"ragas_error": str(e)}
