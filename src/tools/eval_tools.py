"""LangChain @tool wrappers for the evaluation subagent."""
from __future__ import annotations

import json
from langchain_core.tools import tool


@tool
def run_strategy_eval_tool(
    strategy: str,
    sample_n: int = 0,
    recall_k: int = 5,
    ndcg_k: int = 10,
    use_ragas: bool = False,
) -> str:
    """
    Run a single RAG strategy against the gold QA set and return metric summary.
    Args:
        strategy: One of naive, hybrid, reranker, graph, agentic.
        sample_n: Number of gold queries to use (0 = all).
        recall_k: k for Recall@k.
        ndcg_k: k for nDCG@k.
        use_ragas: If True, generate LLM answers and compute RAGAS metrics.
    Returns:
        JSON string with metric scores.
    """
    from src.evaluation.benchmark import run_benchmark

    n = sample_n if sample_n > 0 else None
    results = run_benchmark(
        strategies=[strategy],
        recall_k=recall_k,
        ndcg_k=ndcg_k,
        sample_n=n,
        use_ragas=use_ragas,
    )
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def run_full_benchmark_tool(
    sample_n: int = 0,
    recall_k: int = 5,
    ndcg_k: int = 10,
    use_ragas: bool = False,
) -> str:
    """
    Run all 5 strategies (naive, hybrid, reranker, graph, agentic) against the gold QA set.
    Args:
        sample_n: Number of gold queries to use (0 = all).
        recall_k: k for Recall@k.
        ndcg_k: k for nDCG@k.
        use_ragas: If True, generate LLM answers and compute RAGAS metrics.
    Returns:
        JSON string with per-strategy metric scores.
    """
    from src.evaluation.benchmark import run_benchmark

    n = sample_n if sample_n > 0 else None
    results = run_benchmark(
        strategies=["naive", "hybrid", "reranker", "graph", "agentic"],
        recall_k=recall_k,
        ndcg_k=ndcg_k,
        sample_n=n,
        use_ragas=use_ragas,
    )
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def compute_recall_tool(retrieved_ids_json: str, relevant_ids_json: str, k: int = 5) -> str:
    """
    Compute Recall@k for a single query.
    Args:
        retrieved_ids_json: JSON list of retrieved doc_id strings.
        relevant_ids_json: JSON list of relevant doc_id strings.
        k: cutoff.
    Returns:
        Recall@k score as a string.
    """
    from src.evaluation.metrics import recall_at_k

    retrieved = json.loads(retrieved_ids_json)
    relevant = json.loads(relevant_ids_json)
    score = recall_at_k(retrieved, relevant, k)
    return str(score)


@tool
def compute_ndcg_tool(retrieved_ids_json: str, relevant_ids_json: str, k: int = 10) -> str:
    """
    Compute nDCG@k for a single query.
    Args:
        retrieved_ids_json: JSON list of retrieved doc_id strings.
        relevant_ids_json: JSON list of relevant doc_id strings.
        k: cutoff.
    Returns:
        nDCG@k score as a string.
    """
    from src.evaluation.metrics import ndcg_at_k

    retrieved = json.loads(retrieved_ids_json)
    relevant = json.loads(relevant_ids_json)
    score = ndcg_at_k(retrieved, relevant, k)
    return str(score)


@tool
def run_ragas_tool(
    questions_json: str,
    answers_json: str,
    contexts_json: str,
    ground_truths_json: str,
) -> str:
    """
    Run RAGAS evaluation metrics (faithfulness, relevancy, context precision).
    Args:
        questions_json: JSON list of question strings.
        answers_json: JSON list of generated answer strings.
        contexts_json: JSON list of lists of context strings (one list per question).
        ground_truths_json: JSON list of ground truth answer strings.
    Returns:
        JSON string with RAGAS metric scores.
    """
    from src.evaluation.metrics import compute_ragas_metrics

    result = compute_ragas_metrics(
        json.loads(questions_json),
        json.loads(answers_json),
        json.loads(contexts_json),
        json.loads(ground_truths_json),
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
