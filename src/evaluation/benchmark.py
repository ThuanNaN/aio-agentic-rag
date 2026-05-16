"""Run all 5 RAG strategies against the gold set and produce a results table."""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from langchain_core.documents import Document
from configs import config
from src.evaluation.gold_set import load_gold_set
from src.evaluation.metrics import aggregate_metrics, ndcg_at_k, recall_at_k


def _retrieve_naive(query: str, k: int = 5) -> list[Document]:
    from src.retrieval.dense import dense_search
    from src.tools.retrieval_tools import _get_store
    return dense_search(_get_store(), query, k=k)


def _retrieve_hybrid(query: str, k: int = 5) -> list[Document]:
    from src.retrieval.hybrid import hybrid_search
    from src.tools.retrieval_tools import _get_bm25, _get_store
    return hybrid_search(_get_store(), _get_bm25(), query, k=k)


def _retrieve_reranker(query: str, k: int = 5) -> list[Document]:
    from src.retrieval.hybrid import hybrid_search
    from src.retrieval.reranker import rerank
    from src.tools.retrieval_tools import _get_bm25, _get_store
    candidates = hybrid_search(_get_store(), _get_bm25(), query, k=k * 2)
    return rerank(query, candidates, k=k)


def _retrieve_graph(query: str, k: int = 5) -> list[Document]:
    from src.retrieval.graph import graph_search
    from src.tools.retrieval_tools import _get_graph, _get_store
    return graph_search(_get_store(), _get_graph(), query, k=k)


def _retrieve_agentic(query: str, k: int = 5) -> list[Document]:
    """Classify query type, route to the optimal retrieval path, retry on poor results."""
    from src.retrieval.dense import dense_search
    from src.retrieval.graph import graph_search
    from src.retrieval.hybrid import hybrid_search
    from src.retrieval.reranker import rerank
    from src.tools.retrieval_tools import _get_bm25, _get_graph, _get_store

    store = _get_store()
    bm25 = _get_bm25()
    graph = _get_graph()
    q_lower = query.lower()

    # Step 1: classify and route
    if any(kw in q_lower for kw in ["sửa đổi", "thay thế", "bãi bỏ", "tham chiếu", "được quy định tại"]):
        # multi-hop: follow amendment/reference chains
        docs = graph_search(store, graph, query, k=k, initial_k=3, max_hops=2)
    elif any(kw in q_lower for kw in ["còn hiệu lực", "hết hiệu lực", "sau năm", "trước năm", "từ ngày"]):
        # temporal: hybrid with date-aware retrieval
        docs = hybrid_search(store, bm25, query, k=k)
    else:
        # factual / reasoning: hybrid then rerank for precision
        candidates = hybrid_search(store, bm25, query, k=k * 2)
        docs = rerank(query, candidates, k=k)

    # Step 2: grade — retry with dense if fewer than 2 docs returned
    if len(docs) < 2:
        docs = dense_search(store, query, k=k)

    return docs


STRATEGIES: dict[str, Callable] = {
    "naive": _retrieve_naive,
    "hybrid": _retrieve_hybrid,
    "reranker": _retrieve_reranker,
    "graph": _retrieve_graph,
    "agentic": _retrieve_agentic,
}


def run_benchmark(
    strategies: list[str] | None = None,
    gold_path: str | None = None,
    recall_k: int | None = None,
    ndcg_k: int | None = None,
    output_dir: str | None = None,
    sample_n: int | None = None,
) -> dict[str, dict]:
    """
    Run each strategy against the gold set and return per-strategy metric summaries.
    Also writes a timestamped CSV to output_dir.
    """
    _gold_path = gold_path or config.evaluation.gold_set_path
    _recall_k = recall_k or config.evaluation.recall_k
    _ndcg_k = ndcg_k or config.evaluation.ndcg_k
    _output_dir = output_dir or config.paths.results

    entries = load_gold_set(_gold_path)
    if sample_n:
        entries = entries[:sample_n]

    selected = strategies or list(STRATEGIES.keys())
    all_results: dict[str, dict] = {}
    rows: list[dict] = []

    for strategy_name in selected:
        if strategy_name not in STRATEGIES:
            print(f"Unknown strategy: {strategy_name}, skipping.")
            continue
        retrieve_fn = STRATEGIES[strategy_name]
        per_query: list[dict[str, float]] = []
        latencies: list[float] = []

        for entry in entries:
            question = entry["question"]
            relevant_ids: list[str] = entry.get("expected_doc_ids", [])

            t0 = time.perf_counter()
            docs = retrieve_fn(question, k=max(_recall_k, _ndcg_k))
            latency_ms = (time.perf_counter() - t0) * 1000

            retrieved_ids = [d.metadata.get("doc_id", "") for d in docs]
            per_query.append(
                {
                    f"recall@{_recall_k}": recall_at_k(retrieved_ids, relevant_ids, _recall_k),
                    f"ndcg@{_ndcg_k}": ndcg_at_k(retrieved_ids, relevant_ids, _ndcg_k),
                }
            )
            latencies.append(latency_ms)

        summary = aggregate_metrics(per_query)
        summary["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0.0
        all_results[strategy_name] = summary
        rows.append({"strategy": strategy_name, **summary})

    # Write CSV
    Path(_output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(_output_dir) / f"benchmark_{ts}.csv"
    if rows:
        fieldnames = ["strategy"] + [k for k in rows[0] if k != "strategy"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Results saved to {csv_path}")

    # Print table
    _print_table(all_results)
    return all_results


def _print_table(results: dict[str, dict]) -> None:
    if not results:
        return
    strategies = list(results.keys())
    metrics = list(next(iter(results.values())).keys())
    col_w = 18
    header = f"{'Strategy':<15}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for strategy, scores in results.items():
        row = f"{strategy:<15}" + "".join(f"{scores[m]:>{col_w}.4f}" for m in metrics)
        print(row)
    print("=" * len(header) + "\n")
