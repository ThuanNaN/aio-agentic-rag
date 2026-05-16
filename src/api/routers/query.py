"""POST /query — answer a single legal question using a chosen RAG strategy."""
from __future__ import annotations

import json
import time
from fastapi import APIRouter, HTTPException
from src.api.schemas import QueryRequest, QueryResponse, SourceDocument

router = APIRouter()


def _strategy_retrieve(strategy: str, question: str, k: int) -> tuple[list[dict], float]:
    from src.tools.retrieval_tools import (
        _get_bm25,
        _get_graph,
        _get_store,
    )

    t0 = time.perf_counter()

    if strategy == "naive":
        from src.retrieval.dense import dense_search
        docs = dense_search(_get_store(), question, k=k)

    elif strategy == "hybrid":
        from src.retrieval.hybrid import hybrid_search
        docs = hybrid_search(_get_store(), _get_bm25(), question, k=k)

    elif strategy == "reranker":
        from src.retrieval.hybrid import hybrid_search
        from src.retrieval.reranker import rerank
        candidates = hybrid_search(_get_store(), _get_bm25(), question, k=k * 2)
        docs = rerank(question, candidates, k=k)

    elif strategy == "graph":
        from src.retrieval.graph import graph_search
        docs = graph_search(_get_store(), _get_graph(), question, k=k)

    elif strategy == "agentic":
        # For single-query agentic use the orchestrator agent
        return _agentic_retrieve(question, k)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    latency_ms = (time.perf_counter() - t0) * 1000
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs], latency_ms


def _agentic_retrieve(question: str, k: int) -> tuple[list[dict], float]:
    from src.tools.retrieval_tools import (
        _get_bm25,
        _get_graph,
        _get_store,
    )
    # Simplified agentic: classify + route (full agent used via /benchmark endpoint)
    t0 = time.perf_counter()
    q_lower = question.lower()
    if any(kw in q_lower for kw in ["sửa đổi", "thay thế", "bãi bỏ", "tham chiếu"]):
        from src.retrieval.graph import graph_search
        docs = graph_search(_get_store(), _get_graph(), question, k=k)
    elif any(kw in q_lower for kw in ["còn hiệu lực", "hết hiệu lực", "sau năm", "trước năm"]):
        from src.retrieval.hybrid import hybrid_search
        docs = hybrid_search(_get_store(), _get_bm25(), question, k=k)
    else:
        from src.retrieval.hybrid import hybrid_search
        from src.retrieval.reranker import rerank
        candidates = hybrid_search(_get_store(), _get_bm25(), question, k=k * 2)
        docs = rerank(question, candidates, k=k)
    latency_ms = (time.perf_counter() - t0) * 1000
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs], latency_ms


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        docs_list, latency_ms = _strategy_retrieve(request.strategy, request.question, request.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    from src.tools.retrieval_tools import generate_answer_tool

    try:
        answer = generate_answer_tool.invoke({
            "query": request.question,
            "docs_json": json.dumps(docs_list, ensure_ascii=False),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    sources = [
        SourceDocument(
            doc_id=d["metadata"].get("doc_id", ""),
            title=d["metadata"].get("title", ""),
            authority=d["metadata"].get("authority", ""),
            issue_date=d["metadata"].get("issue_date", ""),
            chunk_index=d["metadata"].get("chunk_index", 0),
            excerpt=d["page_content"][:200],
        )
        for d in docs_list
    ]

    return QueryResponse(
        question=request.question,
        strategy=request.strategy,
        answer=answer,
        sources=sources,
        latency_ms=latency_ms,
    )
