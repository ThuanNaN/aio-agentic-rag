"""POST /query — answer a single legal question using a chosen RAG strategy."""
from __future__ import annotations

import json
import time
from fastapi import APIRouter, HTTPException
from src.api.schemas import QueryRequest, QueryResponse, SourceDocument, TraceStep

router = APIRouter()


def _ts(icon: str, label: str, detail: str, t0: float) -> dict:
    return {"icon": icon, "label": label, "detail": detail,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def _strategy_retrieve(strategy: str, question: str, k: int) -> tuple[list[dict], float, list[dict]]:
    from src.tools.retrieval_tools import _get_bm25, _get_graph, _get_store

    trace: list[dict] = []
    t_total = time.perf_counter()

    if strategy == "naive":
        from src.retrieval.dense import dense_search
        t0 = time.perf_counter()
        docs = dense_search(_get_store(), question, k=k)
        trace.append(_ts("🔍", "Dense search", f"{len(docs)} docs retrieved", t0))

    elif strategy == "hybrid":
        from src.retrieval.hybrid import hybrid_search
        t0 = time.perf_counter()
        docs = hybrid_search(_get_store(), _get_bm25(), question, k=k, trace=trace)
        trace[-1]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    elif strategy == "reranker":
        from src.retrieval.hybrid import hybrid_search
        from src.retrieval.reranker import rerank
        t0 = time.perf_counter()
        candidates = hybrid_search(_get_store(), _get_bm25(), question, k=k * 2, trace=trace)
        t1 = time.perf_counter()
        docs = rerank(question, candidates, k=k)
        trace.append(_ts("📊", "CrossEncoder reranker", f"{len(candidates)} candidates → top {len(docs)} docs", t1))

    elif strategy == "graph":
        from src.retrieval.graph import graph_search
        t0 = time.perf_counter()
        docs = graph_search(_get_store(), _get_graph(), question, k=k, trace=trace)

    elif strategy == "agentic":
        docs, _, trace = _agentic_retrieve(question, k)
        latency_ms = (time.perf_counter() - t_total) * 1000
        return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs], latency_ms, trace

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    latency_ms = (time.perf_counter() - t_total) * 1000
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs], latency_ms, trace


def _agentic_retrieve(question: str, k: int):
    from src.tools.retrieval_tools import _get_bm25, _get_graph, _get_store

    trace: list[dict] = []
    t_total = time.perf_counter()
    q_lower = question.lower()

    multi_hop_kws = ["sửa đổi", "thay thế", "bãi bỏ", "tham chiếu"]
    temporal_kws  = ["còn hiệu lực", "hết hiệu lực", "sau năm", "trước năm", "từ ngày"]

    if any(kw in q_lower for kw in multi_hop_kws):
        intent = "multi_hop"
        matched = next(kw for kw in multi_hop_kws if kw in q_lower)
    elif any(kw in q_lower for kw in temporal_kws):
        intent = "temporal"
        matched = next(kw for kw in temporal_kws if kw in q_lower)
    else:
        intent = "factual/reasoning"
        matched = None

    trace.append({
        "icon": "🤔",
        "label": "Classify intent",
        "detail": f"`{intent}`" + (f' — detected keyword: "{matched}"' if matched else ""),
        "latency_ms": 0.0,
    })

    if intent == "multi_hop":
        from src.retrieval.graph import graph_search
        trace.append({"icon": "➡️", "label": "Route", "detail": "graph_traverse (multi-hop relationship chain)", "latency_ms": 0.0})
        t0 = time.perf_counter()
        docs = graph_search(_get_store(), _get_graph(), question, k=k, trace=trace)
    elif intent == "temporal":
        from src.retrieval.hybrid import hybrid_search
        trace.append({"icon": "➡️", "label": "Route", "detail": "hybrid_search (temporal filter)", "latency_ms": 0.0})
        t0 = time.perf_counter()
        docs = hybrid_search(_get_store(), _get_bm25(), question, k=k, trace=trace)
        trace[-1]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    else:
        from src.retrieval.hybrid import hybrid_search
        from src.retrieval.reranker import rerank
        trace.append({"icon": "➡️", "label": "Route", "detail": "hybrid_search → reranker (factual/reasoning)", "latency_ms": 0.0})
        t0 = time.perf_counter()
        candidates = hybrid_search(_get_store(), _get_bm25(), question, k=k * 2, trace=trace)
        t1 = time.perf_counter()
        docs = rerank(question, candidates, k=k)
        trace.append(_ts("📊", "CrossEncoder reranker", f"{len(candidates)} candidates → top {len(docs)} docs", t1))

    latency_ms = (time.perf_counter() - t_total) * 1000
    return docs, latency_ms, trace


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        docs_list, latency_ms, trace = _strategy_retrieve(
            request.strategy, request.question, request.k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    from src.tools.retrieval_tools import generate_answer_tool

    t_gen = time.perf_counter()
    try:
        answer = generate_answer_tool.invoke({
            "query": request.question,
            "docs_json": json.dumps(docs_list, ensure_ascii=False),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    trace.append(_ts("✍️", "Generate answer", f"{len(docs_list)} docs in context", t_gen))

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
        trace=[TraceStep(**s) for s in trace],
    )
