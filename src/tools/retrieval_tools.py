"""LangChain @tool wrappers for retrieval strategies (used by the eval subagent)."""
from __future__ import annotations

import json
from functools import lru_cache
from langchain_core.tools import tool
from configs import config
from src.llm import get_llm


def _doc_to_dict(doc) -> dict:
    return {"page_content": doc.page_content, "metadata": doc.metadata}


@lru_cache(maxsize=1)
def _get_store():
    from src.indexing.chroma_store import get_store
    if not config.chroma.host:
        raise RuntimeError(
            "CHROMA_HOST is not set. Set CHROMA_HOST (and optionally CHROMA_PORT) to connect to the Chroma server."
        )
    return get_store()


@lru_cache(maxsize=1)
def _get_bm25():
    from src.indexing.bm25_index import BM25Index
    return BM25Index.load(config.paths.bm25_index)


@lru_cache(maxsize=1)
def _get_graph():
    from src.retrieval.graph import load_graph
    return load_graph(config.paths.graph_index)


@tool
def dense_search_tool(query: str, k: int = config.retrieval.k, metadata_filter_json: str = "") -> str:
    """
    Perform dense vector search in Chroma.
    Args:
        query: Search query string.
        k: Number of results.
        metadata_filter_json: Optional JSON string of Chroma metadata filter dict.
    Returns:
        JSON string list of {page_content, metadata} dicts.
    """
    from src.retrieval.dense import dense_search

    flt = json.loads(metadata_filter_json) if metadata_filter_json else None
    docs = dense_search(_get_store(), query, k=k, metadata_filter=flt)
    return json.dumps([_doc_to_dict(d) for d in docs], ensure_ascii=False)


@tool
def bm25_search_tool(query: str, k: int = config.retrieval.bm25_k) -> str:
    """
    Perform BM25 keyword search.
    Returns:
        JSON string list of {page_content, metadata} dicts.
    """
    from src.retrieval.bm25 import bm25_search

    docs = bm25_search(_get_bm25(), query, k=k)
    return json.dumps([_doc_to_dict(d) for d in docs], ensure_ascii=False)


@tool
def hybrid_search_tool(
    query: str,
    k: int = config.retrieval.k,
    bm25_k: int = config.retrieval.bm25_k,
    dense_k: int = config.retrieval.dense_k,
    metadata_filter_json: str = "",
) -> str:
    """
    Perform hybrid BM25 + dense retrieval with RRF fusion.
    Returns:
        JSON string list of {page_content, metadata} dicts.
    """
    from src.retrieval.hybrid import hybrid_search

    flt = json.loads(metadata_filter_json) if metadata_filter_json else None
    docs = hybrid_search(
        _get_store(), _get_bm25(), query,
        k=k, bm25_k=bm25_k, dense_k=dense_k,
        metadata_filter=flt,
    )
    return json.dumps([_doc_to_dict(d) for d in docs], ensure_ascii=False)


@tool
def rerank_tool(query: str, docs_json: str, k: int = config.retrieval.k) -> str:
    """
    Re-rank a list of documents with a CrossEncoder.
    Args:
        query: Original query.
        docs_json: JSON string list of {page_content, metadata} dicts.
        k: Top-k to return after reranking.
    Returns:
        JSON string list of reranked {page_content, metadata} dicts.
    """
    from langchain_core.documents import Document
    from src.retrieval.reranker import rerank

    raw = json.loads(docs_json)
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw]
    reranked = rerank(query, docs, k=k, model_name=config.retrieval.reranker_model)
    return json.dumps([_doc_to_dict(d) for d in reranked], ensure_ascii=False)


@tool
def graph_traverse_tool(
    query: str,
    k: int = config.retrieval.k,
    initial_k: int = 3,
    max_hops: int = config.retrieval.graph_max_hops,
) -> str:
    """
    Retrieve documents using graph-guided multi-hop retrieval.
    Seed with dense search, then expand via relationship graph edges.
    Returns:
        JSON string list of {page_content, metadata} dicts.
    """
    from src.retrieval.graph import graph_search

    docs = graph_search(
        _get_store(), _get_graph(), query,
        k=k, initial_k=initial_k, max_hops=max_hops,
        edge_types=set(config.retrieval.graph_edge_types),
    )
    return json.dumps([_doc_to_dict(d) for d in docs], ensure_ascii=False)


@tool
def generate_answer_tool(query: str, docs_json: str) -> str:
    """
    Generate an answer from retrieved documents using the configured LLM.
    Args:
        query: User question.
        docs_json: JSON string list of {page_content, metadata} dicts.
    Returns:
        Generated answer string with citations.
    """
    docs_list = json.loads(docs_json)
    context_parts = []
    for i, doc in enumerate(docs_list, 1):
        doc_id = doc["metadata"].get("doc_id", "unknown")
        context_parts.append(f"[{i}] (doc_id: {doc_id})\n{doc['page_content']}")
    context = "\n\n".join(context_parts)

    llm = get_llm()
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý pháp lý chuyên về văn bản pháp luật Việt Nam. "
                "Hãy trả lời câu hỏi dựa trên các đoạn văn bản pháp luật được cung cấp. "
                "Trích dẫn số hiệu văn bản khi có thể. "
                "Nếu không tìm thấy thông tin trong tài liệu, hãy nói rõ."
            ),
        },
        {
            "role": "user",
            "content": f"Tài liệu tham khảo:\n\n{context}\n\nCâu hỏi: {query}",
        },
    ]
    response = llm.invoke(messages)
    return response.content
