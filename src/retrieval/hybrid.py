"""Reciprocal Rank Fusion (RRF) hybrid retrieval: BM25 + dense."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.indexing.bm25_index import BM25Index
from src.retrieval.bm25 import bm25_search
from src.retrieval.dense import dense_search


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


def hybrid_search(
    store: Chroma,
    bm25_index: BM25Index,
    query: str,
    k: int = 5,
    bm25_k: int = 10,
    dense_k: int = 10,
    rrf_k: int = 60,
    metadata_filter: dict | None = None,
) -> list[Document]:
    """
    Fuse BM25 and dense results via RRF.
    Returns top-k unique documents ranked by combined score.
    """
    bm25_results = bm25_search(bm25_index, query, k=bm25_k)
    dense_results = dense_search(store, query, k=dense_k, metadata_filter=metadata_filter)

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(bm25_results):
        key = f"{doc.metadata.get('doc_id', '')}_{doc.metadata.get('chunk_index', 0)}"
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank, rrf_k)
        doc_map[key] = doc

    for rank, doc in enumerate(dense_results):
        key = f"{doc.metadata.get('doc_id', '')}_{doc.metadata.get('chunk_index', 0)}"
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank, rrf_k)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:k]]
