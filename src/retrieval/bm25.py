"""BM25 retrieval wrapper."""
from __future__ import annotations

from langchain_core.documents import Document
from src.indexing.bm25_index import BM25Index


def bm25_search(index: BM25Index, query: str, k: int = 10) -> list[Document]:
    return index.search(query, k=k)
