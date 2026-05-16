"""Dense vector retrieval via Chroma."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document


def dense_search(
    store: Chroma,
    query: str,
    k: int = 5,
    metadata_filter: dict | None = None,
) -> list[Document]:
    """Return top-k chunks by cosine similarity, with optional metadata filter."""
    kwargs: dict = {"k": k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    return store.similarity_search(query, **kwargs)
