"""CrossEncoder reranker for re-scoring retrieved documents."""
from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_cached: dict[str, CrossEncoder] = {}


def _get_model(model_name: str) -> CrossEncoder:
    if model_name not in _cached:
        _cached[model_name] = CrossEncoder(model_name)
    return _cached[model_name]


def rerank(
    query: str,
    docs: list[Document],
    k: int = 5,
    model_name: str = _DEFAULT_MODEL,
) -> list[Document]:
    """Re-score docs with a CrossEncoder and return top-k."""
    if not docs:
        return []
    model = _get_model(model_name)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]
