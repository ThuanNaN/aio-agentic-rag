"""BM25 index over chunked document texts."""
from __future__ import annotations
import pickle
from pathlib import Path
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    """Thin wrapper around BM25Okapi that keeps track of source documents."""

    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        corpus = [
            _tokenize(c.page_content)
            for c in tqdm(chunks, desc="Tokenizing for BM25", unit="chunk")
        ]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 10) -> list[Document]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_indices]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_bm25_index(chunks: list[Document], save_path: str | None = None) -> BM25Index:
    index = BM25Index(chunks)
    if save_path:
        index.save(save_path)
    return index
