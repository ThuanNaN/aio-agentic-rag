from __future__ import annotations

import chromadb
from chromadb.api import ClientAPI
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from src.indexing.embeddings import VLLMEmbeddings, get_embeddings
from configs import config


def _make_client(host: str | None, port: int) -> ClientAPI:
    if not host:
        raise RuntimeError(
            "CHROMA_HOST is not set. Set CHROMA_HOST (and optionally CHROMA_PORT) "
            "env vars to connect to the Chroma server."
        )
    return chromadb.HttpClient(host=host, port=port)


def get_store() -> Chroma:
    """Return a Chroma instance backed by the HTTP server."""
    client = _make_client(config.chroma.host, config.chroma.port)
    return Chroma(
        client=client,
        collection_name=config.chroma.collection_name,
        embedding_function=get_embeddings(),
    )


def upsert_documents(
    store: Chroma,
    chunks: list[Document],
    emb_fn: VLLMEmbeddings | None = None,
    batch_size: int | None = None,
    upsert_workers: int = 4,
) -> int:
    """
    Embed all chunks in parallel (across all vLLM backends), then upsert to
    Chroma with concurrent workers. Parallel embedding uses embed_workers
    threads each sending batch_size texts — saturating all backends at once.
    """
    if emb_fn is None:
        emb_fn = get_embeddings()

    texts = [c.page_content for c in chunks]
    ids = [f"{c.metadata['doc_id']}_chunk{c.metadata['chunk_index']}" for c in chunks]
    metadatas = [c.metadata for c in chunks]

    print(f"Embedding {len(texts)} chunks in parallel...")
    all_embeddings = emb_fn.embed_parallel(texts)

    collection = store._collection  # type: ignore[attr-defined]
    bs = batch_size if batch_size is not None else config.chroma.upsert_bs

    def _upsert(i: int) -> int:
        end = min(i + bs, len(chunks))
        collection.upsert(
            ids=ids[i:end],
            embeddings=all_embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )
        return end - i

    offsets = list(range(0, len(chunks), bs))
    with tqdm(total=len(chunks), desc="Upserting", unit="chunk") as bar:
        with ThreadPoolExecutor(max_workers=upsert_workers) as executor:
            futures = {executor.submit(_upsert, i): i for i in offsets}
            for future in as_completed(futures):
                bar.update(future.result())

    return len(chunks)


def build_store_from_chunks(chunks: list[Document]) -> Chroma:
    """Index all chunks into a Chroma collection via the HTTP server."""
    client = _make_client(config.chroma.host, config.chroma.port)
    emb_fn = get_embeddings()
    store = Chroma(
        client=client,
        collection_name=config.chroma.collection_name,
        embedding_function=emb_fn,
    )
    upsert_documents(store, chunks, emb_fn=emb_fn)
    return store
