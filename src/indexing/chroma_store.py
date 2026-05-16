from __future__ import annotations

import chromadb
from chromadb.api import ClientAPI
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
) -> int:
    """
    Embed and upsert chunks in the same batch loop so peak RAM stays at one
    batch rather than holding all embeddings in memory at once.
    """
    if emb_fn is None:
        emb_fn = get_embeddings()

    texts = [c.page_content for c in chunks]
    ids = [f"{c.metadata['doc_id']}_chunk{c.metadata['chunk_index']}" for c in chunks]
    metadatas = [c.metadata for c in chunks]

    collection = store._collection  # type: ignore[attr-defined]
    bs = batch_size if batch_size is not None else config.chroma.upsert_bs
    with tqdm(total=len(chunks), desc="Embedding & Upserting", unit="chunk") as bar:
        for i in range(0, len(chunks), bs):
            end = min(i + bs, len(chunks))
            batch_texts = texts[i:end]
            embeddings = emb_fn._embed_batch(batch_texts)
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=metadatas[i:end],
            )
            bar.update(end - i)

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
