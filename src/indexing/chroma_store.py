from __future__ import annotations

import chromadb
import json
from chromadb.api import ClientAPI
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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
    start_offset: int = 0,
    progress_path: str | None = None,
    segment_size: int = 10000,
) -> int:
    """
    Embed and upsert chunks into Chroma in segments, checkpointing after each.

    When progress_path is given, the file records {"last_offset": N, "total": M}
    so an interrupted run can resume from the last completed segment boundary.
    Set start_offset to the value read from that file to skip already-done work.
    """
    if emb_fn is None:
        emb_fn = get_embeddings()

    all_chunks = chunks[start_offset:]
    if not all_chunks:
        print("All chunks already indexed — nothing to do.")
        return 0

    texts_all = [c.page_content for c in chunks]
    ids_all = [f"{c.metadata['doc_id']}_chunk{c.metadata['chunk_index']}" for c in chunks]
    metas_all = [c.metadata for c in chunks]

    collection = store._collection  # type: ignore[attr-defined]
    bs = batch_size if batch_size is not None else config.chroma.upsert_bs
    total = len(chunks)
    processed = 0

    seg_start = start_offset
    while seg_start < total:
        seg_end = min(seg_start + segment_size, total)
        seg_texts = texts_all[seg_start:seg_end]
        seg_ids = ids_all[seg_start:seg_end]
        seg_metas = metas_all[seg_start:seg_end]

        print(f"Embedding segment [{seg_start}:{seg_end}] ({len(seg_texts)} chunks)...")
        seg_embeddings = emb_fn.embed_parallel(seg_texts)

        def _upsert(i: int, _seg_ids=seg_ids, _seg_embs=seg_embeddings,
                    _seg_texts=seg_texts, _seg_metas=seg_metas) -> int:
            end = min(i + bs, len(_seg_texts))
            collection.upsert(
                ids=_seg_ids[i:end],
                embeddings=_seg_embs[i:end],
                documents=_seg_texts[i:end],
                metadatas=_seg_metas[i:end],
            )
            return end - i

        offsets = list(range(0, len(seg_texts), bs))
        with tqdm(total=len(seg_texts), desc=f"Upserting [{seg_start}:{seg_end}]", unit="chunk") as bar:
            with ThreadPoolExecutor(max_workers=upsert_workers) as executor:
                futures = {executor.submit(_upsert, i): i for i in offsets}
                for future in as_completed(futures):
                    bar.update(future.result())

        processed += len(seg_texts)
        seg_start = seg_end

        if progress_path:
            Path(progress_path).write_text(
                json.dumps({"last_offset": seg_start, "total": total})
            )

    return processed


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
