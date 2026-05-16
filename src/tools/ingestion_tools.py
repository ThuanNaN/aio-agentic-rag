"""LangChain @tool wrappers for the ingestion subagent."""
from __future__ import annotations

import json
from pathlib import Path
from langchain_core.tools import tool
from configs import config


@tool
def load_dataset_tool(sample_size: int = 0) -> str:
    """
    Download the Vietnamese legal documents dataset from HuggingFace and save raw documents
    to the configured data_processed path.

    Args:
        sample_size: Number of documents to load (0 = use full dataset from config).
    Returns:
        Summary string with document count.
    """
    from src.ingestion.loader import load_documents

    docs = load_documents(config, sample_size=sample_size if sample_size > 0 else None)
    out_path = Path(config.paths.data_processed) / "raw_docs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    out_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2))
    return f"Loaded {len(docs)} documents → saved to {out_path}"


@tool
def load_relationships_tool() -> str:
    """
    Download the cross-document relationships config and save to the configured path.
    If raw_docs.json exists (produced by load_dataset_tool), only relationships that
    touch those doc IDs are kept — so sampling is respected end-to-end.
    Returns:
        Summary string with relationship count.
    """
    from src.ingestion.loader import load_relationships

    raw_path = Path(config.paths.data_processed) / "raw_docs.json"
    doc_ids: set[str] | None = None
    if raw_path.exists():
        data = json.loads(raw_path.read_text())
        doc_ids = {d["metadata"].get("doc_id", "") for d in data} or None

    rels = load_relationships(config, doc_ids=doc_ids)
    out_path = Path(config.paths.data_processed) / "relationships.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rels, ensure_ascii=False, indent=2))
    return f"Loaded {len(rels)} relationships → saved to {out_path}"


@tool
def clean_docs_tool() -> str:
    """
    Strip HTML and normalize encoding for all documents in the raw_docs.json file.
    Saves cleaned docs to cleaned_docs.json in the configured data_processed path.
    Returns:
        Summary string.
    """
    from langchain_core.documents import Document
    from src.ingestion.cleaner import clean_documents

    raw_path = Path(config.paths.data_processed) / "raw_docs.json"
    data = json.loads(raw_path.read_text())
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    cleaned = clean_documents(docs, workers=config.chunking.clean_workers)
    out_path = Path(config.paths.data_processed) / "cleaned_docs.json"
    serialized = [{"page_content": d.page_content, "metadata": d.metadata} for d in cleaned]
    out_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2))
    return f"Cleaned {len(cleaned)} documents → saved to {out_path}"


@tool
def chunk_docs_tool() -> str:
    """
    Chunk cleaned documents using article-aware splitting (config from chunking section).
    Reads from cleaned_docs.json, saves to chunks.json in the configured data_processed path.
    Returns:
        Summary string with chunk count.
    """
    from langchain_core.documents import Document
    from src.ingestion.chunker import chunk_documents

    cleaned_path = Path(config.paths.data_processed) / "cleaned_docs.json"
    data = json.loads(cleaned_path.read_text())
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    chunks = chunk_documents(docs, config)
    out_path = Path(config.paths.data_processed) / "chunks.json"
    serialized = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    out_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2))
    return f"Created {len(chunks)} chunks from {len(docs)} docs → saved to {out_path}"


@tool
def build_chroma_tool() -> str:
    """
    Embed chunks and upsert into a Chroma collection via the HTTP server.
    Reads from chunks.json in the configured data_processed path.
    Connection settings (host, port, collection) come from the chroma config section.
    Returns:
        Summary with count indexed.
    """
    from langchain_core.documents import Document
    from src.indexing.chroma_store import build_store_from_chunks

    chunks_path = Path(config.paths.data_processed) / "chunks.json"
    data = json.loads(chunks_path.read_text())
    chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    build_store_from_chunks(chunks)
    return f"Indexed {len(chunks)} chunks into Chroma @ {config.chroma.host}:{config.chroma.port}"


@tool
def build_bm25_tool() -> str:
    """
    Build a BM25 index over all chunks and save to the configured bm25_index path.
    Returns:
        Summary string.
    """
    from langchain_core.documents import Document
    from src.indexing.bm25_index import build_bm25_index

    chunks_path = Path(config.paths.data_processed) / "chunks.json"
    data = json.loads(chunks_path.read_text())
    chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    build_bm25_index(chunks, save_path=config.paths.bm25_index)
    return f"Built BM25 index over {len(chunks)} chunks → {config.paths.bm25_index}"


@tool
def build_graph_tool() -> str:
    """
    Build a NetworkX relationship graph from relationships.json and save to the configured
    graph_index path.
    Returns:
        Summary with node/edge counts.
    """
    from src.retrieval.graph import build_graph

    rels_path = Path(config.paths.data_processed) / "relationships.json"
    rels = json.loads(rels_path.read_text())
    G = build_graph(rels, save_path=config.paths.graph_index)
    return f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {config.paths.graph_index}"
