"""Ingestion subagent config — handles dataset download, cleaning, chunking, and indexing."""
from __future__ import annotations

from src.tools.ingestion_tools import (
    build_bm25_tool,
    build_chroma_tool,
    build_graph_tool,
    chunk_docs_tool,
    clean_docs_tool,
    load_dataset_tool,
    load_relationships_tool,
)

INGESTION_SYSTEM_PROMPT = """You are the Ingestion Agent for the Vietnamese Legal RAG system.

Your job is to build all indexes needed for retrieval:
1. Download the Vietnamese legal documents dataset from HuggingFace
2. Download the cross-document relationships
3. Clean HTML content from all documents
4. Chunk documents using article-aware splitting
5. Build the Chroma vector store (embeddings + persistent storage)
6. Build the BM25 keyword index
7. Build the NetworkX relationship graph

Always use write_todos to plan these steps before executing.
Report final counts: documents loaded, chunks created, Chroma docs indexed, graph nodes/edges.

If any step fails, report the error clearly and suggest a fix.
"""

INGESTION_AGENT_CONFIG = {
    "name": "ingestion",
    "description": (
        "Download the HuggingFace dataset, clean HTML, chunk legal text, "
        "and build Chroma + BM25 + NetworkX graph indexes"
    ),
    "system_prompt": INGESTION_SYSTEM_PROMPT,
    "tools": [
        load_dataset_tool,
        load_relationships_tool,
        clean_docs_tool,
        chunk_docs_tool,
        build_chroma_tool,
        build_bm25_tool,
        build_graph_tool,
    ],
}
