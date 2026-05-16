"""Chunker module to split cleaned documents into chunks for embedding and retrieval."""
from __future__ import annotations
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from configs import Config


def build_splitter(config: Config) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        separators=config.chunking.separators,
        keep_separator=True,
    )


def chunk_documents(
    docs: list[Document],
    config: Config,
) -> list[Document]:
    """
    Split cleaned documents into chunks, preserving all metadata fields plus
    a `chunk_index` counter per source document.
    """
    splitter = build_splitter(config)
    chunks: list[Document] = []
    for doc in tqdm(docs, desc="Chunking documents", unit="doc"):
        splits = splitter.split_documents([doc])
        for i, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = i
            chunks.append(chunk)
    return chunks
