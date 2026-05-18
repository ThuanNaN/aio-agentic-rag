"""Pydantic request/response models for the FastAPI service."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

Strategy = Literal["naive", "hybrid", "reranker", "graph", "agentic"]


class QueryRequest(BaseModel):
    question: str = Field(..., description="Legal question in Vietnamese")
    strategy: Strategy = Field("hybrid", description="RAG strategy to use")
    k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")


class SourceDocument(BaseModel):
    doc_id: str
    title: str = ""
    authority: str = ""
    issue_date: str = ""
    chunk_index: int = 0
    excerpt: str = ""


class QueryResponse(BaseModel):
    question: str
    strategy: Strategy
    answer: str
    sources: list[SourceDocument]
    latency_ms: float


class BenchmarkRequest(BaseModel):
    strategies: list[Strategy] = Field(
        default=["naive", "hybrid", "reranker", "graph"],
        description="Strategies to compare",
    )
    sample_n: int = Field(0, ge=0, description="Gold queries to use (0 = all)")
    recall_k: int = Field(5, ge=1)
    ndcg_k: int = Field(10, ge=1)
    use_ragas: bool = Field(
        False,
        description="Generate LLM answers and run RAGAS metrics (faithfulness, relevancy, context precision)",
    )


class BenchmarkResponse(BaseModel):
    results: dict[str, dict[str, Any]]
    csv_path: str


class HealthResponse(BaseModel):
    status: str = "ok"
    indexes_ready: bool
