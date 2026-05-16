"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from configs import config
from src.api.routers import benchmark, query
from src.api.schemas import HealthResponse


def _chroma_ready() -> bool:
    import chromadb
    if not config.chroma.host:
        return False
    try:
        client = chromadb.HttpClient(host=config.chroma.host, port=config.chroma.port)
        client.heartbeat()
        return True
    except Exception:
        return False


def _indexes_ready() -> bool:
    return (
        _chroma_ready()
        and Path(config.paths.bm25_index).exists()
        and Path(config.paths.graph_index).exists()
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up cached index handles on startup
    if _indexes_ready():
        from src.tools.retrieval_tools import _get_bm25, _get_graph, _get_store
        _get_store()
        _get_bm25()
        _get_graph()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AIO Agentic RAG — Vietnamese Legal Benchmark",
        description="Benchmarking 5 RAG strategies on Vietnamese legal documents",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(query.router)
    app.include_router(benchmark.router)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", indexes_ready=_indexes_ready())

    return app


app = create_app()
