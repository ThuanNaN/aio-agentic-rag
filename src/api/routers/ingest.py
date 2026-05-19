"""API endpoints for running and monitoring the ingestion pipeline."""
from __future__ import annotations

import threading
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_task_lock = threading.Lock()
_running = False


class IngestRequest(BaseModel):
    sample: int = Field(0, ge=0, description="Docs to load (0 = full dataset)")
    resume: bool = Field(False, description="Skip steps whose outputs already exist")
    from_step: Literal["load", "clean", "chunk", "chroma", "bm25", "graph"] | None = Field(
        None, description="Force-restart from this step"
    )
    segment_size: int = Field(10000, ge=100, description="Chunks per Chroma upsert segment")


def _run(sample: int, resume: bool, from_step: str | None, segment_size: int) -> None:
    global _running
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from scripts.ingest import run_pipeline
        run_pipeline(
            sample_size=sample if sample > 0 else None,
            resume=resume,
            from_step=from_step,
            segment_size=segment_size,
        )
    finally:
        with _task_lock:
            _running = False


@router.post("/start", summary="Start ingestion pipeline")
async def start_ingest(req: IngestRequest):
    global _running
    with _task_lock:
        if _running:
            raise HTTPException(status_code=409, detail="Ingestion already running")
        _running = True

    t = threading.Thread(
        target=_run,
        args=(req.sample, req.resume, req.from_step, req.segment_size),
        daemon=True,
    )
    t.start()
    return {"started": True, "sample": req.sample, "resume": req.resume}


@router.get("/status", summary="Get ingestion pipeline status")
async def ingest_status():
    from src.ingestion.progress import get_state
    state = get_state()
    state["is_running"] = _running
    return state
