"""POST /benchmark — run strategies against the gold set via the orchestrator agent."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from src.api.schemas import BenchmarkRequest, BenchmarkResponse

router = APIRouter()


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    try:
        from src.evaluation.benchmark import run_benchmark as _run

        n = request.sample_n if request.sample_n > 0 else None
        results = _run(
            strategies=list(request.strategies),
            recall_k=request.recall_k,
            ndcg_k=request.ndcg_k,
            sample_n=n,
            use_ragas=request.use_ragas,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {e}")

    from datetime import datetime
    from pathlib import Path

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = str(Path("results") / f"benchmark_{ts}.csv")

    return BenchmarkResponse(results=results, csv_path=csv_path)
