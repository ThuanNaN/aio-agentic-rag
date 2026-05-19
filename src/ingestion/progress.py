"""Thread-safe pipeline progress state shared between the ingest runner and the API."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Literal

StepStatus = Literal["pending", "running", "done", "skipped", "error"]

STEPS = ["load", "clean", "chunk", "chroma", "bm25", "graph"]


@dataclass
class StepState:
    status: StepStatus = "pending"
    count: int = 0
    total: int = 0
    message: str = ""


@dataclass
class _PipelineState:
    status: Literal["idle", "running", "done", "error"] = "idle"
    steps: dict[str, StepState] = field(
        default_factory=lambda: {s: StepState() for s in STEPS}
    )
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


_lock = threading.Lock()
_state = _PipelineState()


def get_state() -> dict:
    with _lock:
        elapsed = (
            round((_state.finished_at or time.time()) - _state.started_at, 1)
            if _state.started_at
            else 0.0
        )
        return {
            "status": _state.status,
            "steps": {
                name: {
                    "status": s.status,
                    "count": s.count,
                    "total": s.total,
                    "message": s.message,
                }
                for name, s in _state.steps.items()
            },
            "error": _state.error,
            "elapsed_s": elapsed,
        }


def reset() -> None:
    global _state
    with _lock:
        _state = _PipelineState(status="running", started_at=time.time())


def step_start(name: str, total: int = 0) -> None:
    with _lock:
        _state.steps[name].status = "running"
        _state.steps[name].total = total
        _state.steps[name].count = 0
        _state.steps[name].message = ""


def step_update(name: str, count: int, total: int | None = None) -> None:
    with _lock:
        _state.steps[name].count = count
        if total is not None:
            _state.steps[name].total = total


def step_done(name: str, count: int = 0, message: str = "") -> None:
    with _lock:
        _state.steps[name].status = "done"
        _state.steps[name].count = count
        _state.steps[name].total = count
        _state.steps[name].message = message


def step_skip(name: str, message: str = "") -> None:
    with _lock:
        _state.steps[name].status = "skipped"
        _state.steps[name].message = message


def step_error(name: str, message: str = "") -> None:
    with _lock:
        _state.steps[name].status = "error"
        _state.steps[name].message = message
        _state.status = "error"
        _state.error = message
        _state.finished_at = time.time()


def pipeline_done() -> None:
    with _lock:
        _state.status = "done"
        _state.finished_at = time.time()


def pipeline_error(message: str) -> None:
    with _lock:
        _state.status = "error"
        _state.error = message
        _state.finished_at = time.time()
