"""HTTP client wrapping the FastAPI backend endpoints."""
from __future__ import annotations

import os
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def health() -> dict:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return {"status": "error", "indexes_ready": False, "detail": "API unreachable"}
    except Exception as e:
        return {"status": "error", "indexes_ready": False, "detail": str(e)}


def query(question: str, strategy: str, k: int) -> dict:
    try:
        r = requests.post(
            f"{BASE_URL}/query",
            json={"question": question, "strategy": strategy, "k": k},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        return {"error": detail}
    except requests.ConnectionError:
        return {"error": "Cannot reach API. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}


def run_benchmark(
    strategies: list[str],
    sample_n: int,
    recall_k: int,
    ndcg_k: int,
) -> dict:
    try:
        r = requests.post(
            f"{BASE_URL}/benchmark",
            json={
                "strategies": strategies,
                "sample_n": sample_n,
                "recall_k": recall_k,
                "ndcg_k": ndcg_k,
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        return {"error": detail}
    except requests.ConnectionError:
        return {"error": "Cannot reach API. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}
