from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class DatasetConfig:
    name: str
    sample_size: int | None
    config_metadata: str
    config_content: str
    config_relationships: str


@dataclass
class PathsConfig:
    data_raw: str
    data_processed: str
    data_eval: str
    bm25_index: str
    graph_index: str
    results: str


@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    separators: list[str]
    clean_workers: int


@dataclass
class EmbeddingsConfig:
    model: str
    base_url: str
    api_key: str
    batch_size: int
    embed_workers: int
    max_length: int


@dataclass
class ChromaConfig:
    collection_name: str
    distance_metric: str
    host: str
    port: int
    upsert_bs: int


@dataclass
class RetrievalConfig:
    k: int
    bm25_k: int
    dense_k: int
    rrf_k: int
    reranker_model: str
    graph_max_hops: int
    graph_edge_types: list[str]


@dataclass
class GenerationConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    openai_api_key: str | None
    openai_base_url: str | None
    openrouter_api_key: str | None
    vllm_base_url: str | None
    vllm_api_key: str
    google_api_key: str | None


@dataclass
class EvaluationConfig:
    gold_set_path: str
    recall_k: int
    ndcg_k: int
    query_types: list[str]


@dataclass
class Config:
    dataset: DatasetConfig
    paths: PathsConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    chroma: ChromaConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    hf_token: str | None


def _load_yaml() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_config(raw: dict) -> Config:
    ds = raw["dataset"]
    dataset = DatasetConfig(
        name=ds["name"],
        sample_size=ds["sample_size"],
        config_metadata=ds["configs"]["metadata"],
        config_content=ds["configs"]["content"],
        config_relationships=ds["configs"]["relationships"],
    )

    p = raw["paths"]
    paths = PathsConfig(
        data_raw=p["data_raw"],
        data_processed=p["data_processed"],
        data_eval=p["data_eval"],
        bm25_index=p["bm25_index"],
        graph_index=p["graph_index"],
        results=p["results"],
    )

    ck = raw["chunking"]
    chunking = ChunkingConfig(
        chunk_size=ck["chunk_size"],
        chunk_overlap=ck["chunk_overlap"],
        separators=ck["separators"],
        clean_workers=int(os.getenv("CLEAN_WORKERS") or ck.get("clean_workers", 1)),
    )

    em = raw["embeddings"]
    embeddings = EmbeddingsConfig(
        model=os.getenv("EMBEDDING_MODEL") or em["model"],
        base_url=os.getenv("EMBEDDING_BASE_URL") or em["base_url"],
        api_key=os.getenv("EMBEDDING_API_KEY") or em["api_key"],
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE") or em["batch_size"]),
        embed_workers=int(os.getenv("EMBEDDING_WORKERS") or em["embed_workers"]),
        max_length=int(os.getenv("EMBEDDING_MAX_LENGTH") or em.get("max_length", 2048)),
    )

    ch = raw["chroma"]
    chroma = ChromaConfig(
        collection_name=ch["collection_name"],
        distance_metric=ch["distance_metric"],
        host=os.getenv("CHROMA_HOST") or ch["host"],
        port=int(os.getenv("CHROMA_PORT") or ch["port"]),
        upsert_bs=int(os.getenv("CHROMA_UBS") or ch["upsert_bs"]),
    )

    rt = raw["retrieval"]
    retrieval = RetrievalConfig(
        k=rt["k"],
        bm25_k=rt["bm25_k"],
        dense_k=rt["dense_k"],
        rrf_k=rt["rrf_k"],
        reranker_model=rt["reranker_model"],
        graph_max_hops=rt["graph_max_hops"],
        graph_edge_types=rt["graph_edge_types"],
    )

    gen = raw["generation"]
    generation = GenerationConfig(
        provider=os.getenv("LLM_PROVIDER") or gen["provider"],
        model=os.getenv("LLM_MODEL") or gen["model"],
        temperature=float(os.getenv("LLM_TEMPERATURE") or gen["temperature"]),
        max_tokens=gen["max_tokens"],
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        vllm_base_url=os.getenv("VLLM_BASE_URL"),
        vllm_api_key=os.getenv("VLLM_API_KEY", "empty"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    ev = raw["evaluation"]
    evaluation = EvaluationConfig(
        gold_set_path=ev["gold_set_path"],
        recall_k=ev["recall_k"],
        ndcg_k=ev["ndcg_k"],
        query_types=ev["query_types"],
    )

    return Config(
        dataset=dataset,
        paths=paths,
        chunking=chunking,
        embeddings=embeddings,
        chroma=chroma,
        retrieval=retrieval,
        generation=generation,
        evaluation=evaluation,
        hf_token=os.getenv("HF_TOKEN"),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    load_dotenv()
    return _build_config(_load_yaml())

config: Config = get_config()
