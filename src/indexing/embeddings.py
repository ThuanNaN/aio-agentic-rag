from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
from configs import config, Config, EmbeddingsConfig


class VLLMEmbeddings(Embeddings):
    """OpenAI-compatible /v1/embeddings client for a vLLM embedding server."""

    def __init__(self, config: Config):
        self._config: EmbeddingsConfig = config.embeddings
        self._client = OpenAI(base_url=config.embeddings.base_url, api_key=config.embeddings.api_key)
        self._model = config.embeddings.model
        self._batch_size = config.embeddings.batch_size
        self._workers = config.embeddings.embed_workers

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            input=batch,
            extra_body={"truncate_prompt_tokens": self._config.max_length},
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Serial embedding — used for single queries at inference time."""
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            results.extend(self._embed_batch(texts[i : i + self._batch_size]))
        return results

    def embed_parallel(self, texts: list[str]) -> list[list[float]]:
        """Parallel embedding — used during ingestion for large text collections."""
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        ordered: list[list[list[float]] | None] = [None] * len(batches)

        def _task(idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            return idx, self._embed_batch(batch)

        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            futures = {executor.submit(_task, i, b): i for i, b in enumerate(batches)}
            with tqdm(total=len(texts), desc="Embedding", unit="text") as bar:
                for future in as_completed(futures):
                    idx, embs = future.result()
                    ordered[idx] = embs
                    bar.update(len(embs))

        return [emb for batch_embs in ordered for emb in batch_embs]  # type: ignore[union-attr]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_embeddings() -> VLLMEmbeddings:
    """Return a VLLMEmbeddings instance configured from environment variables."""
    base_url = config.embeddings.base_url
    if not base_url:
        raise ValueError(
            "EMBEDDING_BASE_URL must be set (e.g. http://localhost:8080/v1). "
            "Start the server: vllm serve google/embeddinggemma-300m --port 8080"
        )
    return VLLMEmbeddings(config)
