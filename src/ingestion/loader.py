"""Loader module to read the dataset and return structured Documents and relationships."""
from __future__ import annotations
from pathlib import Path
from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import snapshot_download
from langchain_core.documents import Document
from tqdm import tqdm
from configs import Config

# Actual parquet columns use large_string — declare explicitly to skip the cast
_CONTENT_FEATURES = Features({
    "id": Value("large_string"),
    "content_html": Value("large_string"),
})


def _ensure_local_dataset(dataset_name: str, data_raw: str, token: str | None) -> str:
    """Download the full HF dataset repo to data/raw/ once; return the local path.

    Subsequent calls skip the download if parquet files are already present.
    """
    local = Path(data_raw) / dataset_name.replace("/", "--")
    if not any(local.rglob("*.parquet")):
        print(f"Downloading {dataset_name} → {local} (one-time, future runs load locally)")
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=str(local),
            token=token,
        )
    return str(local)


def _iter_content(source: str, sample_size: int | None) -> list[dict]:
    """Load content rows from local parquet, optionally capped at sample_size."""
    ds = load_dataset(source, "content", split="data", features=_CONTENT_FEATURES)
    total = sample_size if sample_size else len(ds)
    rows = []
    with tqdm(desc="Loading content", total=total, unit=" doc") as bar:
        for i, row in enumerate(ds):
            if sample_size and i >= sample_size:
                break
            rows.append(dict(row))
            bar.update(1)
    return rows


def _load_metadata_lookup(
    source: str, doc_ids: set[str] | None = None
) -> dict[str, dict]:
    """Build id → metadata dict from local parquet, optionally filtered to doc_ids."""
    ds: Dataset = load_dataset(source, "metadata", split="data")
    remaining = set(doc_ids) if doc_ids is not None else None
    result: dict[str, dict] = {}
    desc = f"Loading metadata ({len(doc_ids)} docs)" if doc_ids else "Loading metadata"
    for row in tqdm(ds, desc=desc, unit="doc"):
        row_id = str(row["id"])
        if remaining is None:
            result[row_id] = dict(row)
        elif row_id in remaining:
            result[row_id] = dict(row)
            remaining.discard(row_id)
            if not remaining:
                break
    return result


def _load_relationships(
    source: str, doc_ids: set[str] | None = None
) -> list[dict]:
    """Return relationship rows from local parquet, filtered to doc_ids if provided."""
    ds: Dataset = load_dataset(source, "relationships", split="data")
    return [
        dict(row)
        for row in tqdm(ds, desc="Loading relationships", unit="rel")
        if doc_ids is None
        or str(row.get("doc_id", "")) in doc_ids
        or str(row.get("other_doc_id", "")) in doc_ids
    ]


def load_documents(config: Config, sample_size: int | None = None) -> list[Document]:
    """
    Load Documents from the locally cached dataset.

    On first call, downloads the full HF dataset to data/raw/.
    Subsequent calls load from local parquet files (fast).
    Streams content first to collect doc IDs, then loads only matching metadata rows.
    """
    dataset_cfg = config.dataset
    effective_sample = sample_size if sample_size is not None else dataset_cfg.sample_size
    source = _ensure_local_dataset(dataset_cfg.name, config.paths.data_raw, config.hf_token)

    content_rows = _iter_content(source, effective_sample)
    doc_ids = {str(row.get("id", "")) for row in content_rows} if effective_sample else None
    meta_lookup = _load_metadata_lookup(source, doc_ids=doc_ids)

    docs: list[Document] = []
    for row in content_rows:
        doc_id = str(row.get("id", ""))
        meta = meta_lookup.get(doc_id, {})
        docs.append(
            Document(
                page_content=row.get("content_html", ""),
                metadata={
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "doc_type": meta.get("loai_van_ban", ""),
                    "authority": meta.get("co_quan_ban_hanh", ""),
                    "issue_date": meta.get("ngay_ban_hanh", ""),
                    "effective_date": meta.get("ngay_co_hieu_luc", ""),
                    "expiry_date": meta.get("ngay_het_hieu_luc", ""),
                    "sector": meta.get("linh_vuc", ""),
                    "status": meta.get("tinh_trang_hieu_luc", ""),
                },
            )
        )
    return docs


def load_relationships(config: Config, doc_ids: set[str] | None = None) -> list[dict]:
    """Return cross-document relationship rows from local cache, filtered to doc_ids if provided."""
    source = _ensure_local_dataset(config.dataset.name, config.paths.data_raw, config.hf_token)
    return _load_relationships(source, doc_ids=doc_ids)
