"""CLI: download → clean → chunk → index all artifacts.

Resume support
--------------
Pass --resume to skip any step whose output file already exists.
For the Chroma upsert step, a chroma_progress.json sidecar is written after
each segment so the run can continue from the last completed segment boundary
without re-embedding already-indexed chunks.

Usage
-----
  python scripts/ingest.py                      # full run
  python scripts/ingest.py --sample 1000        # quick smoke test
  python scripts/ingest.py --resume             # continue interrupted run
  python scripts/ingest.py --from-step chunk    # force-restart from chunk step
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from configs import config

STEPS = ["load", "clean", "chunk", "chroma", "bm25", "graph"]

_PROCESSED = Path(config.paths.data_processed)


def _skip(label: str, path: Path) -> bool:
    print(f"[skip] {label} — {path} already exists")
    return True


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def step_load(sample_size: int | None, resume: bool) -> None:
    raw_path = _PROCESSED / "raw_docs.json"
    rels_path = _PROCESSED / "relationships.json"
    if resume and raw_path.exists() and rels_path.exists():
        _skip("load", raw_path)
        return

    from src.ingestion.loader import load_documents, load_relationships

    docs = load_documents(config, sample_size=sample_size)
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            ensure_ascii=False, indent=2,
        )
    )
    print(f"[load] {len(docs)} docs → {raw_path}")

    doc_ids = {d.metadata.get("doc_id", "") for d in docs} if sample_size else None
    rels = load_relationships(config, doc_ids=doc_ids)
    rels_path.write_text(json.dumps(rels, ensure_ascii=False, indent=2))
    print(f"[load] {len(rels)} relationships → {rels_path}")


def step_clean(resume: bool) -> None:
    out_path = _PROCESSED / "cleaned_docs.json"
    if resume and out_path.exists():
        _skip("clean", out_path)
        return

    from langchain_core.documents import Document
    from src.ingestion.cleaner import clean_documents

    data = json.loads((_PROCESSED / "raw_docs.json").read_text())
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    cleaned = clean_documents(docs, workers=config.chunking.clean_workers)
    out_path.write_text(
        json.dumps(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in cleaned],
            ensure_ascii=False, indent=2,
        )
    )
    print(f"[clean] {len(cleaned)} docs → {out_path}")


def step_chunk(resume: bool) -> None:
    out_path = _PROCESSED / "chunks.json"
    if resume and out_path.exists():
        _skip("chunk", out_path)
        return

    from langchain_core.documents import Document
    from src.ingestion.chunker import chunk_documents

    data = json.loads((_PROCESSED / "cleaned_docs.json").read_text())
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    chunks = chunk_documents(docs, config)
    out_path.write_text(
        json.dumps(
            [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
            ensure_ascii=False, indent=2,
        )
    )
    print(f"[chunk] {len(chunks)} chunks → {out_path}")


def step_chroma(resume: bool, segment_size: int) -> None:
    progress_path = _PROCESSED / "chroma_progress.json"

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from src.indexing.chroma_store import _make_client, upsert_documents
    from src.indexing.embeddings import get_embeddings

    data = json.loads((_PROCESSED / "chunks.json").read_text())
    chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    total = len(chunks)

    start_offset = 0
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        start_offset = progress.get("last_offset", 0)
        if start_offset >= total:
            print(f"[skip] chroma — all {total} chunks already indexed")
            return
        print(f"[resume] chroma — continuing from chunk {start_offset}/{total}")
    else:
        progress_path.write_text(json.dumps({"last_offset": 0, "total": total}))

    client = _make_client(config.chroma.host, config.chroma.port)
    emb_fn = get_embeddings()
    store = Chroma(
        client=client,
        collection_name=config.chroma.collection_name,
        embedding_function=emb_fn,
    )
    n = upsert_documents(
        store, chunks, emb_fn=emb_fn,
        start_offset=start_offset,
        progress_path=str(progress_path),
        segment_size=segment_size,
    )
    print(f"[chroma] indexed {n} chunks (total {total}) → {config.chroma.host}:{config.chroma.port}")


def step_bm25(resume: bool) -> None:
    out_path = Path(config.paths.bm25_index)
    if resume and out_path.exists():
        _skip("bm25", out_path)
        return

    from langchain_core.documents import Document
    from src.indexing.bm25_index import build_bm25_index

    data = json.loads((_PROCESSED / "chunks.json").read_text())
    chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    build_bm25_index(chunks, save_path=str(out_path))
    print(f"[bm25] {len(chunks)} chunks → {out_path}")


def step_graph(resume: bool) -> None:
    out_path = Path(config.paths.graph_index)
    if resume and out_path.exists():
        _skip("graph", out_path)
        return

    from src.retrieval.graph import build_graph

    rels = json.loads((_PROCESSED / "relationships.json").read_text())
    G = build_graph(rels, save_path=str(out_path))
    print(f"[graph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest Vietnamese legal documents (with resume support)"
    )
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of docs to load (0 = full dataset)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip steps whose output files already exist")
    parser.add_argument("--from-step", choices=STEPS, default=None,
                        dest="from_step",
                        help="Force-restart pipeline from this step (implies --resume for earlier steps)")
    parser.add_argument("--segment-size", type=int, default=10000,
                        dest="segment_size",
                        help="Chunks per Chroma upsert segment (default: 10000)")
    args = parser.parse_args()

    # --from-step implies resume for everything before the chosen step,
    # and forces re-run of the chosen step and everything after.
    force_from = STEPS.index(args.from_step) if args.from_step else len(STEPS)

    def should_resume(step_name: str) -> bool:
        if args.from_step is None:
            return args.resume
        idx = STEPS.index(step_name)
        if idx < force_from:
            return True   # earlier step: always resume (skip if done)
        if idx == force_from:
            return False  # target step: always re-run
        return args.resume  # later steps: respect --resume flag

    sample_size = args.sample if args.sample > 0 else None

    step_load(sample_size, resume=should_resume("load"))
    step_clean(resume=should_resume("clean"))
    step_chunk(resume=should_resume("chunk"))
    step_chroma(resume=should_resume("chroma"), segment_size=args.segment_size)
    step_bm25(resume=should_resume("bm25"))
    step_graph(resume=should_resume("graph"))

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
