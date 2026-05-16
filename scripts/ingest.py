"""CLI: download → clean → chunk → index all artifacts."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()
from src.agents.orchestrator import agent


def main():
    parser = argparse.ArgumentParser(description="Ingest Vietnamese legal documents dataset")
    parser.add_argument("--sample", type=int, default=0, help="Number of docs to ingest (0 = full)")
    parser.add_argument("--thread-id", default="ingest-01", help="Thread ID for the agent session")
    args = parser.parse_args()

    if args.sample > 0:
        instruction = (
            f"Ingest a sample of {args.sample} documents from the Vietnamese legal documents dataset. "
            f"Call load_dataset_tool with sample_size={args.sample}. "
            "Then run all remaining steps: load relationships, clean docs, chunk docs, "
            "build Chroma index, build BM25 index, build relationship graph. "
            "Report final counts for each step."
        )
    else:
        instruction = (
            "Ingest the full Vietnamese legal documents dataset. "
            "Run all steps: load dataset, load relationships, clean docs, chunk docs, "
            "build Chroma index, build BM25 index, build relationship graph. "
            "Report final counts for each step."
        )

    config = {"configurable": {"thread_id": args.thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": instruction}]}, config=config)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
