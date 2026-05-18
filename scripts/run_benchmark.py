"""CLI: run strategy benchmarks against the gold QA set."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run RAG strategy benchmarks")
    parser.add_argument(
        "--strategy",
        choices=["naive", "hybrid", "reranker", "graph", "agentic", "all"],
        default="all",
        help="Which strategy to benchmark",
    )
    parser.add_argument("--sample", type=int, default=0, help="Number of gold queries (0 = all)")
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Generate LLM answers and run RAGAS metrics (faithfulness, relevancy, context precision)",
    )
    args = parser.parse_args()

    from src.evaluation.benchmark import run_benchmark, STRATEGIES

    strategies = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]
    n = args.sample if args.sample > 0 else None

    run_benchmark(
        strategies=strategies,
        recall_k=args.recall_k,
        ndcg_k=args.ndcg_k,
        sample_n=n,
        use_ragas=args.ragas,
    )


if __name__ == "__main__":
    main()
