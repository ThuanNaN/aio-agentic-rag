"""CLI: create or validate the gold QA set."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from configs import config
from src.evaluation.gold_set import SAMPLE_GOLD_SET, save_gold_set


def main():
    out_path = config.evaluation.gold_set_path
    if Path(out_path).exists():
        print(f"Gold set already exists at {out_path}")
        data = json.loads(Path(out_path).read_text())
        print(f"Entries: {len(data)}")
        from collections import Counter
        types = Counter(e["query_type"] for e in data)
        for t, c in sorted(types.items()):
            print(f"  {t}: {c}")
    else:
        save_gold_set(SAMPLE_GOLD_SET, out_path)
        print(f"Saved {len(SAMPLE_GOLD_SET)} sample queries to {out_path}")
        print(f"Edit {out_path} to add expected_doc_ids and answer_ground_truth.")


if __name__ == "__main__":
    main()
