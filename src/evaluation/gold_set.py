"""Gold QA set management — load, validate, and sample evaluation queries."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal
from configs import config

QueryType = Literal["factual", "multi_hop", "temporal", "reasoning"]

SAMPLE_GOLD_SET: list[dict] = [
    # Factual
    {
        "question": "Luật Đất đai số 31/2024/QH15 có hiệu lực từ ngày nào?",
        "expected_doc_ids": [],
        "answer_ground_truth": "Luật Đất đai số 31/2024/QH15 có hiệu lực thi hành từ ngày 01 tháng 01 năm 2025.",
        "query_type": "factual",
    },
    {
        "question": "Bộ Lao động - Thương binh và Xã hội có chức năng gì?",
        "expected_doc_ids": [],
        "answer_ground_truth": "Bộ Lao động - Thương binh và Xã hội là cơ quan của Chính phủ, thực hiện chức năng quản lý nhà nước về các lĩnh vực lao động, người có công và xã hội.",
        "query_type": "factual",
    },
    # Multi-hop
    {
        "question": "Nghị định nào sửa đổi Nghị định 43/2014/NĐ-CP về thi hành Luật Đất đai?",
        "expected_doc_ids": [],
        "answer_ground_truth": "Nghị định 01/2017/NĐ-CP sửa đổi, bổ sung một số nghị định quy định chi tiết thi hành Luật Đất đai, trong đó có Nghị định 43/2014/NĐ-CP.",
        "query_type": "multi_hop",
    },
    {
        "question": "Thông tư nào hướng dẫn thực hiện Nghị định 105/2022/NĐ-CP?",
        "expected_doc_ids": [],
        "answer_ground_truth": "",
        "query_type": "multi_hop",
    },
    # Temporal
    {
        "question": "Những quy định về bảo hiểm xã hội nào còn hiệu lực sau năm 2023?",
        "expected_doc_ids": [],
        "answer_ground_truth": "",
        "query_type": "temporal",
    },
    {
        "question": "Quy định về thuế giá trị gia tăng nào được ban hành sau năm 2020?",
        "expected_doc_ids": [],
        "answer_ground_truth": "",
        "query_type": "temporal",
    },
    # Reasoning
    {
        "question": "Doanh nghiệp nhỏ và vừa phải tuân theo quy định nào về thuế thu nhập doanh nghiệp?",
        "expected_doc_ids": [],
        "answer_ground_truth": "",
        "query_type": "reasoning",
    },
    {
        "question": "Người lao động làm việc theo hợp đồng thời vụ có quyền hưởng những chế độ bảo hiểm gì?",
        "expected_doc_ids": [],
        "answer_ground_truth": "",
        "query_type": "reasoning",
    },
]


def load_gold_set(path: str | None = None) -> list[dict]:
    """Load gold set from file, falling back to the built-in sample."""
    p = Path(path or config.evaluation.gold_set_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return SAMPLE_GOLD_SET


def save_gold_set(entries: list[dict], path: str | None = None) -> None:
    p = Path(path or config.evaluation.gold_set_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_by_type(entries: list[dict], query_type: QueryType) -> list[dict]:
    return [e for e in entries if e.get("query_type") == query_type]


def sample_gold_set(entries: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(entries, min(n, len(entries)))
