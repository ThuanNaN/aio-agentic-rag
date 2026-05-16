"""Gradio UI for the AIO Agentic RAG system."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

from api_client import health, query as api_query, run_benchmark as api_benchmark

STRATEGIES = ["naive", "hybrid", "reranker", "graph", "agentic"]
_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
STRATEGY_COLORS = dict(zip(STRATEGIES, _COLORS))


# ── callbacks ────────────────────────────────────────────────────────────────

def check_health() -> str:
    h = health()
    if h.get("indexes_ready"):
        return "✅ Indexes ready — all retrieval strategies available"
    if h.get("status") == "error":
        return f"❌ API unreachable — {h.get('detail', 'start FastAPI with: uvicorn src.api.app:app --port 8000')}"
    return "⚠️ Indexes not ready — run: python scripts/ingest.py"


def do_query(question: str, strategy: str, k: int):
    if not question.strip():
        return "Please enter a question.", None, ""

    result = api_query(question, strategy, int(k))

    if "error" in result:
        return f"Error: {result['error']}", None, ""

    answer = result["answer"]
    info = (
        f"{result['latency_ms']:.0f} ms  ·  strategy: {result['strategy']}"
        f"  ·  {len(result['sources'])} sources retrieved"
    )

    rows = []
    for s in result["sources"]:
        title = s["title"]
        rows.append({
            "doc_id": s["doc_id"],
            "title": title[:65] + "…" if len(title) > 65 else title,
            "authority": s["authority"],
            "issued": s["issue_date"],
            "excerpt": s["excerpt"],
        })
    sources_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    return answer, sources_df, info


def do_benchmark(strategies: list[str], sample_n: int, recall_k: int, ndcg_k: int):
    if not strategies:
        return "Select at least one strategy.", None, None

    result = api_benchmark(strategies, int(sample_n), int(recall_k), int(ndcg_k))

    if "error" in result:
        return f"Error: {result['error']}", None, None

    results: dict = result["results"]

    # Results table
    rows = []
    for strat, metrics in results.items():
        rows.append({"strategy": strat, **{k: round(v, 4) for k, v in metrics.items()}})
    df = pd.DataFrame(rows)

    # Bar chart
    recall_key = f"recall@{recall_k}"
    ndcg_key = f"ndcg@{ndcg_k}"
    metrics_to_plot = [m for m in [recall_key, ndcg_key, "avg_latency_ms"] if m in df.columns]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    strat_list = df["strategy"].tolist()
    colors = [STRATEGY_COLORS.get(s, "#888888") for s in strat_list]

    for ax, metric in zip(axes, metrics_to_plot):
        values = df[metric].tolist()
        bars = ax.bar(strat_list, values, color=colors)
        ax.set_title(metric, fontsize=11)
        ceiling = max(values) * 1.25 if max(values) > 0 else 1
        ax.set_ylim(0, ceiling)
        ax.tick_params(axis="x", rotation=20)
        offset = max(values) * 0.02 if max(values) > 0 else 0.01
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(
        "RAG Strategy Comparison — Vietnamese Legal Documents",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    csv_note = f"Results saved → {result.get('csv_path', 'results/')}"
    return csv_note, df, fig


# ── layout ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="AIO Agentic RAG", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# AIO Agentic RAG\n### Vietnamese Legal Document QA · 5 Retrieval Strategies")

    status_box = gr.Textbox(label="System Status", interactive=False, max_lines=1)

    with gr.Tab("Query"):
        gr.Markdown("Ask a legal question and get an answer with cited sources.")

        question_in = gr.Textbox(
            label="Question (Vietnamese)",
            placeholder="Luật Đất đai 2024 có hiệu lực từ ngày nào?",
            lines=2,
        )
        with gr.Row():
            strategy_in = gr.Dropdown(
                choices=STRATEGIES,
                value="hybrid",
                label="Strategy",
                scale=2,
            )
            k_in = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="k (docs to retrieve)", scale=3)

        ask_btn = gr.Button("Ask", variant="primary")

        info_lbl = gr.Textbox(label="Latency & info", interactive=False, max_lines=1)
        answer_box = gr.Textbox(label="Answer", lines=10, interactive=False)
        sources_table = gr.DataFrame(label="Sources", interactive=False, wrap=True)

        gr.Examples(
            examples=[
                ["Luật Đất đai 2024 có hiệu lực từ ngày nào?"],
                ["Điều kiện để được cấp Giấy chứng nhận quyền sử dụng đất là gì?"],
                ["Luật Doanh nghiệp 2020 quy định những loại hình doanh nghiệp nào?"],
                ["Thủ tục thành lập công ty trách nhiệm hữu hạn gồm những bước nào?"],
                ["Bộ luật Lao động 2019 quy định thời gian làm việc tối đa trong một tuần là bao nhiêu giờ?"],
                ["Luật Hôn nhân và Gia đình quy định độ tuổi kết hôn tối thiểu là bao nhiêu?"],
                ["Luật nào đã sửa đổi, bổ sung Luật Thuế thu nhập doanh nghiệp?"],
                ["Các trường hợp miễn thuế thu nhập cá nhân theo quy định hiện hành?"],
            ],
            inputs=question_in,
            label="Example Questions",
        )

        ask_btn.click(
            fn=do_query,
            inputs=[question_in, strategy_in, k_in],
            outputs=[answer_box, sources_table, info_lbl],
        )
        question_in.submit(
            fn=do_query,
            inputs=[question_in, strategy_in, k_in],
            outputs=[answer_box, sources_table, info_lbl],
        )

    with gr.Tab("Benchmark"):
        gr.Markdown("Compare all strategies on the gold evaluation set.")

        strategies_chk = gr.CheckboxGroup(
            choices=STRATEGIES,
            value=["naive", "hybrid", "reranker", "graph", "agentic"],
            label="Strategies to run",
        )
        with gr.Row():
            sample_n_in = gr.Number(value=8, label="Sample N  (0 = all gold queries)", precision=0, minimum=0)
            recall_k_in = gr.Number(value=5, label="recall @ k", precision=0, minimum=1)
            ndcg_k_in = gr.Number(value=10, label="nDCG @ k", precision=0, minimum=1)

        run_btn = gr.Button("Run Benchmark", variant="primary")

        bench_status = gr.Textbox(label="Status", interactive=False, max_lines=1)
        bench_table = gr.DataFrame(label="Results", interactive=False)
        bench_chart = gr.Plot(label="Comparison Chart")

        run_btn.click(
            fn=do_benchmark,
            inputs=[strategies_chk, sample_n_in, recall_k_in, ndcg_k_in],
            outputs=[bench_status, bench_table, bench_chart],
        )

    demo.load(fn=check_health, outputs=status_box)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
