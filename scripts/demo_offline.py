"""离线演示 — 零模型 / 零 API Key / 零外部服务也能看到问答形态

用途
----
让访客克隆仓库后，不用下载 8GB 本地 BGE 模型、不填任何 API Key、不起 MySQL/Redis，
就能把「知识库问答」这条最短链路跑起来。

用法
----
    python scripts/demo_offline.py                       # 有 gradio 起网页(7863)，否则进终端问答
    python scripts/demo_offline.py --terminal            # 强制终端问答
    python scripts/demo_offline.py --ask "怎么退货？"      # 单问单答（便于脚本化验证）
    python scripts/demo_offline.py --top-k 5 --min-score 0.3

说明
----
本链路只做「字符二元组 IDF 加权的关键词匹配 + FAQ 原文直出」，不加载模型、不调用 LLM、
不做幻觉核验，**输出不代表系统真实回答质量**。完整链路（双路召回 / 六级降级 / 幻觉自纠正 /
转人工）见 README「快速开始」。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 允许直接 python scripts/demo_offline.py 运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.demo.offline_retrieval import (  # noqa: E402
    DEFAULT_FAQ_PATH,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    DemoAnswer,
    OfflineFAQIndex,
)

DEMO_PORT = 7863

SAMPLE_QUESTIONS = (
    "怎么申请退货？",
    "退款要等多久才能到账？",
    "快递显示签收但我没收到包裹",
    "客服态度恶劣",
    "怎么联系人工？",
)


def _candidates_md(answer: DemoAnswer) -> str:
    """把候选条目渲染成 markdown 表格（命中/未命中都展示，便于观察召回）"""
    if not answer.hits:
        return ""
    lines = ["| # | 候选 FAQ | 匹配度 |", "|---|----------|--------|"]
    lines.extend(
        f"| {i} | {hit.question} | {hit.score:.2f} |"
        for i, hit in enumerate(answer.hits, 1)
    )
    return "\n".join(lines)


def _render_terminal(answer: DemoAnswer) -> str:
    """终端渲染：把 markdown 里的强调符号去掉，保证可读"""
    text = answer.text.replace("**", "").replace("_", "")
    # 未命中时正文里已经列了候选条目，避免重复输出
    if answer.matched and answer.hits:
        text += "\n候选：" + "；".join(
            f"{h.question}({h.score:.2f})" for h in answer.hits
        )
    return text


def run_terminal(index: OfflineFAQIndex, top_k: int) -> int:
    """终端交互问答，Ctrl+C / 输入 exit 退出"""
    print("=" * 66)
    print("  电商智能客服 · 离线演示（无模型 / 无 API Key / 无外部服务）")
    print("=" * 66)
    print(f"  知识库: {index.faq_path} | 阈值: {index.min_score} | Top-K: {top_k}")
    print("  示例问题：" + " / ".join(SAMPLE_QUESTIONS[:3]))
    print("  输入 exit 或按 Ctrl+C 退出")
    print("-" * 66)

    while True:
        try:
            query = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if query.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            return 0
        if not query:
            continue

        answer = index.answer(query, top_k=top_k)
        print("\n客服:\n" + _render_terminal(answer))


def build_ui(index: OfflineFAQIndex, top_k: int):
    """构建 Gradio 界面（只在需要网页时导入 gradio，终端模式无需安装）"""
    import gradio as gr

    def respond(query: str):
        if not query or not query.strip():
            return "请先输入问题。", ""
        answer = index.answer(query, top_k=top_k)
        return answer.text, _candidates_md(answer)

    with gr.Blocks(title="电商智能客服 · 离线演示") as demo:
        gr.Markdown(
            "# 电商智能客服 · 离线演示\n"
            "本页只跑 **离线 FAQ 关键词检索**：不加载 BGE 模型、不调用 LLM、不连数据库。\n"
            "完整链路（双路召回 / L0 至 L5 六级降级 / claims 级幻觉核验 / 转人工）见 README。\n"
            "示例：" + "、".join(f"`{q}`" for q in SAMPLE_QUESTIONS)
        )
        with gr.Row():
            query_box = gr.Textbox(
                label="你的问题", placeholder="例如：怎么退货？", scale=4, lines=1
            )
            submit_btn = gr.Button("提问", variant="primary", scale=1)
        answer_md = gr.Markdown(label="回答")
        candidates_md = gr.Markdown(label="候选条目")
        submit_btn.click(
            respond, inputs=[query_box], outputs=[answer_md, candidates_md]
        )
        query_box.submit(
            respond, inputs=[query_box], outputs=[answer_md, candidates_md]
        )
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="电商智能客服 — 零依赖离线演示")
    parser.add_argument("--faq", default=str(DEFAULT_FAQ_PATH), help="FAQ 知识库路径")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="候选条数")
    parser.add_argument(
        "--min-score", type=float, default=DEFAULT_MIN_SCORE, help="命中匹配度阈值"
    )
    parser.add_argument("--ask", default="", help="单问单答后退出（便于脚本化验证）")
    parser.add_argument("--terminal", action="store_true", help="强制使用终端问答")
    parser.add_argument("--port", type=int, default=DEMO_PORT, help="网页端口")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        index = OfflineFAQIndex(faq_path=args.faq, min_score=args.min_score)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    if args.ask:
        answer = index.answer(args.ask, top_k=args.top_k)
        print(_render_terminal(answer))
        return 0

    if args.terminal:
        return run_terminal(index, args.top_k)

    try:
        import gradio  # noqa: F401
    except ImportError:
        print(
            "[提示] 未安装 gradio，自动降级为终端问答（pip install gradio 可启用网页）\n"
        )
        return run_terminal(index, args.top_k)

    demo = build_ui(index, args.top_k)
    print(f"离线演示已启动：http://127.0.0.1:{args.port}（Ctrl+C 停止）")
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
