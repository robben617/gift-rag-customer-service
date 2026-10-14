"""单路 vs 双路召回 A/B 对照实验

补上 docs/ADR.md 里挂着的那个缺口：原文引用的是「行业实测召回率 +30%」，
本项目从未做过单路 vs 双路的对照跑分，也没有基线数字。

本脚本用同一份 49 题自建评测集、同一套参数（top_k / reranker / 相似度阈值），
只切换检索路径，输出两条召回与排序指标及相对提升幅度，结果落盘 JSON 便于引用。

为什么要同时看 MRR / Precision / nDCG:
    评测集是 49 题、知识库只有 60 条 FAQ，Recall@5 是二值命中口径，
    颗粒度很粗（差一题就是 2 个百分点），很可能掩盖排序上的变化。
    MRR 与 nDCG 对"相关文档排在第几位"敏感，能看出双路是否真的改善了排序。

用法:
    python scripts/eval_ab_dual_path.py              # 全量 49 题
    python scripts/eval_ab_dual_path.py --limit 10   # 冒烟
    python scripts/eval_ab_dual_path.py --no-rerank  # 关掉 CrossEncoder 重排做对照

前置条件:
    - 知识库已入库（data/chroma 有 collection）
    - EMBEDDING_MODEL_PATH / RERANKER_MODEL_PATH 指向本地 BGE 模型
    - DEEPSEEK_API_KEY 可用（双路需要真实改写问句；不可用时自动退化为原问句，
      此时两路等价，脚本会明确提示结果不可引用）
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# 允许直接从 scripts/ 目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 显式加载 .env：改写问句依赖 LLM，缺 Key 会静默退化为原问句导致两路等价
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # 未装 python-dotenv 时依赖 pydantic-settings 自身读取 .env
    pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

TOP_K = 5


def _load_cases(limit: int = 0) -> list[dict]:
    path = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    return cases[:limit] if limit else cases


def _rewrite(query: str) -> tuple[str, bool, bool]:
    """取生产链路里的改写问句

    返回 (改写问句, 是否实际改动, 是否发生异常)。
    区分"改写后与原句相同"与"改写调用报错"两种情形:
    后者会让两路等价，使 A/B 结论不成立，必须单独统计出来。
    """
    try:
        from src.routing import get_router

        rewritten = get_router().route(query).rewritten_query
        if rewritten and rewritten != query:
            return rewritten, True, False
        return rewritten or query, False, False
    except Exception as e:
        print(f"  ! 改写失败，退化为原问句: {e}")
        return query, False, True


def main() -> None:
    parser = argparse.ArgumentParser(description="单路 vs 双路召回 A/B 对照")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 CrossEncoder 重排")
    parser.add_argument("--save-dir", default="reports", help="结果保存目录")
    args = parser.parse_args()

    use_rerank = not args.no_rerank

    from src.embedding.retriever import get_retriever
    from src.evaluation.metrics import (
        calculate_mrr,
        calculate_ndcg,
        calculate_precision,
        calculate_recall,
    )

    def metrics(docs: list[str], ground_truth: str) -> dict:
        return {
            "recall@5": calculate_recall(docs, ground_truth, k=TOP_K),
            "precision@5": calculate_precision(docs, ground_truth, k=TOP_K),
            "mrr": calculate_mrr(docs, ground_truth),
            "ndcg@5": calculate_ndcg(docs, ground_truth, k=TOP_K),
        }

    cases = _load_cases(args.limit)
    if not cases:
        print("评估数据集为空")
        sys.exit(1)

    retriever = get_retriever()
    print(f"加载 {len(cases)} 条；top_k={TOP_K}；use_rerank={use_rerank}")

    metric_keys = ["recall@5", "precision@5", "mrr", "ndcg@5"]
    totals = {"single": {k: 0.0 for k in metric_keys}, "dual": {k: 0.0 for k in metric_keys}}
    rewrites_ok = 0
    rewrite_errors = 0
    per_case = []

    for i, case in enumerate(cases, 1):
        query = case["question"]
        ground_truth = case["ground_truth_answer"]

        rewritten, rewrote, rewrite_err = _rewrite(query)
        rewrites_ok += int(rewrote)
        rewrite_errors += int(rewrite_err)

        single = retriever.search(query=query, top_k=TOP_K, use_rerank=use_rerank)
        dual = retriever.search_dual_path(
            query=query,
            secondary_query=rewritten,
            top_k=TOP_K,
            use_rerank=use_rerank,
        )

        single_m = metrics([r.text for r in single.results], ground_truth)
        dual_m = metrics([r.text for r in dual.results], ground_truth)

        for k in metric_keys:
            totals["single"][k] += single_m[k]
            totals["dual"][k] += dual_m[k]

        per_case.append(
            {
                "question": query,
                "rewritten": rewritten,
                "rewritten_changed": rewrote,
                "rewrite_error": rewrite_err,
                "single": {k: round(single_m[k], 4) for k in metric_keys},
                "dual": {k: round(dual_m[k], 4) for k in metric_keys},
            }
        )
        print(
            f"[{i}/{len(cases)}] recall 单={single_m['recall@5']:.0f} 双={dual_m['recall@5']:.0f} "
            f"| mrr 单={single_m['mrr']:.2f} 双={dual_m['mrr']:.2f}  {query[:26]}"
        )

    total = len(cases)
    summary = {}
    for name in ("single", "dual"):
        summary[name] = {k: round(totals[name][k] / total, 4) for k in metric_keys}
    gains = {
        k: round(summary["dual"][k] - summary["single"][k], 4) for k in metric_keys
    }
    relative = {}
    for k in metric_keys:
        base = summary["single"][k]
        relative[k] = round(gains[k] / base, 4) if base > 0 else None

    comparable = rewrite_errors == 0

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "path": "data/eval_dataset.json",
            "total": total,
            "top_k": TOP_K,
            "use_rerank": use_rerank,
            "note": (
                "相关性判定为 Embedding 余弦相似度 ≥0.70（命中率口径），"
                "单路用 Retriever.search，双路用 Retriever.search_dual_path，"
                "其余参数完全一致"
            ),
        },
        "single_path": summary["single"],
        "dual_path": summary["dual"],
        "absolute_gain": gains,
        "relative_gain": relative,
        "rewrites_applied": rewrites_ok,
        "rewrite_errors": rewrite_errors,
        "comparable": comparable,
        "per_case": per_case,
    }

    print("\n" + "=" * 62)
    print("  单路 vs 双路召回 A/B 对照")
    print("=" * 62)
    head = f"  {'指标':<14}{'单路':>10}{'双路':>10}{'绝对提升':>12}{'相对提升':>12}"
    print(head)
    for k in metric_keys:
        rel = relative[k]
        rel_text = "N/A" if rel is None else f"{rel:+.1%}"
        print(f"  {k:<14}{summary['single'][k]:>10.4f}{summary['dual'][k]:>10.4f}"
              f"{gains[k]:>+12.4f}{rel_text:>12}")
    print(f"  实际生效改写问句数  {rewrites_ok}/{total}")
    print(f"  改写调用失败数      {rewrite_errors}/{total}")
    if not comparable:
        print("  ⚠️ 存在改写调用失败，两路部分等价 → 结果不可对外引用，请检查网络与 API Key")
    print("=" * 62)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = save_dir / f"ab_dual_path_{ts}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已保存: {out}")


if __name__ == "__main__":
    main()
