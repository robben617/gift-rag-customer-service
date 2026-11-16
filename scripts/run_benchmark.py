"""FAQ ID 基线默认入口；--legacy-similarity 保留历史相似度评测。

默认仅使用本地模型，在独立目录索引固定 FAQ，按文档 ID 计算指标。
--validate-only 无需加载模型，检查标注；--no-rerank 运行 Dense 对照。

用法:
    python scripts/run_benchmark.py                            # ID 检索基线
    python scripts/run_benchmark.py --validate-only            # 标注校验
    python scripts/run_benchmark.py --no-rerank                 # Dense 对照
    python scripts/run_benchmark.py --legacy-similarity --with-generation
    python scripts/run_benchmark.py --limit 10                 # 只跑前10条（冒烟）
    python scripts/run_benchmark.py --save-dir reports         # 报告保存目录

前置条件:
    - 本地 BGE 模型路径：EMBEDDING_MODEL_PATH / RERANKER_MODEL_PATH
    - ID 基线不需要 API Key 或已有知识库；历史模式仍需已入库知识库

输出:
    - ID 基线输出 Hit@5 / Recall@5 / MRR@5 / Precision@5 / NDCG@5
    - JSON 报告保存到 <save-dir>/id_baseline_<时间戳>/report.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# 允许直接从 scripts/ 目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索/生成 benchmark")
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="运行完整 RAG（检索+生成），默认仅检索",
    )
    parser.add_argument(
        "--use-llm-eval",
        action="store_true",
        help="用 LLM 评估忠实度，默认 Embedding 快速模式",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    parser.add_argument("--save-dir", default="reports", help="报告保存目录")
    parser.add_argument("--dataset", type=Path, help="ID 基线评测集 JSON")
    parser.add_argument("--corpus", type=Path, help="ID 基线 FAQ JSON")
    parser.add_argument(
        "--validate-only", action="store_true", help="仅校验 ID 标注，不加载模型"
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="ID 基线只运行 Dense 检索"
    )
    parser.add_argument(
        "--legacy-similarity",
        action="store_true",
        help="历史相似度代理指标（不能与 ID 指标直接比较）",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit 必须 >= 0")
    if not args.legacy_similarity:
        if args.with_generation or args.use_llm_eval:
            parser.error("生成/LLM 评估请显式加 --legacy-similarity；ID 基线仅评估检索")
        from collections import Counter

        from src.evaluation.id_baseline import (
            CORPUS_PATH,
            DATASET_PATH,
            load_labels,
            run_baseline,
        )

        dataset = args.dataset or DATASET_PATH
        corpus = args.corpus or CORPUS_PATH
        try:
            cases, documents = load_labels(dataset, corpus)
            print(f"标注校验通过：{len(cases)} 题 / {len(documents)} FAQ")
            print(dict(Counter(case["retrieval_status"] for case in cases)))
            if args.validate_only:
                return
            run_dir = Path(args.save_dir) / datetime.now().strftime(
                "id_baseline_%Y%m%d_%H%M%S_%f"
            )
            report = run_baseline(
                dataset,
                corpus,
                run_dir,
                use_rerank=not args.no_rerank,
                limit=args.limit,
            )
        except Exception as exc:
            print(f"ID 基线失败（未输出有效跑分）：{exc}")
            raise SystemExit(1) from exc
        for key in (
            "total_cases",
            "scored_cases",
            "hit@5",
            "recall@5",
            "precision@5",
            "mrr@5",
            "ndcg@5",
        ):
            print(f"{key}: {report[key]}")
        print(f"漏检题目: {report['miss_case_ids']}")
        print(f"报告: {run_dir / 'report.json'}")
        return

    if args.dataset or args.corpus or args.validate_only or args.no_rerank:
        parser.error("自定义数据/标注校验/关闭重排选项仅适用于 ID 基线")

    from src.evaluation import get_evaluator

    evaluator = get_evaluator()

    # ── 1. 加载数据集 ──
    test_cases = evaluator.load_dataset()
    if args.limit:
        test_cases = test_cases[: args.limit]
    if not test_cases:
        print("❌ 评估数据集为空")
        sys.exit(1)
    print(f"\n📊 加载评估数据: {len(test_cases)} 条")

    # ── 2. 运行评测 ──
    mode = "完整RAG" if args.with_generation else "仅检索"
    eval_mode = "LLM精确" if args.use_llm_eval else "Embedding快速"
    print(f"🔄 运行模式: {mode} / {eval_mode}")

    try:
        results = evaluator.evaluate_dataset(
            test_cases,
            use_llm_eval=args.use_llm_eval,
            with_generation=args.with_generation,
        )
        results["metric_basis"] = "legacy_embedding_similarity_proxy"
        results["warning"] = (
            "历史相似度代理值；recall@5 实为 Hit@5。检索模式下 faithfulness/keyword_coverage 使用标准答案，不代表生成质量。"
        )
    except Exception as e:
        print(f"\n❌ 评测失败: {e}")
        print(
            "请检查: 1) 知识库是否已入库  2) 本地 BGE 模型是否就位  3) API Key 是否配置(.env)"
        )
        sys.exit(1)

    # ── 3. 输出报告 ──
    _print_report(results)

    # ── 4. 保存报告 ──
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "gen" if args.with_generation else "retrieval"
    report_path = save_dir / f"benchmark_{mode_tag}_{ts}.json"
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n💾 报告已保存: {report_path}")


def _print_report(results: dict) -> None:
    if "error" in results:
        print(f"\n❌ {results['error']}")
        return

    print("\n" + "=" * 56)
    print("  📈 RAG Benchmark 指标报告")
    print("=" * 56)
    quality_keys = [
        ("recall@5", "检索召回率"),
        ("precision@5", "检索精确率"),
        ("mrr", "平均倒数排名"),
        ("ndcg@5", "归一化折损累积"),
        ("faithfulness", "忠实度"),
        ("keyword_coverage", "关键词覆盖率"),
        ("hallucination_rate", "幻觉率"),
        ("avg_correction_rounds", "平均纠正轮数"),
    ]
    for key, label in quality_keys:
        if key in results:
            print(f"  {label:<8} {key:<16} {results[key]:.4f}")
    print("-" * 56)
    latency_keys = [
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "latency_score",
    ]
    for key in latency_keys:
        if key in results:
            print(
                f"  {'延迟' if 'latency_ms' in key else '延迟分':<8} {key:<16} {results[key]}"
            )
    print("=" * 56)


if __name__ == "__main__":
    main()
