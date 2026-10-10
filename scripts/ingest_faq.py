"""把 data/faq/ecommerce_kb.json 入库到 Chroma（本地 bge）。

用法（项目根目录执行）:
    .venv\\Scripts\\python.exe scripts/ingest_faq.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.pipeline import IndexingPipeline
from src.ingestion.models import DocType
from src.ingestion.router import parse_file

FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "faq" / "ecommerce_kb.json"


def main() -> int:
    if not FAQ_PATH.exists():
        print(f"FAQ 文件不存在: {FAQ_PATH}")
        return 1

    print(f"解析 FAQ: {FAQ_PATH}")
    result = parse_file(str(FAQ_PATH), doc_type=DocType.FAQ_JSON)
    if not result.markdown:
        print("解析失败:", result.errors)
        return 1
    print(f"解析成功，markdown {len(result.markdown)} 字符")

    pipeline = IndexingPipeline()
    report = pipeline.run_from_text(
        result.markdown,
        source_file="ecommerce_kb.json",
        doc_type=DocType.FAQ_JSON,
    )
    print("入库报告:", report)
    print("向量库统计:", pipeline.store.stats())
    return 0 if report.get("status") in ("ok", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
