"""实测两条入库链路：图片→SiliconFlow VLM、PDF→RapidOCR

输出写为 UTF-8 文件便于核对；控制台只打印状态摘要（避免乱码）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.router import parse_file

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "scripts"


def dump(name: str) -> None:
    p = BASE / name
    print(f"\n===== parsing {name} =====")
    if not p.exists():
        print("FILE NOT FOUND")
        return
    result = parse_file(str(p))
    status = getattr(result.status, "value", str(result.status))
    md = result.markdown or ""
    print(f"status={status} markdown_len={len(md)}")
    if result.errors:
        print("errors=", result.errors[:5])
    if getattr(result, "warnings", None):
        print("warnings=", result.warnings[:3])
    if hasattr(result, "total_pages"):
        print(f"total_pages={result.total_pages} parsed_pages={result.parsed_pages}")
    out_file = OUT / f"out_{name.replace('.', '_')}.txt"
    out_file.write_text(
        f"STATUS: {status}\n"
        f"ERRORS: {result.errors}\n"
        f"WARNINGS: {getattr(result, 'warnings', [])}\n\n"
        f"--- MARKDOWN ---\n{md}",
        encoding="utf-8",
    )
    print(f"saved -> {out_file.name}")


if __name__ == "__main__":
    dump("Chinese.jpg")
    dump("HCAI-AI证书.pdf")
    print("\nDONE")
