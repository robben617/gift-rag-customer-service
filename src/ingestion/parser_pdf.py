"""PDF 解析器 — fitz 文本快路径 + 本地 RapidOCR 扫描页识别

流程:
  1. 逐页分类：fitz.get_text() 文本充足（≥50 字）→ 文本页（零成本）
  2. 文本不足（扫描件）→ 渲染图片 → RapidOCR 本地识别文字
  3. 按页码合并为 Markdown

说明: 本地 RapidOCR 是纯文字 OCR，不做结构化 layout / 图片语义描述。
"""

import logging
import time
from pathlib import Path

import fitz
from PIL import Image

from src.config import settings

from .defenses import validate_file
from .models import ParseResult, ParseStatus, RawDocument

_log = logging.getLogger(__name__)

PDF_DPI = 200  # 扫描页渲染 DPI

_ocr_engine = None  # 模块级懒加载单例，避免每页重复加载模型


def _get_ocr_engine():
    """懒加载 RapidOCR 引擎（进程内单例）"""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def parse_pdf(doc: RawDocument) -> ParseResult:
    """PDF → 文本快路径 + 扫描页本地 OCR → Markdown"""
    t0 = time.time()
    result = ParseResult(document=doc, status=ParseStatus.SUCCESS)

    validation = validate_file(doc.file_path, "pdf")
    result.file_validation = validation
    if not validation.is_valid:
        result.status = ParseStatus.FAILED
        result.errors.extend(validation.errors)
        result.parse_time_ms = (time.time() - t0) * 1000
        return result

    try:
        markdown_text, page_count = _parse_pdf(doc.file_path)
        result.markdown = markdown_text
        result.total_pages = page_count
        result.parsed_pages = page_count
        result.api_calls = 0
        result.api_cost_estimate = 0.0
    except Exception as e:
        _log.exception("PDF 解析失败")
        result.status = ParseStatus.FAILED
        result.errors.append(f"PDF解析失败: {e}")
        result.parse_time_ms = (time.time() - t0) * 1000
        return result

    if not markdown_text or len(markdown_text.strip()) < 20:
        result.status = ParseStatus.PARTIAL
        result.warnings.append("PDF解析结果为空或内容极少")

    result.parse_time_ms = (time.time() - t0) * 1000
    return result


def _render_page(page, target_dpi: int = PDF_DPI) -> Image.Image:
    """fitz 页面 → PIL 图片"""
    mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    if pm.width > 4500 or pm.height > 4500:
        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)


def _page_has_meaningful_images(doc, page_idx: int, min_size: int = 50) -> bool:
    """检测页面是否含有效图片（≥min_size 的非装饰图片）。

    证书/图文混排类 PDF 的正文常在图片里，仅靠 fitz 文本层会丢内容，
    因此含有效图片的页即使已有部分文本也应走 OCR。
    """
    try:
        for img in doc[page_idx].get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue
            if pix.width >= min_size and pix.height >= min_size:
                return True
    except Exception:
        pass
    return False


def _ocr_page(image: Image.Image) -> str:
    """RapidOCR 识别单页文字（逐行拼接）"""
    try:
        engine = _get_ocr_engine()
        result, _ = engine(image)
        lines = []
        for item in result or []:
            # RapidOCR 返回 [box, text, score]
            if len(item) >= 2 and item[1]:
                lines.append(str(item[1]).strip())
        return "\n".join(lines)
    except Exception as e:
        return f"[OCR 失败: {e}]"


def _parse_pdf(file_path: str) -> tuple[str, int]:
    """PDF → 文本快路径 + 扫描页 OCR → (markdown, page_count)"""
    print(f"[PDF] 正在加载 PDF: {file_path}")
    text_map: dict[int, str] = {}
    ocr_pages: list[int] = []
    total_pages = 0

    with fitz.open(file_path) as doc:
        total_pages = doc.page_count
        if total_pages == 0:
            raise RuntimeError("PDF 页数为 0，文件可能为空或损坏")

        max_pages = getattr(settings, "pdf_max_pages", 50) or 50
        limit = min(total_pages, max_pages)
        if total_pages > max_pages:
            print(f"[PDF] 超过 {max_pages} 页，截断到前 {max_pages} 页")

        ocr_images: dict[int, Image.Image] = {}
        for idx in range(limit):
            t = doc[idx].get_text().strip()
            has_img = _page_has_meaningful_images(doc, idx)
            if len(t) >= 50 and not has_img:
                text_map[idx] = t
            else:
                ocr_pages.append(idx)
                if getattr(settings, "pdf_ocr_enabled", True):
                    ocr_images[idx] = _render_page(doc[idx])

    print(f"[PDF] 文本页 {len(text_map)} 个, 扫描页 {len(ocr_pages)} 个")

    # 本地 OCR 扫描页
    ocr_texts: dict[int, str] = {}
    for idx in ocr_pages:
        if idx not in ocr_images:
            continue
        ocr_texts[idx] = _ocr_page(ocr_images[idx])
        print(f"\r[PDF OCR] 进度: {len(ocr_texts)}/{len(ocr_pages)}", end="")
    if ocr_pages:
        print()

    if not text_map and not ocr_texts:
        raise RuntimeError(f"PDF {total_pages} 页均无法提取内容")

    # 按页码合并（剔除 OCR 失败页）
    failed_count = 0
    markdown_parts = []
    for idx in range(limit):
        page_num = idx + 1
        if idx in text_map:
            markdown_parts.append(f"## 第{page_num}页\n\n{text_map[idx]}")
            continue
        txt = ocr_texts.get(idx, "")
        if not txt or txt.startswith("[OCR 失败"):
            failed_count += 1
            continue
        markdown_parts.append(f"## 第{page_num}页\n\n{txt}")

    if not markdown_parts:
        raise RuntimeError(
            f"全部 {limit} 页解析均失败，请检查 PDF 是否可读或 OCR 依赖是否安装"
        )

    markdown_text = "\n\n".join(markdown_parts)
    print(
        f"[PDF] 完成: {len(markdown_text)} 字符, OCR {len(ocr_pages)} 页, 失败 {failed_count} 页"
    )
    return markdown_text, limit
