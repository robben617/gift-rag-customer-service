"""离线演示检索 — 纯标准库实现，不加载任何模型、不依赖任何外部服务

用途
----
让访客在 **没有本地 BGE 模型、没有 API Key、没有 MySQL / Redis** 的前提下，
克隆仓库后就能把「知识库问答」这条最短链路跑起来，看到真实的问答形态。

与生产链路的区别（务必注意，不要混淆）
--------------------------------------
    生产链路 = 双路召回 + Chroma 向量检索 + CrossEncoder 精排 + LLM 生成 + claims 级幻觉核验
    本模块   = 字符二元组 IDF 加权的关键词匹配 + FAQ 原文直出，**没有任何生成与核验**

因此本模块的输出**不代表系统真实回答质量**，只用于展示交互形态与召回是否命中。

设计约束
--------
    - 只依赖 Python 标准库（不引入 jieba / numpy / torch），保证零安装成本
    - 不读取 .env、不初始化任何单例、不触碰网络与数据库
    - 评分是「查询词命中比例」型指标，落在 0~1，阈值语义与生产链路的兜底一致
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 仓库根目录：按文件位置定位，保证从任意工作目录运行都能找到知识库
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认知识库：仓库内已入库的 60 条 FAQ
DEFAULT_FAQ_PATH = PROJECT_ROOT / "data" / "faq" / "ecommerce_kb.json"

# 低于该匹配度视为「知识库无相关内容」，与生产链路的诚实兜底语义一致
DEFAULT_MIN_SCORE = 0.25
DEFAULT_TOP_K = 3

# 高敏话题：真实系统中此类问题必须过 claims 级忠实度核验，不达标强制转人工
HIGH_STAKES_KEYWORDS: tuple[str, ...] = (
    "退款",
    "赔付",
    "赔偿",
    "价格",
    "报价",
    "承诺",
    "保质期",
    "过期",
    "过敏",
    "过敏原",
    "食品安全",
    "召回",
    "投诉",
    "假货",
    # 口语化说法同样要识别，否则「多少钱 / 能便宜点吗」会绕过护栏
    "多少钱",
    "便宜",
    "打折",
    "折扣",
    "涨价",
    "差价",
    "免单",
)

# 查询词与 FAQ 关键词字面命中时的加成。
# 取 0.20 的实测依据：49 题评测集上「是假货 / 客服态度差 / 退款退到哪里」这类
# 关键词清晰的问法能从 0.09~0.21 抬到阈值之上，而「现在几点了」等库外问法不变。
KEYWORD_BONUS = 0.20

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """中文按字符二元组切分，英文/数字按单词切分，标点与空白直接丢弃。

    用二元组而不是 jieba：标准库就能完成，且对「怎么退货 / 退货流程」这类短语召回不差。
    """
    tokens: list[str] = []
    for chunk in _CJK_RUN.findall(text):
        if len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    tokens.extend(word.lower() for word in _ASCII_WORD.findall(text))
    return tokens


def _dedupe(tokens: list[str]) -> list[str]:
    """去重且保持首次出现顺序（顺序只影响可读性，不影响打分）"""
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


@dataclass
class FAQHit:
    """一条 FAQ 命中结果"""

    question: str
    answer: str
    keywords: list[str]
    score: float


@dataclass
class DemoAnswer:
    """演示模式的一次问答结果"""

    text: str
    hits: list[FAQHit] = field(default_factory=list)
    high_stakes: bool = False
    matched: bool = True
    top_score: float = 0.0


class OfflineFAQIndex:
    """FAQ 关键词索引（IDF 加权 + 字符二元组）

    使用方式:
        index = OfflineFAQIndex()
        result = index.answer("怎么退货？")
        print(result.text)
    """

    def __init__(
        self,
        faq_path: str | Path | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self._path = Path(faq_path) if faq_path is not None else DEFAULT_FAQ_PATH
        self._min_score = min_score
        self._entries: list[dict] = []
        self._doc_tokens: list[set[str]] = []
        self._idf: dict[str, float] = {}
        self._max_idf: float = 1.0

        self._load()
        self._build_index()

    # ── 只读属性（供脚本/测试读取，避免外部直接碰私有字段） ──

    @property
    def faq_path(self) -> Path:
        return self._path

    @property
    def min_score(self) -> float:
        return self._min_score

    @property
    def size(self) -> int:
        return len(self._entries)

    # ── 构建 ──

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(
                f"未找到知识库文件: {self._path}（应在仓库 data/faq/ 下，请确认工作目录）"
            )
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError(f"知识库文件格式异常，期望非空列表: {self._path}")
        self._entries = data
        logger.info("离线演示索引加载 %d 条 FAQ: %s", len(self._entries), self._path)

    def _build_index(self) -> None:
        total = len(self._entries)
        df: dict[str, int] = {}
        for entry in self._entries:
            tokens = set(_tokenize(self._entry_text(entry)))
            self._doc_tokens.append(tokens)
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        # 平滑 IDF：出现越少的词权重越高
        self._idf = {
            token: math.log((total + 1) / (count + 1)) + 1.0
            for token, count in df.items()
        }
        # 完全未出现的查询词按最高权重计，使「知识库外的问法」得分被自然压低
        self._max_idf = math.log((total + 1) / 1) + 1.0

    @staticmethod
    def _entry_text(entry: dict) -> str:
        keywords = entry.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        return " ".join(
            [
                str(entry.get("question", "")),
                str(entry.get("answer", "")),
                " ".join(str(k) for k in keywords),
            ]
        )

    # ── 检索 ──

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[FAQHit]:
        """返回按匹配度降序的 Top-K 命中；查询为空时返回空列表"""
        if not query or not query.strip():
            return []

        query_tokens = _dedupe(_tokenize(query))
        if not query_tokens:
            return []

        denominator = sum(self._idf.get(t, self._max_idf) for t in query_tokens)
        if denominator <= 0:
            return []

        hits: list[FAQHit] = []
        for entry, tokens in zip(self._entries, self._doc_tokens):
            numerator = sum(
                self._idf.get(t, self._max_idf) for t in query_tokens if t in tokens
            )
            score = numerator / denominator

            # 关键词字面命中给一点加成，避免短问句（如「退货」「假货」）被长文档稀释
            keywords = entry.get("keywords") or []
            if isinstance(keywords, list) and any(
                str(k) and str(k) in query for k in keywords
            ):
                score += KEYWORD_BONUS

            hits.append(
                FAQHit(
                    question=str(entry.get("question", "")),
                    answer=str(entry.get("answer", "")),
                    keywords=[str(k) for k in keywords]
                    if isinstance(keywords, list)
                    else [],
                    score=round(min(1.0, max(0.0, score)), 4),
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    # ── 组装回答 ──

    def answer(self, query: str, top_k: int = DEFAULT_TOP_K) -> DemoAnswer:
        """组装一次演示回答（FAQ 原文直出，不做任何生成）"""
        hits = self.search(query, top_k=top_k)
        high_stakes = any(kw in query for kw in HIGH_STAKES_KEYWORDS)
        best = hits[0] if hits else None

        if best is None or best.score < self._min_score:
            top_score = best.score if best is not None else 0.0
            return DemoAnswer(
                text=self._no_match_text(top_score, high_stakes, hits),
                hits=hits,
                high_stakes=high_stakes,
                matched=False,
                top_score=top_score,
            )

        return DemoAnswer(
            text=self._match_text(best, high_stakes),
            hits=hits,
            high_stakes=high_stakes,
            matched=True,
            top_score=best.score,
        )

    @staticmethod
    def _high_stakes_notice() -> str:
        return (
            "> ⚠️ 该问题命中高风险话题（退款 / 价格 / 承诺 / 食品安全类）。"
            "真实系统在此路径上会先做 claims 级忠实度核验，**不达标即强制转人工审批**，"
            "不会直接给出承诺性答复。\n"
        )

    @staticmethod
    def _demo_disclaimer() -> str:
        return (
            "> 演示模式：以上内容为**离线 FAQ 关键词检索直接命中的原文**，"
            "**未经过 LLM 生成、claims 级幻觉核验与高敏护栏**。"
            "完整链路（双路召回 / 六级降级 / 幻觉自纠正 / 转人工）见 README。"
        )

    def _match_text(self, best: FAQHit, high_stakes: bool) -> str:
        parts: list[str] = []
        if high_stakes:
            parts.append(self._high_stakes_notice())
        parts.append(f"**匹配度 {best.score:.2f} · 命中条目：{best.question}**\n")
        parts.append(best.answer)
        parts.append("\n" + self._demo_disclaimer())
        return "\n".join(parts)

    def _no_match_text(
        self, top_score: float, high_stakes: bool, hits: list[FAQHit]
    ) -> str:
        parts: list[str] = []
        if high_stakes:
            parts.append(self._high_stakes_notice())
        parts.append(
            f"**未命中知识库**（最高匹配度 {top_score:.2f}，低于阈值 {self._min_score:.2f}）\n"
        )
        parts.append(
            "演示模式下没有可直答的条目。真实系统在此情况下会进入 **L0 至 L5 六级降级**"
            "（查询分解 → 双路召回 → 改写 → Multi-Query/HyDE → 联网搜索 → 诚实兜底），"
            "必要时携带交接包转人工。"
        )
        if hits:
            candidates = "；".join(f"{h.question}（{h.score:.2f}）" for h in hits)
            parts.append(f"\n\n_候选条目（未达阈值，仅供参考）：{candidates}_")
        parts.append("\n" + self._demo_disclaimer())
        return "\n".join(parts)
