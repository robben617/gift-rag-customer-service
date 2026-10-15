"""离线演示检索回归测试

为什么单独测这块:
  这条链路是「访客不装模型、不填 Key 就能跑通」的唯一入口，
  它一旦静默退化（比如阈值调错把库外问题也答了，或加载不到知识库直接崩），
  损失的不是一个功能，而是项目的第一印象。
  因此这里锁住三件事：能加载、该中的中、不该答的不答。

约束: 纯标准库实现，测试不加载模型/不联网，毫秒级完成。
"""

import ast
import json
from pathlib import Path

import pytest

from src.demo import offline_retrieval as mod
from src.demo.offline_retrieval import (
    DEFAULT_FAQ_PATH,
    KEYWORD_BONUS,
    DemoAnswer,
    OfflineFAQIndex,
)

MODULE_FILE = Path(mod.__file__)


@pytest.fixture(scope="module")
def index() -> OfflineFAQIndex:
    return OfflineFAQIndex()


# ── 加载 ──


def test_default_faq_path_exists_and_loads(index: OfflineFAQIndex) -> None:
    """默认知识库必须存在且条数与仓库内 FAQ 一致"""
    assert DEFAULT_FAQ_PATH.exists(), f"知识库缺失: {DEFAULT_FAQ_PATH}"
    payload = json.loads(DEFAULT_FAQ_PATH.read_text(encoding="utf-8"))
    assert index.size == len(payload) == 60


def test_missing_faq_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="未找到知识库文件"):
        OfflineFAQIndex(faq_path=tmp_path / "not-exist.json")


def test_bad_faq_format_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"question": "not a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="格式异常"):
        OfflineFAQIndex(faq_path=bad)


# ── 该中的必须中 ──


@pytest.mark.parametrize(
    ("query", "expected_keyword"),
    [
        ("怎么申请退货？", "退货"),
        ("退款要等多久才能到账？", "退款"),
        ("快递显示签收但我没收到包裹", "签收"),
        ("怎么用优惠券？", "优惠券"),
        ("怀疑是假货", "假货"),
    ],
)
def test_relevant_query_hits(
    index: OfflineFAQIndex, query: str, expected_keyword: str
) -> None:
    """相关问法必须命中，且命中的是同一主题的条目

    注意：FAQ 的 answer 里不一定出现该关键词（例如「怀疑是假货」的回答只写了鉴定与赔偿），
    因此主题一致性要在「问题 + 回答 + 关键词」整体上校验，而不是只看 answer。
    """
    answer = index.answer(query)
    assert answer.matched, f"应当命中: {query}（最高分 {answer.top_score}）"
    assert answer.top_score >= index.min_score

    hit = answer.hits[0]
    haystack = f"{hit.question}{hit.answer}{''.join(hit.keywords)}"
    assert expected_keyword in haystack


def test_exact_question_scores_top(index: OfflineFAQIndex) -> None:
    """与 FAQ 原句完全一致时，必须命中该条且分数为满分"""
    question = index.search("怎么申请退货？")[0].question
    answer = index.answer(question)
    assert answer.hits[0].question == question
    assert answer.top_score == 1.0


def test_answer_is_faq_original_text(index: OfflineFAQIndex) -> None:
    """演示模式直出 FAQ 原文，不允许出现加工痕迹"""
    answer = index.answer("怎么申请退货？")
    assert answer.hits[0].answer in answer.text


# ── 不该答的必须不答 ──


@pytest.mark.parametrize("query", ["你好", "谢谢你的帮助", "现在几点了？"])
def test_out_of_scope_query_is_rejected(index: OfflineFAQIndex, query: str) -> None:
    """寒暄与库外问题不能瞎答，必须走「未命中」分支"""
    answer = index.answer(query)
    assert not answer.matched
    assert "未命中知识库" in answer.text
    assert answer.top_score < index.min_score


def test_empty_query_does_not_raise(index: OfflineFAQIndex) -> None:
    for query in ("", "   ", "！？。"):
        answer = index.answer(query)
        assert isinstance(answer, DemoAnswer)
        assert not answer.matched
        assert answer.hits == []
        assert index.search(query) == []


def test_threshold_is_respected(index: OfflineFAQIndex) -> None:
    """阈值调高后，中等匹配度的问法必须从命中转为未命中"""
    query = "退款退到哪里？"
    assert index.answer(query).matched, "默认阈值下该问法应命中"

    strict = OfflineFAQIndex(min_score=0.95)
    assert strict.min_score == 0.95
    assert not strict.answer(query).matched


# ── 高分风险话题必须标记 ──


@pytest.mark.parametrize(
    "query",
    [
        "退款要等多久才能到账？",
        "你们这个多少钱，能便宜点吗",
        "买贵了能退差价吗",
        "食品过期了还能吃吗",
        "我要投诉你们",
    ],
)
def test_high_stakes_query_is_flagged(index: OfflineFAQIndex, query: str) -> None:
    """命中高风险词时必须给出转人工提示，否则演示会误导成系统做了承诺"""
    answer = index.answer(query)
    assert answer.high_stakes
    assert "强制转人工审批" in answer.text


def test_normal_query_not_flagged(index: OfflineFAQIndex) -> None:
    answer = index.answer("怎么用优惠券？")
    assert not answer.high_stakes
    assert "强制转人工审批" not in answer.text


# ── 打分与候选 ──


def test_scores_sorted_and_bounded(index: OfflineFAQIndex) -> None:
    hits = index.search("怎么申请退货？", top_k=5)
    scores = [h.score for h in hits]
    assert len(hits) == 5
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_no_match_still_returns_candidates(index: OfflineFAQIndex) -> None:
    """未命中时也要给出候选，界面上不能是死路"""
    answer = index.answer("现在几点了？")
    assert not answer.matched
    assert answer.hits, "未命中时应返回候选项供人判断"
    assert "候选条目" in answer.text


def test_every_answer_has_demo_disclaimer(index: OfflineFAQIndex) -> None:
    """无论命中与否，都必须声明这是演示链路、不代表真实质量"""
    for query in ("怎么申请退货？", "现在几点了？"):
        assert "演示模式" in index.answer(query).text


def test_keyword_bonus_is_reasonable() -> None:
    """加成系数要在合理区间：太小救不回短口语问句，太大会把库外问句也答了"""
    assert 0 < KEYWORD_BONUS <= 0.3


def test_keyword_bonus_lifts_borderline_query(index: OfflineFAQIndex) -> None:
    """「卖假货」这类口语短问句纯词重叠分低于阈值，必须靠关键词加成才能命中"""
    answer = index.answer("我要投诉！卖假货你们管不管？")
    assert answer.matched
    assert answer.top_score >= index.min_score


# ── 设计约束 ──


def test_module_only_imports_stdlib() -> None:
    """这条链路的卖点是「零安装」：不能悄悄引入 jieba / numpy / torch 等依赖"""
    tree = ast.parse(MODULE_FILE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    forbidden = {
        "jieba",
        "numpy",
        "torch",
        "sentence_transformers",
        "gradio",
        "chromadb",
    }
    assert not (imported & forbidden), f"离线演示不应引入重依赖: {imported & forbidden}"
    assert imported <= {
        "__future__",
        "json",
        "logging",
        "math",
        "re",
        "dataclasses",
        "pathlib",
        "typing",
    }
