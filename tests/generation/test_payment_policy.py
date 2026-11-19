"""A payment policy can be quoted only for a complete, known request and source."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.embedding.degradation import DegradationStrategy
from src.embedding.models import SearchResponse, SearchResult
from src.evaluation.id_baseline import faq_text
from src.generation.payment_policy import match_payment_policy
from src.graph import workflow


@pytest.fixture
def official():
    corpus = json.loads(Path("data/faq/ecommerce_kb.json").read_text(encoding="utf-8"))
    item = next(row for row in corpus if row["faq_id"] == "faq-060")
    # FAQSplitter preserves this Q&A block, but regular ingestion IDs are not faq-060.
    return {
        "chunk_id": "ingested-cod-chunk",
        "doc_type": "faq_json",
        "text": faq_text(item, 60),
    }


@pytest.mark.parametrize(
    "query",
    [
        "能不能东西到了我手上再给钱啊",
        "换个话题，能到货再付款不",
        "请问，能不能货到付款呀？",
        "这款礼盒收到货再给钱行不行？",
    ],
)
def test_matches_complete_cash_on_delivery_question(official, query):
    quote = match_payment_policy(query, "本平台是否支持货到付款？", [official])
    assert quote and quote.doc_id == official["chunk_id"]
    assert "不支持货到付款" in quote.text
    assert quote.text in official["text"]


@pytest.mark.parametrize(
    "query",
    [
        "能货到付款吗？我的订单是不是已经付款了？",
        "给订单 ORD20261119001 改成货到付款",
        "运费能到付吗？",
        "不说货到付款了，礼盒含花生吗？",
        "不是问货到付款，是问能不能先退钱",
    ],
)
def test_rewrite_cannot_hide_an_extra_question_or_operation(official, query):
    assert match_payment_policy(query, "支持货到付款吗？", [official]) is None


@pytest.mark.parametrize(
    "change", ["missing", "revised", "wrong_type", "partial", "conflict"]
)
def test_absent_or_inapplicable_evidence_never_quotes(official, change):
    docs = [official]
    if change == "missing":
        docs = []
    elif change == "revised":
        docs = [
            {
                **official,
                "text": official["text"].replace(
                    "。支付成功", "，部分商品除外。支付成功"
                ),
            }
        ]
    elif change == "wrong_type":
        docs = [{**official, "doc_type": "web"}]
    elif change == "partial":
        docs = [{**official, "text": "本平台不支持货到付款，所有订单需在线支付。"}]
    elif change == "conflict":
        docs.append(
            {
                "chunk_id": "newer",
                "doc_type": "faq_json",
                "text": "会员订单可以货到付款。",
            }
        )
    q = "支持货到付款吗？"
    assert match_payment_policy(q, q, docs) is None


def test_completed_query_must_remain_in_scope(official):
    assert (
        match_payment_policy(
            "能货到付款吗？", "能货到付款吗？我这单已经缴费了吗？", [official]
        )
        is None
    )


def test_low_similarity_policy_candidate_and_reply_need_no_model(official, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No model call for this verified payment FAQ")

    item = SearchResult(**official, score=0.55, source_file="ecommerce_kb.json")
    retriever = SimpleNamespace(
        search_dual_path=lambda **kw: SearchResponse(
            query=kw["query"],
            results=[item],
            total_found=1,
            elapsed_ms=1,
        ),
        reranker=None,
    )
    monkeypatch.setattr("src.embedding.retriever.get_retriever", lambda: retriever)
    monkeypatch.setattr(
        workflow, "get_degradation_strategy", lambda retr: DegradationStrategy(retr)
    )
    monkeypatch.setattr(
        "src.generation.answerability.get_answerability_verifier", forbidden
    )
    monkeypatch.setattr(workflow, "get_corrector", forbidden)
    state = {
        "query": "能不能东西到了我手上再给钱啊",
        "resolved_query": "支持货到付款吗？",
        "rewritten_query": "是否支持货到付款",
        "use_reranker": False,
    }
    state.update(workflow.retrieve_docs(state))
    assert state["degradation_method"] == "evidence_recovered"
    state.update(workflow.generate_answer(state))
    assert state["evidence_status"] == state["answer_status"] == "complete"
    assert state["answer"][:-3] in official["text"] and state["answer"].endswith("[1]")
    assert state["answer_assessment"]["supports"][0]["doc_id"] == official["chunk_id"]
    assert workflow.evaluate_quality(state)["evaluation_passed"]
