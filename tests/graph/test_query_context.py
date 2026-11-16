"""A resolved follow-up must reach classification, retrieval and generation once."""

from types import SimpleNamespace

import pytest

from src.business.after_sales import after_sales_kind
from src.graph import workflow
from src.routing.models import Intent, IntentResult, RouteResult, RouteTarget


def test_real_graph_resolves_original_followup_before_lossy_rewrite(monkeypatch):
    from src.generation.answerability import Assessment, evidence_input

    def assess(query, docs, *, answer=None):
        return Assessment(
            "complete",
            "covered",
            evidence_input(query, docs, answer)[1],
            [
                {
                    "doc_id": docs[0]["chunk_id"],
                    "source_quote": docs[0]["text"],
                    "answer_quote": answer or "",
                }
            ],
        )

    monkeypatch.setattr(
        "src.generation.answerability.get_answerability_verifier",
        lambda: SimpleNamespace(assess=assess),
    )
    raw = "那多久能收到邮件呢"
    complete = "电子发票申请后多久发送到邮箱"
    calls = []

    def resolve(query, history, memory):
        calls.append(("resolve", query))
        assert query == raw
        assert "发票" in history[-1]["content"]
        return complete

    def route(query):
        calls.append(("classify", query))
        assert query == complete
        return RouteResult(
            intent_result=IntentResult(
                intent=Intent.ORDER_QUERY, confidence=0.95, reasoning="invoice"
            ),
            target=RouteTarget.RAG,
            query=query,
            rewritten_query=query,
            handler="rag",
        )

    def search(**kwargs):
        assert kwargs["query"] == raw  # Original path remains a recall fallback.
        assert kwargs["secondary_query"] == complete
        assert kwargs["access_level"] == "member"
        return SimpleNamespace(
            response=SimpleNamespace(
                results=[
                    SimpleNamespace(
                        chunk_id="faq-054",
                        text="申请后1-3个工作日发送到邮箱",
                        score=0.9,
                        source_file="faq.json",
                        doc_type="faq_json",
                    )
                ],
                elapsed_ms=1,
            ),
            level=1,
            method="dense",
        )

    def generate(**kwargs):
        calls.append(("generate", kwargs["query"]))
        assert kwargs["query"] == complete
        assert kwargs["docs"] == ["申请后1-3个工作日发送到邮箱"]
        assert kwargs["access_level"] == "member"
        return SimpleNamespace(
            answer="电子发票申请后1-3个工作日发送到邮箱。",
            faithfulness=1.0,
            correction_rounds=0,
            was_corrected=False,
        )

    monkeypatch.setattr(
        "src.conversation.get_coreference_resolver",
        lambda: SimpleNamespace(resolve=resolve),
    )
    monkeypatch.setattr(workflow, "get_router", lambda: SimpleNamespace(route=route))
    monkeypatch.setattr("src.embedding.retriever.get_retriever", lambda: object())
    monkeypatch.setattr(
        workflow,
        "get_degradation_strategy",
        lambda retriever: SimpleNamespace(search_with_degradation=search),
    )
    monkeypatch.setattr(
        workflow,
        "get_corrector",
        lambda retriever: SimpleNamespace(generate_with_docs=generate),
    )
    graph = workflow.RAGWorkflow()
    result = graph.run(
        raw,
        history=[{"role": "user", "content": "电子发票怎么申请"}],
        access_level="member",
    )
    assert result["query"] == raw
    assert result["rewritten_query"] == complete
    assert not result["needs_human"]
    assert calls == [("resolve", raw), ("classify", complete), ("generate", complete)]


@pytest.mark.parametrize(
    "query,kind",
    [
        ("先不说退货了，店铺券和平台券可以一起用吗", "none"),
        ("不聊退款了，手机壳什么材质", "none"),
        ("先不说退货了，直接帮我退款", "refund_action"),
        ("先不说退货了，退款还是没到账", "refund_status"),
        ("不要解释，直接帮我退款", "refund_action"),
    ],
)
def test_explicit_topic_dismissal_preserves_later_positive_actions(query, kind):
    assert after_sales_kind(query) == kind
