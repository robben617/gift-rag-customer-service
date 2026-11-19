"""Evidence recovery, cautious wording and fail-closed workflow invariants."""

from types import SimpleNamespace

import pytest

from src.embedding.degradation import DegradationResult, DegradationStrategy
from src.embedding.models import SearchResponse, SearchResult
from src.generation.answerability import Assessment, evidence_input
from src.graph import workflow


def response(score, doc_id="faq-060"):
    return SearchResponse(
        query="q",
        results=[
            SearchResult(chunk_id=doc_id, text="本平台不支持货到付款。", score=score)
        ],
        total_found=1,
        elapsed_ms=1,
    )


def test_rewrite_that_passes_threshold_does_not_need_extra_margin(monkeypatch):
    results = iter([response(0.69), response(0.71)])
    strategy = DegradationStrategy(SimpleNamespace(search=lambda **kw: next(results)))
    monkeypatch.setattr(strategy, "_rewrite_query", lambda *args: "rewrite")
    result = strategy.search_with_degradation("付款方式")
    assert result.level == 2 and result.response.results[0].score == 0.71
    assert result.candidates  # A sufficient rewrite must not erase earlier candidates.


def test_insufficient_candidates_survive_but_are_not_admitted(monkeypatch):
    strategy = DegradationStrategy(SimpleNamespace(search=lambda **kw: response(0.55)))
    monkeypatch.setattr(strategy, "_rewrite_query", lambda *args: "rewrite")
    monkeypatch.setattr(
        strategy, "_expand_and_search", lambda *args, **kw: response(0.56, "faq-013")
    )
    monkeypatch.setattr(strategy, "_web_search", lambda *args: [])
    result = strategy.search_with_degradation("付款方式")
    assert result.level == 5 and not result.response.results
    assert {row.chunk_id for row in result.candidates} == {"faq-060", "faq-013"}


@pytest.mark.parametrize(
    "status,expected",
    [("complete", True), ("insufficient", False), ("unavailable", False)],
)
def test_only_explicit_coverage_can_rescue_local_candidates(
    monkeypatch, status, expected
):
    result = DegradationResult(
        response=SearchResponse(query="q", results=[], total_found=0, elapsed_ms=1),
        level=5,
        method="fallback",
        candidates=response(0.55).results,
    )
    monkeypatch.setattr(
        "src.embedding.retriever.get_retriever", lambda: SimpleNamespace(reranker=None)
    )
    monkeypatch.setattr(
        workflow,
        "get_degradation_strategy",
        lambda retriever: SimpleNamespace(search_with_degradation=lambda **kw: result),
    )

    def assess(query, docs, **kwargs):
        return Assessment(
            status,
            "coverage",
            evidence_input(query, docs)[1],
            [{"doc_id": "faq-060", "source_quote": "不支持货到付款"}]
            if expected
            else [],
        )

    monkeypatch.setattr(
        "src.generation.answerability.get_answerability_verifier",
        lambda: SimpleNamespace(assess=assess),
    )
    state = {
        "query": "到货给钱行吗",
        "rewritten_query": "货到付款",
        "resolved_query": "到货给钱行吗",
    }
    output = workflow.retrieve_docs(state)
    assert bool(output["retrieved_docs"]) is expected
    assert output["degradation_level"] == (3 if expected else 5)


def test_empty_evidence_never_calls_generation_or_hidden_retrieval(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Generation or second retrieval without evidence")

    monkeypatch.setattr(workflow, "get_corrector", forbidden)
    monkeypatch.setattr("src.embedding.retriever.get_retriever", forbidden)
    result = workflow.generate_answer(
        {"rewritten_query": "unknown", "retrieved_docs": []}
    )
    assert result["evidence_status"] == "insufficient"
    assert result["faithfulness"] == 0


@pytest.mark.parametrize(
    "status,expected_checks", [("complete", 1), ("insufficient", 1), ("unavailable", 2)]
)
def test_initial_coverage_can_skip_rewrites_and_reuses_only_exact_stable_verdicts(
    monkeypatch, status, expected_checks
):
    retriever = SimpleNamespace(search=lambda **kw: response(0.55), reranker=None)
    strategy = DegradationStrategy(retriever)
    rewrites = []

    def rewrite(query, attempt):
        rewrites.append(attempt)
        return query + str(attempt)

    monkeypatch.setattr(strategy, "_rewrite_query", rewrite)
    monkeypatch.setattr(strategy, "_expand_and_search", lambda *a, **kw: response(0.55))
    monkeypatch.setattr(strategy, "_web_search", lambda *a: [])
    monkeypatch.setattr("src.embedding.retriever.get_retriever", lambda: retriever)
    monkeypatch.setattr(workflow, "get_degradation_strategy", lambda _: strategy)
    calls = []

    def assess(query, docs, *, answer=None):
        assert query == "到货后能不能再付钱"
        calls.append(answer)
        supports = (
            [{"doc_id": "faq-060", "source_quote": "不支持货到付款"}]
            if status == "complete"
            else []
        )
        return Assessment(
            status, "controlled", evidence_input(query, docs, answer)[1], supports
        )

    monkeypatch.setattr(
        "src.generation.answerability.get_answerability_verifier",
        lambda: SimpleNamespace(assess=assess),
    )
    state = {"query": "到货后能不能再付钱", "rewritten_query": "到货后能不能再付钱"}
    output = workflow.retrieve_docs(state)
    assert len(calls) == expected_checks
    if status == "complete":
        assert rewrites == []
        assert output["degradation_method"] == "evidence_recovered"
        assert output["evidence_assessment"]["status"] == "complete"
        monkeypatch.setattr(
            workflow,
            "get_corrector",
            lambda _: SimpleNamespace(
                generate_with_docs=lambda **kw: SimpleNamespace(
                    answer="不支持货到付款。",
                    faithfulness=1,
                    correction_rounds=0,
                    was_corrected=False,
                )
            ),
        )
        assert (
            workflow.generate_answer({**state, **output})["answer_status"] == "complete"
        )
        assert calls == [None, "不支持货到付款。"]
    else:
        assert rewrites == [0, 1]
        assert output["degradation_method"] == "fallback"
        assert not output["retrieved_docs"]


def test_generation_and_coverage_use_user_intent_instead_of_search_keywords(
    monkeypatch,
):
    request = "手机壳一般是什么材质"
    doc = {"chunk_id": "faq-032", "text": "手机壳采用TPU软胶加PC硬背板双材质。"}
    calls = []

    def assess(query, docs, *, answer=None):
        assert query == request
        calls.append("evidence" if answer is None else "answer")
        return Assessment(
            "complete",
            "covered within document scope",
            evidence_input(query, docs, answer)[1],
            [
                {
                    "doc_id": doc["chunk_id"],
                    "source_quote": doc["text"],
                    "answer_quote": answer or "",
                }
            ],
        )

    def generate(**kwargs):
        assert kwargs["query"] == request
        calls.append("generate")
        return SimpleNamespace(
            answer="资料所述手机壳采用TPU和PC。",
            faithfulness=1.0,
            correction_rounds=0,
            was_corrected=False,
        )

    monkeypatch.setattr(
        "src.generation.answerability.get_answerability_verifier",
        lambda: SimpleNamespace(assess=assess),
    )
    monkeypatch.setattr("src.embedding.retriever.get_retriever", lambda: None)
    monkeypatch.setattr(
        workflow,
        "get_corrector",
        lambda retriever: SimpleNamespace(generate_with_docs=generate),
    )
    result = workflow.generate_answer(
        {
            "resolved_query": request,
            "rewritten_query": "手机壳材质种类",
            "retrieved_docs": [doc],
        }
    )
    assert result["evidence_status"] == result["answer_status"] == "complete"
    assert calls == ["evidence", "generate", "answer"]


def test_qualified_complete_answer_can_pass_but_unchecked_refusal_cannot():
    state = {
        "answer": "手机壳为TPU软胶和PC背板；其他未列型号不确定。",
        "faithfulness": 1.0,
        "degradation_level": 1,
    }
    assert not workflow.evaluate_quality(state)["evaluation_passed"]
    assert workflow.evaluate_quality(
        {**state, "evidence_status": "complete", "answer_status": "complete"}
    )["evaluation_passed"]


@pytest.mark.parametrize(
    "state",
    [
        {"faithfulness": 0.5},
        {"degradation_level": 5},
        {"evidence_status": "unavailable"},
        {"answer_status": "insufficient"},
        {"answer": "退款保证明天到账", "faithfulness": 0.8},
    ],
)
def test_max_loop_never_overrides_evidence_faithfulness_or_high_stakes(state):
    base = {
        "answer": "普通商品说明",
        "faithfulness": 1.0,
        "degradation_level": 1,
        "loop_count": workflow.MAX_LOOP_COUNT,
        "evidence_status": "complete",
        "answer_status": "complete",
    }
    assert not workflow.evaluate_quality({**base, **state})["evaluation_passed"]
