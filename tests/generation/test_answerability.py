"""Coverage verdicts require real spans; missing evidence/model errors fail closed."""

from types import SimpleNamespace

import pytest

from src.generation.answerability import AnswerabilityVerifier, evidence_input

DOCS = [{"chunk_id": "faq-060", "text": "本平台不支持货到付款，所有订单需在线支付。"}]


def verdict():
    return {
        "complete": True,
        "reason": "支付方式明确",
        "supports": [
            {
                "doc_id": "faq-060",
                "source_quote": "不支持货到付款",
                "answer_quote": "不支持货到付款",
            }
        ],
    }


def test_coverage_has_separate_evidence_and_answer_checks():
    calls = []

    def chat_json(**kwargs):
        calls.append(kwargs)
        return verdict()

    verifier = AnswerabilityVerifier(SimpleNamespace(chat_json=chat_json))
    before = verifier.assess("支持货到付款吗", DOCS)
    after = verifier.assess("支持货到付款吗", DOCS, answer="本店不支持货到付款。")
    assert before.status == after.status == "complete"
    assert before.input_hash != after.input_hash
    assert calls[0]["messages"][0]["role"] == "system"
    assert '"answer"' not in calls[0]["messages"][1]["content"]
    assert '"answer"' in calls[1]["messages"][1]["content"]


@pytest.mark.parametrize(
    "problem", ["unknown_doc", "source", "answer", "empty", "string_bool", "extra"]
)
def test_invalid_verdict_never_releases_answer(problem):
    raw = verdict()
    if problem == "unknown_doc":
        raw["supports"][0]["doc_id"] = "invented"
    elif problem == "source":
        raw["supports"][0]["source_quote"] = "支持货到付款并免邮"
    elif problem == "answer":
        raw["supports"][0]["answer_quote"] = "支持货到付款并免邮"
    elif problem == "empty":
        raw["supports"] = []
    elif problem == "string_bool":
        raw["complete"] = "true"
    else:
        raw["trusted"] = True
    answer = "本店不支持货到付款。"
    verifier = AnswerabilityVerifier(SimpleNamespace(chat_json=lambda **kw: raw))
    assert verifier.assess("付款方式", DOCS, answer=answer).status == "unavailable"


def test_absent_evidence_does_not_call_model():
    def forbidden(**kwargs):
        raise AssertionError("No model call without evidence")

    verifier = AnswerabilityVerifier(SimpleNamespace(chat_json=forbidden))
    assert verifier.assess("unknown", []).status == "insufficient"


def test_partial_answer_and_backend_error_fail_closed():
    verifier = AnswerabilityVerifier(
        SimpleNamespace(
            chat_json=lambda **kw: {
                "complete": False,
                "reason": "未包含问题要求的具体型号参数",
                "supports": [],
            }
        )
    )
    assert verifier.assess("X9电池容量", DOCS).status == "insufficient"

    def unavailable(**kwargs):
        raise TimeoutError("private-provider-secret")

    verifier = AnswerabilityVerifier(SimpleNamespace(chat_json=unavailable))
    result = verifier.assess("付款方式", DOCS)
    assert (
        result.status == "unavailable"
        and "private-provider-secret" not in result.reason
    )


def test_quoted_evidence_must_be_inside_generator_excerpt():
    docs = [{"chunk_id": "faq-060", "text": "a" * 800 + "不支持货到付款"}]
    verifier = AnswerabilityVerifier(SimpleNamespace(chat_json=lambda **kw: verdict()))
    assert verifier.assess("付款方式", docs).status == "unavailable"
