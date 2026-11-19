"""Coverage verdicts require real spans; missing evidence/model errors fail closed."""

from types import SimpleNamespace

import pytest

from src.generation.answerability import (
    LOOKUP_GUIDANCE_PROMPT,
    AnswerabilityVerifier,
    evidence_input,
    requests_lookup_guidance,
)

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


@pytest.mark.parametrize(
    "question,expected",
    [
        ("预售哪天发货，去哪儿查最准？", True),
        ("我去哪个页面核对发货时间？", True),
        ("付款后怎么查询发货时间？", True),
        ("预售礼盒明天几点发？", False),
        ("哪里能买到这个礼盒？", False),
        ("现在买能保证今晚到吗？", False),
    ],
)
def test_lookup_rule_requires_how_or_where_to_verify(question, expected):
    assert requests_lookup_guidance(question) is expected


def test_lookup_policy_reaches_both_verifiers_without_bypassing_spans():
    docs = [{"chunk_id": "shipping", "text": "预售商品以页面标注时间为准。"}]
    q = "预售的发货日期看哪里最准？能确定是哪天吗？"
    answer = "请到商品页查看标注时间，现有资料无法确认具体日期。"
    prompts = []

    def verify(**kwargs):
        messages = kwargs["messages"]
        prompts.append(messages[0]["content"])
        return {
            "complete": True,
            "reason": "指引适用于预售发货时间",
            "supports": [
                {
                    "doc_id": "shipping",
                    "source_quote": "以页面标注时间为准",
                    "answer_quote": "查看标注时间",
                },
            ],
        }

    verifier = AnswerabilityVerifier(SimpleNamespace(chat_json=verify))
    assert verifier.assess(q, docs).status == "complete"
    assert verifier.assess(q, docs, answer=answer).status == "complete"
    assert verifier.assess(q, docs, answer="明天会发货。").status == "unavailable"
    verifier.assess("我的商品明天几点发货？", docs)
    assert all(LOOKUP_GUIDANCE_PROMPT in prompt for prompt in prompts[:3])
    assert LOOKUP_GUIDANCE_PROMPT not in prompts[3]
