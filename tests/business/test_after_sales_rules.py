"""Deterministic policy consultation vs actions and conditional applicability."""

import json
from datetime import date

import pytest

from src.business.after_sales import after_sales_kind, order_ids
from src.business.return_policy import (
    POLICY_PATH,
    ReturnPolicyService,
    extract_facts,
    is_policy_followup,
)
from src.conversation.human_in_loop import HumanInLoopHandler
from src.routing.models import Intent
from src.routing.router import IntentRouter

TODAY = date(2026, 11, 16)


@pytest.mark.parametrize(
    "query",
    [
        "退款一般多久到账？",
        "退款多久能到账？",
        "如何申请退款？",
        "我想知道退款流程",
        "我不要退款，只想了解退款政策",
        "这盒礼品能退吗",
        "可以退吗",
        "帮我查一下退款政策",
        "帮我看看退款需要多久",
    ],
)
def test_consultations_do_not_force_human(query):
    assert after_sales_kind(query) == "policy"
    assert not HumanInLoopHandler().check_needs_human(
        query, "return_refund", confidence=0.95
    )["needs_human"]


@pytest.mark.parametrize(
    "query",
    [
        "帮我退款",
        "我要申请退款",
        "直接给我退钱",
        "帮我退一下这笔订单",
        "不要解释政策，直接退款",
        "请办理退货",
    ],
)
def test_real_operations_still_require_human(query):
    assert after_sales_kind(query) == "refund_action"
    assert HumanInLoopHandler().check_needs_human(query, "return_refund")["needs_human"]


@pytest.mark.parametrize(
    "query",
    [
        "我的退款还没到账",
        "退款失败",
        "查一下退款进度",
        "我已经退货5天还没收到钱",
        "还没退款",
    ],
)
def test_actual_refund_status_is_not_replaced_with_general_policy(query):
    assert after_sales_kind(query) == "refund_status"
    assert ReturnPolicyService().evaluate(query).needs_human


def test_explicit_consultation_bypasses_classifier_and_rewrite():
    def forbidden(*args):
        raise AssertionError("No LLM needed for explicit policy consultation")

    router = IntentRouter(type("Classifier", (), {"classify": forbidden})())
    router._rewrite_query = forbidden
    result = router.route("退款一般多久到账")
    assert result.intent_result.intent == Intent.RETURN_REFUND
    assert result.rewritten_query == result.query


def test_ambiguous_refund_asks_speech_act_first():
    first = ReturnPolicyService().evaluate("退款", as_of=TODAY)
    assert first.status == "clarify" and first.missing_fields == ["request_type"]
    assert after_sales_kind("帮我办理", first.context) == "refund_action"
    info = ReturnPolicyService().evaluate("我想了解规则", first.context, as_of=TODAY)
    assert info.status == "informational" and "7天无理由" in info.reply


@pytest.mark.parametrize(
    "query, expected",
    [
        ("退款多久能到账？", "1-3个工作日"),
        ("退款退到哪里？", "原支付渠道"),
        ("怎么申请退款？", "提交审核"),
    ],
)
def test_general_policy_uses_relevant_rule(query, expected):
    result = ReturnPolicyService().evaluate(query, as_of=TODAY)
    assert result.status == "informational" and not result.needs_human
    assert expected in result.reply and result.version == "2026-11-16-v1"


def test_missing_facts_are_clarified_across_turns():
    service = ReturnPolicyService()
    first = service.evaluate("这盒礼品能退吗", as_of=TODAY)
    assert first.status == "clarify" and "category" in first.missing_fields
    second = service.evaluate("签收5天了", first.context, as_of=TODAY)
    final = service.evaluate(
        "普通衣服，未拆封，想无理由退货", second.context, as_of=TODAY
    )
    assert final.status == "conditional" and "5天" in final.reply
    assert "不代表退货或退款已获批" in final.reply


@pytest.mark.parametrize("days,status", [(7, "conditional"), (8, "manual_review")])
def test_return_window_boundary(days, status):
    result = ReturnPolicyService().evaluate(
        f"这件普通衣服签收{days}天，未拆封，能无理由退吗", as_of=TODAY
    )
    assert result.status == status


@pytest.mark.parametrize("days,status", [(15, "conditional"), (16, "manual_review")])
def test_quality_window_boundary(days, status):
    result = ReturnPolicyService().evaluate(
        f"这件普通衣服签收{days}天有质量问题能退吗", as_of=TODAY
    )
    assert result.status == status


@pytest.mark.parametrize("category", ["生鲜", "定制商品", "贴身衣物"])
def test_special_categories_do_not_receive_ordinary_eligibility(category):
    result = ReturnPolicyService().evaluate(
        f"这件{category}想无理由退货，能退吗", as_of=TODAY
    )
    assert result.status == "not_applicable" and "不适用" in result.reply


def test_contradictory_use_claims_require_clarification():
    result = ReturnPolicyService().evaluate(
        "这件普通衣服签收3天，无理由能退吗，没用过但用了两天", as_of=TODAY
    )
    assert result.status == "clarify" and "unused" in result.missing_fields
    used = ReturnPolicyService().evaluate(
        "这件普通衣服签收3天，无理由能退吗，不是没用过", as_of=TODAY
    )
    assert used.status == "not_applicable"


def test_opened_cosmetics_do_not_pass_unopened_rule():
    service = ReturnPolicyService()
    opened = service.evaluate(
        "这瓶化妆品已拆封未使用，签收3天，无理由能退吗", as_of=TODAY
    )
    assert opened.status == "not_applicable"
    missing = service.evaluate("这瓶化妆品未使用，签收3天，无理由能退吗", as_of=TODAY)
    assert missing.status == "clarify" and "opened" in missing.missing_fields


def test_order_verified_signature_cannot_be_overwritten_by_user_statement():
    facts = {"signed_on": "2026-11-01", "signed_source": "order_record"}
    result = ReturnPolicyService().evaluate(
        "这件衣服签收3天未使用，无理由能退吗", facts, as_of=TODAY
    )
    assert (
        result.context["signed_on"] == "2026-11-01" and result.status == "manual_review"
    )


@pytest.mark.parametrize("stamp", ["2026-11-20", "2026-02-31"])
def test_future_and_invalid_signature_dates_do_not_pass(stamp):
    result = ReturnPolicyService().evaluate(
        f"这件普通衣服签收{stamp}未使用，无理由能退吗", as_of=TODAY
    )
    assert result.status == "clarify" and "signed_on" in result.missing_fields


def test_expired_or_broken_policy_fails_closed(tmp_path):
    path = tmp_path / "policy.json"
    rules = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    rules["effective_to"] = "2026-11-15"
    path.write_text(json.dumps(rules), encoding="utf-8")
    assert (
        ReturnPolicyService(path).evaluate("退款一般多久到账", as_of=TODAY).needs_human
    )
    rules["return_days"] = -1
    path.write_text(json.dumps(rules), encoding="utf-8")
    assert ReturnPolicyService(path).evaluate("怎么申请退货", as_of=TODAY).needs_human


def test_unsupported_fee_policy_is_not_answered_with_return_window():
    result = ReturnPolicyService().evaluate("退款手续费怎么收", as_of=TODAY)
    assert result.needs_human and "7天" not in result.reply


def test_general_category_description_follows_config(tmp_path):
    path = tmp_path / "policy.json"
    rules = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    rules["excluded_categories"] = ["custom"]
    rules["sealed_categories"] = ["cosmetics"]
    path.write_text(json.dumps(rules), encoding="utf-8")
    result = ReturnPolicyService(path).evaluate("退货规则是什么", as_of=TODAY)
    assert "排除品类：定制" in result.reply and "拆封情况的品类：化妆品" in result.reply


def test_order_time_and_campaign_exceptions_do_not_use_default_policy():
    service = ReturnPolicyService()
    assert service.evaluate("活动商品可以退吗", as_of=TODAY).needs_human
    result = service.evaluate(
        "这件衣服无理由能退吗",
        {"order_created_on": "2025-12-01", "signed_on": "2026-11-15", "unused": True},
        as_of=TODAY,
    )
    assert result.needs_human and "下单时间" in result.reply


def test_changing_order_drops_old_applicability_facts():
    old = {
        "order_id": "ORD000001",
        "category": "ordinary",
        "signed_on": "2026-11-15",
        "signed_source": "order_record",
        "pending_scenario": "window",
    }
    new = extract_facts("订单号ORD000002", old, TODAY)
    assert (
        new["order_id"] == "ORD000002"
        and "category" not in new
        and "signed_on" not in new
    )
    assert not is_policy_followup("另一件衣服多少钱", old)
    assert not is_policy_followup("这件衣服价格多少", old)
    assert order_ids("订单ORD000001能退吗") == ["ORD000001"]


def test_complaints_and_food_incidents_still_escalate():
    handler = HumanInLoopHandler()
    assert handler.check_needs_human("退款一般多久到账，我要投诉", "return_refund")[
        "needs_human"
    ]
    assert (
        handler.check_needs_human("吃了礼盒食品拉肚子", "product_consult")["scenario"]
        == "food_safety"
    )
