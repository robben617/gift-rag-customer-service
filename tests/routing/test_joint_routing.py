"""The router consumes one structured response and applies its own routing rules."""

import json
from unittest.mock import Mock

import pytest

from src.routing.classifier import IntentClassifier
from src.routing.models import Intent, RouteTarget
from src.routing.router import IntentRouter


def _router(payload):
    client = Mock()
    client.chat.return_value = json.dumps(payload, ensure_ascii=False)
    return IntentRouter(IntentClassifier(client=client)), client.chat


def test_rag_uses_rewrite_from_classifier_without_second_call():
    query = "请问星云X9耳机没有降噪时能用几小时？"
    rewrite = "星云X9耳机关闭降噪时的续航时长"
    router, chat = _router(
        {
            "intent": "product_consult",
            "confidence": 0.94,
            "entities": {"product_name": "星云X9耳机"},
            "rewritten_query": rewrite,
        }
    )
    result = router.route(query)
    chat.assert_called_once()
    assert chat.call_args.kwargs["messages"][-1]["content"] == query
    assert result.target == RouteTarget.RAG
    assert result.query == query and result.rewritten_query == rewrite
    assert result.intent_result.entities == {"product_name": "星云X9耳机"}


@pytest.mark.parametrize("rewrite", [None, [], " ", "第一行\n第二行", "字" * 101])
def test_invalid_optional_rewrite_preserves_classification_and_uses_original(rewrite):
    router, chat = _router(
        {
            "intent": "logistics",
            "confidence": 0.88,
            "rewritten_query": rewrite,
            "entities": [],
        }
    )
    result = router.route("寄东西时能指定快递公司吗？")
    chat.assert_called_once()
    assert result.intent_result.intent == Intent.LOGISTICS
    assert result.target == RouteTarget.RAG
    assert result.rewritten_query == result.query


@pytest.mark.parametrize(
    "confidence,entity,expected",
    [
        (0.9, "ORD20260701001", RouteTarget.SQL),
        (0.2, "ORD20260701001", RouteTarget.RAG),
        (0.9, "ORD20260701002", RouteTarget.RAG),
        (0.9, "ORD2026070100", RouteTarget.RAG),
    ],
)
def test_order_route_needs_grounded_full_id_and_confidence(
    confidence, entity, expected
):
    router, chat = _router(
        {
            "intent": "order_query",
            "confidence": confidence,
            "entities": {"order_id": entity},
            "rewritten_query": "订单发货状态",
        }
    )
    result = router.route("订单 ord20260701001 发货了没？")
    chat.assert_called_once()
    assert result.target == expected
    if expected == RouteTarget.SQL:
        assert result.rewritten_query == result.query
    if entity != "ORD20260701001":
        assert not result.intent_result.entities


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"intent": "unrecognized", "confidence": 0.9},
        {"intent": "order_query", "confidence": True},
        {"intent": "order_query", "confidence": float("nan")},
        {"intent": "order_query", "confidence": 1.2},
    ],
)
def test_malformed_classification_falls_back_without_an_extra_call(payload):
    router, chat = _router(payload)
    result = router.route("请问快递什么时候到？")
    chat.assert_called_once()
    assert result.intent_result.intent == Intent.LOGISTICS
    assert result.target == RouteTarget.RAG
    assert result.rewritten_query == result.query


def test_explicit_after_sales_rule_still_bypasses_model():
    router, chat = _router({})
    result = router.route("帮我退款")
    chat.assert_not_called()
    assert result.intent_result.intent == Intent.RETURN_REFUND
    assert result.rewritten_query == result.query
