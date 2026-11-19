"""Shared JSON/SSE workflows for consultations, operations and clarification."""

import json
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import chat_service
from src.api.auth import CurrentUser
from src.api.deps import get_current_user
from src.api.routers import chat, stream
from src.business import return_policy
from src.conversation.session_manager import SessionManager
from src.graph import workflow
from src.orders.order_service import OrderInfo
from src.routing.router import IntentRouter


@pytest.fixture
def env(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit policy/action must not use generic LLM/RAG")

    router = IntentRouter(SimpleNamespace(classify=forbidden))
    monkeypatch.setattr(workflow, "get_router", lambda: router)
    monkeypatch.setattr(workflow, "retrieve_docs", forbidden)
    monkeypatch.setattr(return_policy, "today_shanghai", lambda: date(2026, 11, 16))
    graph = workflow.RAGWorkflow()
    manager = SessionManager(tmp_path / "sessions.db")
    calls = []
    record = {
        "order": OrderInfo(
            "ORD000001", "已签收", created_at="2026-11-01", signed_at="2026-11-12"
        ),
        "status": "found",
    }

    def lookup(order_id, **kwargs):
        calls.append((order_id, kwargs))
        return record["order"], record["status"]

    monkeypatch.setattr(
        "src.orders.order_service.get_order_service",
        lambda: SimpleNamespace(lookup_policy_order=lookup),
    )
    monkeypatch.setattr(chat_service, "get_workflow", lambda: graph)
    monkeypatch.setattr(chat_service, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        chat_service,
        "get_session_memory",
        lambda sid: SimpleNamespace(
            build_context=lambda query: "", record_turn=lambda *args: None
        ),
    )
    monkeypatch.setattr(
        "src.engineering.get_monitor",
        lambda: SimpleNamespace(record=lambda record: None),
    )
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(stream.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        "alice", "member", "member", 7
    )
    with TestClient(app) as tc:
        yield SimpleNamespace(
            tc=tc, graph=graph, calls=calls, record=record, manager=manager
        )


def _streamed(response):
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    return next(event["data"] for event in events if event["event"] == "result")


def test_independent_dialogue_suite_through_real_chat_graph(env, tmp_path):
    from pathlib import Path

    from src.evaluation.dialogue import evaluate_dialogues, load_dialogues

    cases = load_dialogues(
        Path(__file__).resolve().parents[2] / "data/dialogue_eval.json"
    )

    def send(query, session_id):
        response = env.tc.post(
            "/api/chat", json={"query": query, "session_id": session_id}
        )
        response.raise_for_status()
        return response.json()

    report = evaluate_dialogues(cases, send, execution="controlled_in_process")
    (tmp_path / "dialogue-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failures = [d for d in report["details"] if d["failures"]]
    assert report["complete"], failures
    assert not failures, failures


@pytest.mark.parametrize(
    "query,human,word",
    [
        ("退款一般多久到账？", False, "1-3个工作日"),
        ("怎么申请退款？", False, "提交审核"),
        ("帮我退款", True, "尚未办理"),
        ("退款还没到账", True, "支付记录"),
    ],
)
def test_consultations_and_actions_have_transport_parity(env, query, human, word):
    regular = env.tc.post(
        "/api/chat", json={"query": query, "session_id": "json"}
    ).json()
    streamed = _streamed(
        env.tc.post("/api/chat/stream", params={"query": query, "session_id": "sse"})
    )
    assert regular["trace_id"] and regular["trace_id"] != streamed["trace_id"]
    assert {k: v for k, v in regular.items() if k != "trace_id"} == {
        k: v for k, v in streamed.items() if k != "trace_id"
    }
    assert regular["needs_human"] is human and word in regular["reply"]
    assert env.calls == []
    if not human:
        assert regular["policy_status"] == "informational" and regular["policy_version"]


def test_clarification_facts_survive_multiple_graph_turns(env):
    def ask(query):
        return env.tc.post(
            "/api/chat", json={"query": query, "session_id": "dialog"}
        ).json()

    first = ask("这盒礼品能退吗")
    assert first["policy_status"] == "clarify" and not first["needs_human"]
    assert "category" in first["policy_missing_fields"]
    assert ask("签收5天了")["policy_status"] == "clarify"
    final = ask("普通衣服，没拆封，想无理由退货")
    assert final["policy_status"] == "conditional" and "5天" in final["reply"]
    assert env.calls == []


def test_clarification_context_reopens_from_sqlite_after_restart(env, monkeypatch):
    from src.config import settings

    graphs = [
        workflow.RAGWorkflow(
            workflow.build_checkpointer(settings.workflow_checkpoint_path)
        )
    ]
    monkeypatch.setattr(chat_service, "get_workflow", lambda: graphs[-1])
    try:
        first = env.tc.post("/api/chat", json={"query": "这件衣服能退吗"}).json()
        assert first["policy_status"] == "clarify"
        graphs[-1].close()
        graphs.append(
            workflow.RAGWorkflow(
                workflow.build_checkpointer(settings.workflow_checkpoint_path)
            )
        )
        new_manager = SessionManager(env.manager._db_path)
        monkeypatch.setattr(chat_service, "get_session_manager", lambda: new_manager)
        final = env.tc.post(
            "/api/chat", json={"query": "签收3天，没用过，无理由退货"}
        ).json()
        assert final["policy_status"] == "conditional" and "3天" in final["reply"]
    finally:
        graphs[-1].close()


def test_ambiguous_request_can_be_clarified_into_action(env):
    assert (
        env.tc.post("/api/chat", json={"query": "退款"}).json()["policy_status"]
        == "clarify"
    )
    final = env.tc.post("/api/chat", json={"query": "帮我办理"}).json()
    assert final["needs_human"] and "尚未办理退款" in final["reply"]


def test_clarification_to_action_keeps_structured_handoff_facts(env):
    first = env.tc.post("/api/chat", json={"query": "这件衣服签收3天能退吗"}).json()
    assert first["policy_status"] == "clarify"
    result = env.tc.post("/api/chat", json={"query": "那帮我办理吧"}).json()
    assert result["needs_human"] and "尚未办理退款" in result["reply"]
    assert result["handoff_payload"]["policy_context"]["category"] == "ordinary"


def test_order_signature_and_owner_override_user_date_claim(env):
    query = "订单ORD000001这件衣服签收1天，未使用，无理由能退吗"
    result = env.tc.post("/api/chat", json={"query": query}).json()
    assert result["policy_status"] == "conditional" and "4天" in result["reply"]
    assert env.calls == [("ORD000001", {"user_id": 7, "is_admin": False})]


def test_new_item_action_does_not_keep_old_policy_facts(env):
    env.tc.post("/api/chat", json={"query": "这件衣服签收3天能退吗"})
    result = env.tc.post("/api/chat", json={"query": "另一件商品帮我办理退款"}).json()
    assert result["needs_human"] and result["handoff_payload"]["policy_context"] == {}


@pytest.mark.parametrize(
    "status,expected,human",
    [("not_found", "clarify", False), ("unavailable", "manual_review", True)],
)
def test_missing_order_evidence_cannot_be_replaced_with_policy(
    env, status, expected, human
):
    env.record.update(order=None, status=status)
    result = env.tc.post("/api/chat", json={"query": "订单ORD000001能退吗"}).json()
    assert result["policy_status"] == expected and result["needs_human"] is human
    assert "处于这项通用规则" not in result["reply"]


def test_new_order_does_not_reuse_old_order_category(env):
    first = env.tc.post(
        "/api/chat", json={"query": "订单ORD000001普通衣服能退吗"}
    ).json()
    assert first["policy_status"] == "clarify"
    env.record["order"] = OrderInfo(
        "ORD000002", "已签收", created_at="2026-11-02", signed_at="2026-11-15"
    )
    second = env.tc.post("/api/chat", json={"query": "订单号ORD000002"}).json()
    assert (
        second["policy_status"] == "clarify"
        and "category" in second["policy_missing_fields"]
    )


def test_order_without_actual_signature_requires_human(env):
    env.record["order"].signed_at = ""
    result = env.tc.post(
        "/api/chat",
        json={"query": "订单ORD000001这件衣服签收1天，未使用，无理由能退吗"},
    ).json()
    assert result["needs_human"] and "实际签收时间" in result["reply"]
