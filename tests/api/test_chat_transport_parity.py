"""JSON/SSE business parity, verified output and real graph branch coverage."""

import asyncio
import copy
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import chat_service
from src.api.auth import CurrentUser
from src.api.deps import get_current_user
from src.api.models import ChatRequest
from src.api.routers import chat, stream
from src.conversation.session_manager import SessionManager
from src.engineering.llm_client import LLMClient


def _events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture
def environment(monkeypatch, tmp_path):
    manager = SessionManager(tmp_path / "sessions.db")
    memories = {}
    records = []
    calls = []
    result = {
        "intent": "product_consult",
        "target": "rag",
        "confidence": 0.9,
        "rewritten_query": "商品材质",
        "answer": "商品为纯棉，详见说明。",
        "faithfulness": 1.0,
        "evaluation_passed": True,
        "retrieved_docs": [
            {
                "chunk_id": "faq-1",
                "text": "纯棉",
                "score": 0.9,
                "source_file": "policy.md",
            }
        ],
    }

    def memory(sid):
        if sid not in memories:
            turns = []
            memories[sid] = SimpleNamespace(
                turns=turns,
                build_context=lambda query: "此前已确认商品为礼盒",
                record_turn=lambda *args: turns.append(args),
            )
        return memories[sid]

    def run(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(result)

    client = LLMClient()
    monkeypatch.setattr(chat_service, "get_workflow", lambda: SimpleNamespace(run=run))
    monkeypatch.setattr(chat_service, "get_session_manager", lambda: manager)
    monkeypatch.setattr(chat_service, "get_session_memory", memory)
    monkeypatch.setattr(
        "src.engineering.get_monitor", lambda: SimpleNamespace(record=records.append)
    )
    monkeypatch.setattr("src.engineering.llm_client.get_llm_client", lambda: client)
    user = CurrentUser("alice", "member", "member", seed_user_id=7)
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(stream.router)
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as tc:
        yield SimpleNamespace(
            tc=tc,
            app=app,
            manager=manager,
            memories=memories,
            calls=calls,
            result=result,
            records=records,
            user=user,
        )


@pytest.mark.parametrize(
    "scenario", ["rag", "policy", "sql", "human", "rejected", "paused", "error"]
)
def test_transports_return_same_business_result_and_persist_once(environment, scenario):
    env = environment
    if scenario in ("policy", "sql"):
        env.result.update(
            target=scenario,
            retrieved_docs=[],
            answer="工具已查询到结果",
            evaluation_passed=True,
        )
    elif scenario == "human":
        env.result.update(
            target="human",
            answer="正在转接人工",
            needs_human=True,
            human_reason="退款申请",
            human_priority="high",
            emotion="angry",
            handoff_payload={
                "user_query": "申请退款",
                "entities": {"order_id": "ORD1"},
            },
        )
    elif scenario == "rejected":
        env.result.update(
            answer="保证退款100元", evaluation_passed=False, faithfulness=0.2
        )
    elif scenario == "paused":
        env.result.update(answer="尚未批准的承诺", __interrupt__=True)
    elif scenario == "error":
        env.result.update(answer="内部异常包含敏感数据", error="private exception")

    request = {
        "query": "查询商品",
        "session_id": "same",
        "top_k": 3,
        "use_reranker": False,
    }
    regular = env.tc.post("/api/chat", json=request)
    streaming = env.tc.post("/api/chat/stream", params=request)
    assert regular.status_code == streaming.status_code == 200
    events = _events(streaming)
    result = next(event["data"] for event in events if event["event"] == "result")
    assert result == regular.json()
    assert (
        "".join(event["data"] for event in events if event["event"] == "token")
        == result["reply"]
    )
    assert events[0]["event"] == "status"
    assert events[-1] == {"event": "done", "data": "[DONE]"}
    assert len([event for event in events if event["event"] == "done"]) == 1
    assert streaming.headers["content-type"].startswith("text/event-stream")
    assert streaming.headers["x-accel-buffering"] == "no"
    if scenario in ("rejected", "paused"):
        assert result["reply"] == chat_service.REVIEW_REPLY
        assert result["needs_human"] and result["handoff_payload"]
        assert env.result["answer"] not in streaming.text
    if scenario == "error":
        assert result["reply"] == chat_service.ERROR_REPLY
        assert "private exception" not in streaming.text
    assert len(env.calls) == 2
    for call in env.calls:
        assert call["session_id"] == "alice:same"
        assert call["access_level"] == "member" and call["user_id"] == 7
        assert call["is_admin"] is False
        assert call["top_k"] == 3 and call["use_reranker"] is False
    assert env.calls[1]["history"][-1]["content"] == result["reply"]
    assert len(env.manager.get_session("alice:same").messages) == 4
    assert len(env.memories["alice:same"].turns) == 2
    assert len(env.records) == 2


def test_pii_is_redacted_before_workflow_storage_and_sse(environment):
    env = environment
    query = "我的手机号13812345678，帮我查询"
    response = env.tc.post(
        "/api/chat/stream", params={"query": query, "session_id": "pii"}
    )
    assert "13812345678" not in response.text
    assert "13812345678" not in env.calls[0]["query"]
    messages = env.manager.get_session("alice:pii").messages
    assert len(messages) == 2
    assert "13812345678" not in messages[0].content
    assert "13812345678" not in env.records[0].user_query


def test_session_and_order_identity_come_from_authenticated_user(environment):
    env = environment
    env.tc.post("/api/chat", json={"query": "商品", "session_id": "shared"})
    env.app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        "bob", "admin", "vip", 9
    )
    env.tc.post("/api/chat/stream", params={"query": "商品", "session_id": "shared"})
    assert env.calls[1]["session_id"] == "bob:shared"
    assert env.calls[1]["history"] == []
    assert env.calls[1]["user_id"] == 9 and env.calls[1]["is_admin"] is True
    assert len(env.manager.get_session("alice:shared").messages) == 2
    assert len(env.manager.get_session("bob:shared").messages) == 2


@pytest.mark.parametrize("query", ["", "x" * 1001])
def test_stream_validates_query_before_workflow(environment, query):
    env = environment
    assert env.tc.post("/api/chat/stream", params={"query": query}).status_code == 422
    assert env.calls == []


def test_both_transports_require_authentication(environment):
    env = environment
    env.app.dependency_overrides.clear()
    assert env.tc.post("/api/chat", json={"query": "商品"}).status_code == 401
    assert env.tc.post("/api/chat/stream", params={"query": "商品"}).status_code == 401
    assert env.calls == []


def test_unexpected_stream_adapter_error_is_generic_and_terminal(
    environment, monkeypatch
):
    async def broken(*args):
        raise RuntimeError("private-api-key")

    monkeypatch.setattr(chat_service, "execute_chat", broken)
    response = environment.tc.post("/api/chat/stream", params={"query": "商品"})
    events = _events(response)
    assert [event["event"] for event in events] == ["status", "error", "done"]
    assert "private-api-key" not in response.text


@pytest.mark.asyncio
async def test_sse_only_status_and_heartbeat_before_verified_reply(
    environment, monkeypatch
):
    ready = asyncio.Event()

    async def delayed(*args):
        await ready.wait()
        return chat_service._build_response({"answer": "核验完成"}, "商品")

    monkeypatch.setattr(chat_service, "execute_chat", delayed)
    monkeypatch.setattr(stream, "HEARTBEAT_SECONDS", 0.01)
    response = await stream.chat_stream("商品", "sse", 5, True, environment.user)
    iterator = response.body_iterator
    assert '"status"' in await anext(iterator)
    assert await anext(iterator) == ": keep-alive\n\n"
    ready.set()
    remainder = [part async for part in iterator]
    assert any('"token"' in part and "核验完成" in part for part in remainder)


@pytest.mark.asyncio
async def test_closing_stream_cancels_waiting_adapter_task(environment, monkeypatch):
    cancelled = asyncio.Event()

    async def waiting(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(chat_service, "execute_chat", waiting)
    monkeypatch.setattr(stream, "HEARTBEAT_SECONDS", 0.01)
    response = await stream.chat_stream("商品", "cancel", 5, True, environment.user)
    iterator = response.body_iterator
    await anext(iterator)
    assert await anext(iterator) == ": keep-alive\n\n"
    await iterator.aclose()
    assert cancelled.is_set()


def test_real_graph_token_usage_reaches_request_monitor(environment, monkeypatch):
    from src.engineering.llm_client import get_llm_client
    from src.graph import workflow

    def classify(state):
        get_llm_client()._record_usage(
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        )
        return {
            "intent": "chitchat",
            "target": "direct",
            "confidence": 1.0,
            "rewritten_query": state["query"],
            "entities": {},
        }

    monkeypatch.setattr(workflow, "classify_intent", classify)
    monkeypatch.setattr(
        workflow, "check_human_needed", lambda state: {"needs_human": False}
    )
    graph = workflow.RAGWorkflow()
    monkeypatch.setattr(chat_service, "get_workflow", lambda: graph)
    assert environment.tc.post("/api/chat", json={"query": "你好"}).status_code == 200
    assert environment.records[-1].total_tokens == 12


@pytest.mark.asyncio
async def test_sync_workflow_does_not_block_event_loop(environment, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def blocking(req, user):
        entered.set()
        assert release.wait(3)
        return chat_service._build_response({"answer": "完成"}, req.query)

    monkeypatch.setattr(chat_service, "run_chat", blocking)
    task = asyncio.create_task(
        chat_service.execute_chat(ChatRequest(query="商品"), environment.user)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert not task.done()
    finally:
        release.set()
    assert (await task).reply == "完成"


def test_same_session_is_serialized_but_other_sessions_can_progress(
    environment, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Event()
    release = threading.Event()
    other_entered = threading.Event()
    first_completed = threading.Event()

    def worker(req, user):
        if req.query == "first":
            entered.set()
            assert release.wait(3)
            first_completed.set()
        elif req.query == "second":
            assert first_completed.is_set()
        else:
            other_entered.set()
        return chat_service._build_response({"answer": req.query}, req.query)

    monkeypatch.setattr(chat_service, "_run_chat", worker)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(
            chat_service.run_chat,
            ChatRequest(query="first", session_id="same"),
            environment.user,
        )
        try:
            assert entered.wait(1)
            second = pool.submit(
                chat_service.run_chat,
                ChatRequest(query="second", session_id="same"),
                environment.user,
            )
            other = pool.submit(
                chat_service.run_chat,
                ChatRequest(query="other", session_id="different"),
                environment.user,
            )
            assert other_entered.wait(1)
            assert other.result(timeout=1).reply == "other"
        finally:
            release.set()
        assert first.result(timeout=2).reply == "first"
        assert second.result(timeout=2).reply == "second"


@pytest.mark.parametrize("branch", ["policy", "sql", "rag", "reject", "human"])
def test_real_langgraph_branches_have_transport_parity(
    environment, monkeypatch, branch
):
    from src.graph import workflow

    env = environment
    query = "退货运费谁出" if branch == "policy" else "查询商品或订单"
    tool_calls = []
    from src.generation.answerability import Assessment, evidence_input

    def assess(query, docs, *, answer=None):
        # Hold coverage fixed to exercise transport parity and the independent
        # faithfulness gate (including a deliberately low-score generated draft).
        return Assessment(
            "complete",
            "controlled",
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

    def classify(state):
        return {
            "intent": "return_refund"
            if branch == "policy"
            else "order_query"
            if branch == "sql"
            else "product_consult",
            "target": "sql" if branch == "sql" else "rag",
            "confidence": 0.99,
            "rewritten_query": state["query"],
            "entities": {"order_id": "ORD1"},
        }

    monkeypatch.setattr(workflow, "classify_intent", classify)
    monkeypatch.setattr(
        workflow,
        "check_human_needed",
        lambda state: {
            "needs_human": branch == "human",
            "emotion": "angry" if branch == "human" else "calm",
            "human_reason": "需要人工" if branch == "human" else "",
            "human_priority": "high",
        },
    )
    monkeypatch.setattr(
        workflow,
        "retrieve_docs",
        lambda state: {
            "retrieved_docs": [
                {
                    "chunk_id": "c",
                    "text": "商品纯棉",
                    "score": 0.9,
                    "source_file": "kb.md",
                }
            ],
            "degradation_level": 1,
            "degradation_method": "dense",
            "search_time_ms": 1,
        },
    )
    monkeypatch.setattr("src.embedding.retriever.get_retriever", lambda: None)
    monkeypatch.setattr(
        workflow,
        "get_corrector",
        lambda retriever: SimpleNamespace(
            generate_with_docs=lambda **kwargs: SimpleNamespace(
                answer="保证退款100元" if branch == "reject" else "商品纯棉",
                faithfulness=0.1 if branch == "reject" else 1.0,
                correction_rounds=0,
                was_corrected=False,
            ),
        ),
    )

    def order_reply(*args, **kwargs):
        tool_calls.append((args, kwargs))
        return "订单已发货", True

    monkeypatch.setattr(
        "src.orders.order_service.get_order_service",
        lambda: SimpleNamespace(reply_for=order_reply),
    )
    graph = workflow.RAGWorkflow()
    monkeypatch.setattr(chat_service, "get_workflow", lambda: graph)
    regular = env.tc.post(
        "/api/chat", json={"query": query, "session_id": "json"}
    ).json()
    response = env.tc.post(
        "/api/chat/stream", params={"query": query, "session_id": "sse"}
    )
    streamed = next(
        event["data"] for event in _events(response) if event["event"] == "result"
    )
    if branch == "reject":
        assert streamed["review_task_id"] and regular["review_task_id"]
        assert streamed["review_task_id"] != regular["review_task_id"]
        assert {k: v for k, v in streamed.items() if k != "review_task_id"} == {
            k: v for k, v in regular.items() if k != "review_task_id"
        }
    else:
        assert streamed == regular
    if branch == "policy":
        assert "商家承担" in streamed["reply"]
    elif branch == "sql":
        assert streamed["reply"] == "订单已发货"
        assert len(tool_calls) == 2
        assert all(
            kwargs == {"user_id": 7, "is_admin": False} for _, kwargs in tool_calls
        )
    elif branch == "reject":
        assert streamed["reply"] == chat_service.REVIEW_REPLY
        assert "保证退款100元" not in response.text
        assert graph.app.get_state(
            {"configurable": {"thread_id": "alice:sse"}}
        ).next == ("human_approval",)
        assert (
            graph.app.get_state({"configurable": {"thread_id": "alice:sse"}}).values[
                "answer"
            ]
            == "保证退款100元"
        )
    elif branch == "human":
        assert streamed["needs_human"] and streamed["handoff_payload"]
    else:
        assert streamed["reply"] == "商品纯棉" and not streamed["needs_human"]
