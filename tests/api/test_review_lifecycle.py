"""Real SQLite checkpoints + graph execution; external models are controlled."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import chat_service, review_service
from src.api.auth import CurrentUser
from src.api.deps import get_current_user
from src.api.routers import chat, reviews, stream
from src.config import settings
from src.conversation.review_store import ReviewStore, get_review_store
from src.conversation.session_manager import SessionManager
from src.graph import workflow


@pytest.fixture
def env(monkeypatch, tmp_path):
    counts = {"generate": 0, "rewrite": 0, "always_bad": False}
    user = [CurrentUser("alice", "member", "member", 7)]
    manager = SessionManager(tmp_path / "history.db")

    monkeypatch.setattr(
        workflow,
        "classify_intent",
        lambda state: {
            "intent": "product_consult",
            "target": "rag",
            "confidence": 1.0,
            "rewritten_query": state["query"],
            "entities": {},
        },
    )
    monkeypatch.setattr(
        workflow, "check_human_needed", lambda state: {"needs_human": False}
    )
    monkeypatch.setattr(
        workflow,
        "retrieve_docs",
        lambda state: {
            "retrieved_docs": [
                {
                    "chunk_id": "c",
                    "text": "礼盒说明",
                    "score": 0.9,
                    "source_file": "kb.md",
                }
            ],
            "degradation_level": 1,
            "degradation_method": "dense",
        },
    )

    def generate(state):
        counts["generate"] += 1
        return {
            "answer": f"待审核草稿{counts['generate']}",
            "faithfulness": 0.1
            if counts["always_bad"] or counts["generate"] == 1
            else 1.0,
        }

    def rewrite(state):
        counts["rewrite"] += 1
        return {"loop_count": state["loop_count"] + 1, "human_decision": ""}

    monkeypatch.setattr(workflow, "generate_answer", generate)
    monkeypatch.setattr(workflow, "rewrite_and_retrieve", rewrite)
    graphs = []

    def restart():
        if graphs:
            graphs[-1].close()
        graph = workflow.RAGWorkflow(
            workflow.build_checkpointer(settings.workflow_checkpoint_path)
        )
        graphs.append(graph)
        return graph

    restart()
    monkeypatch.setattr(chat_service, "get_workflow", lambda: graphs[-1])
    monkeypatch.setattr(chat_service, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        chat_service,
        "get_session_memory",
        lambda sid: SimpleNamespace(
            build_context=lambda q: "礼盒上下文",
            record_turn=lambda *args: None,
        ),
    )
    monkeypatch.setattr(
        "src.engineering.get_monitor",
        lambda: SimpleNamespace(record=lambda record: None),
    )
    app = FastAPI()
    for router in (chat.router, stream.router, reviews.router):
        app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user[0]
    with TestClient(app) as tc:
        result = tc.post(
            "/api/chat", json={"query": "礼盒的问题", "session_id": "s1"}
        ).json()
        assert result["review_task_id"] and result["reply"] == chat_service.REVIEW_REPLY
        data = SimpleNamespace(
            tc=tc,
            app=app,
            user=user,
            graphs=graphs,
            counts=counts,
            manager=manager,
            task_id=result["review_task_id"],
            restart=restart,
        )
        yield data
    graphs[-1].close()


def _admin(env):
    env.user[0] = CurrentUser("reviewer", "admin", "vip")


def _decision(env, decision="approve", task_id=None):
    return env.tc.post(
        f"/api/reviews/{task_id or env.task_id}/decision",
        json={"decision": decision, "note": "核验完成"},
    )


def test_owner_cannot_view_draft_or_approve_and_other_users_are_isolated(env):
    response = env.tc.get(f"/api/reviews/{env.task_id}")
    assert response.status_code == 200
    assert "草稿" not in response.text and "draft" not in response.json()
    assert _decision(env).status_code == 403
    env.user[0] = CurrentUser("bob", "member", "member", 8)
    assert env.tc.get(f"/api/reviews/{env.task_id}").status_code == 404
    assert env.tc.get("/api/reviews").json()["tasks"] == []
    env.user[0] = CurrentUser("alice", "normal", "public", 7)
    assert env.tc.get(f"/api/reviews/{env.task_id}").status_code == 404


def test_admin_sees_evidence_approves_once_and_owner_receives_final_result(env):
    _admin(env)
    draft = env.tc.get(f"/api/reviews/{env.task_id}").json()
    assert (
        draft["draft"] == "待审核草稿1"
        and draft["evidence"][0]["source_file"] == "kb.md"
    )
    approved = _decision(env)
    assert approved.status_code == 200, approved.text
    assert approved.json()["result"]["reply"] == "待审核草稿1"
    assert approved.json()["result"]["needs_human"] is False
    assert _decision(env).json() == approved.json()
    assert _decision(env, "reject").status_code == 409
    assert env.counts["generate"] == 1
    messages = env.manager.get_session("alice:s1").messages
    assert [m.content for m in messages] == [
        "礼盒的问题",
        chat_service.REVIEW_REPLY,
        "待审核草稿1",
    ]
    env.user[0] = CurrentUser("alice", "member", "member", 7)
    owner = env.tc.get("/api/reviews", params={"session_id": "s1"}).json()["tasks"][0]
    assert owner["result"]["reply"] == "待审核草稿1" and "draft" not in owner
    assert len(env.manager.get_session("alice:s1").messages) == 3


def test_pending_chat_does_not_replace_checkpoint_or_duplicate_task(env):
    before = env.graphs[-1].snapshot("alice:s1")
    response = env.tc.post(
        "/api/chat", json={"query": "另一个问题", "session_id": "s1"}
    ).json()
    assert response["review_task_id"] == env.task_id
    after = env.graphs[-1].snapshot("alice:s1")
    assert after.config == before.config and after.values["query"] == "礼盒的问题"
    assert env.counts["generate"] == 1
    assert len(get_review_store().list()) == 1


def test_next_worker_refreshes_cached_history_after_approval(env, monkeypatch):
    other_worker = SessionManager(env.manager._db_path)
    assert len(other_worker.get_session("alice:s1").messages) == 2
    _admin(env)
    assert _decision(env).status_code == 200
    monkeypatch.setattr(chat_service, "get_session_manager", lambda: other_worker)
    env.user[0] = CurrentUser("alice", "member", "member", 7)
    result = env.tc.post("/api/chat", json={"query": "继续问", "session_id": "s1"})
    assert result.status_code == 200
    history = env.graphs[-1].snapshot("alice:s1").values["history"]
    assert any(message["content"] == "待审核草稿1" for message in history)


def test_restart_loads_real_sqlite_checkpoint_and_can_approve(env):
    script = """import json,sys
from pathlib import Path
from src.graph.workflow import RAGWorkflow,build_checkpointer
graph=RAGWorkflow(build_checkpointer(Path(sys.argv[1])))
snapshot=graph.snapshot('alice:s1')
print(json.dumps({'next':snapshot.next,'answer':snapshot.values['answer']}))
graph.close()
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(settings.workflow_checkpoint_path)],
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == {
        "next": ["human_approval"],
        "answer": "待审核草稿1",
    }
    env.restart()
    _admin(env)
    assert _decision(env).json()["result"]["reply"] == "待审核草稿1"
    assert env.counts["generate"] == 1


def test_production_factory_reopens_persistent_checkpoint(env):
    from langgraph.checkpoint.sqlite import SqliteSaver

    workflow.close_workflow()
    try:
        graph = workflow.get_workflow()
        assert isinstance(graph.app.checkpointer, SqliteSaver)
        checkpoint_id = graph.snapshot("alice:s1").config["configurable"][
            "checkpoint_id"
        ]
        workflow.close_workflow()
        reopened = workflow.get_workflow()
        assert reopened.pending_review("alice:s1")
        assert (
            reopened.snapshot("alice:s1").config["configurable"]["checkpoint_id"]
            == checkpoint_id
        )
    finally:
        workflow.close_workflow()


def test_reject_rewrites_and_resumes_original_graph(env):
    _admin(env)
    result = _decision(env, "reject")
    assert result.status_code == 200, result.text
    assert result.json()["result"]["reply"] == "待审核草稿2"
    assert env.counts["rewrite"] == 1 and env.counts["generate"] == 2
    assert _decision(env, "reject").status_code == 200
    assert (
        env.counts["rewrite"] == 1
        and len(env.manager.get_session("alice:s1").messages) == 3
    )


def test_reject_to_new_review_cannot_reuse_old_decision(env):
    env.counts["always_bad"] = True
    _admin(env)
    old = _decision(env, "reject").json()
    next_id = old["result"]["review_task_id"]
    assert next_id and next_id != env.task_id
    assert "草稿2" not in old["result"]["reply"]
    assert len(env.manager.get_session("alice:s1").messages) == 2
    assert _decision(env, "reject").json()["result"]["review_task_id"] == next_id
    assert _decision(env, "approve").status_code == 409
    new = _decision(env, "approve", next_id)
    assert new.status_code == 200 and new.json()["result"]["reply"] == "待审核草稿2"


def test_crash_after_graph_completion_recovers_without_rerunning(env, monkeypatch):
    _admin(env)
    original = ReviewStore.complete

    def crash(*args):
        raise RuntimeError("simulated crash before result save")

    monkeypatch.setattr(ReviewStore, "complete", crash)
    assert _decision(env).status_code == 503
    assert get_review_store().get(env.task_id)["status"] == "processing"
    assert env.graphs[-1].snapshot("alice:s1").next == ()
    before = env.graphs[-1].snapshot("alice:s1").config
    env.user[0] = CurrentUser("alice", "member", "member", 7)
    waiting = env.tc.post(
        "/api/chat", json={"query": "新问题", "session_id": "s1"}
    ).json()
    assert waiting["review_task_id"] == env.task_id
    assert env.graphs[-1].snapshot("alice:s1").config == before
    env.restart()
    _admin(env)
    monkeypatch.setattr(ReviewStore, "complete", original)
    assert _decision(env).json()["result"]["reply"] == "待审核草稿1"
    assert (
        env.counts["generate"] == 1
        and len(env.manager.get_session("alice:s1").messages) == 3
    )


def test_crash_after_decision_before_graph_update_can_retry(env, monkeypatch):
    _admin(env)
    monkeypatch.setattr(
        env.graphs[-1].app,
        "update_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    assert _decision(env).status_code == 503
    env.restart()
    assert _decision(env).status_code == 200
    assert env.counts["generate"] == 1


def test_crash_after_reject_repause_does_not_reject_new_draft_again(env, monkeypatch):
    env.counts["always_bad"] = True
    _admin(env)
    original = ReviewStore.complete
    monkeypatch.setattr(
        ReviewStore,
        "complete",
        lambda *args: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    assert _decision(env, "reject").status_code == 503
    assert env.counts["generate"] == 2
    assert (
        len(get_review_store().list()) == 1
    )  # successor is not published before parent completion
    env.restart()
    monkeypatch.setattr(ReviewStore, "complete", original)
    result = _decision(env, "reject")
    assert result.status_code == 200 and result.json()["result"]["review_task_id"]
    assert env.counts["generate"] == 2 and env.counts["rewrite"] == 1


def test_outbox_retries_delivery_without_duplicate_history(env, monkeypatch):
    _admin(env)
    original = ReviewStore.mark_delivered
    monkeypatch.setattr(
        ReviewStore,
        "mark_delivered",
        lambda *args: (_ for _ in ()).throw(RuntimeError("crash after insert")),
    )
    assert _decision(env).status_code == 503
    assert len(env.manager.get_session("alice:s1").messages) == 3
    monkeypatch.setattr(ReviewStore, "mark_delivered", original)
    env.restart()
    assert _decision(env).status_code == 200
    assert len(env.manager.get_session("alice:s1").messages) == 3
    assert get_review_store().get(env.task_id)["delivered_at"]


def test_old_snapshot_cannot_be_approved(env):
    env.graphs[-1].app.update_state(
        {"configurable": {"thread_id": "alice:s1"}}, {"answer": "new draft"}
    )
    _admin(env)
    assert _decision(env).status_code == 409


def test_outbox_can_recover_delivery_via_owner_poll(env, monkeypatch):
    _admin(env)
    original = env.manager.add_message_once
    monkeypatch.setattr(
        env.manager,
        "add_message_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("delivery unavailable")
        ),
    )
    assert _decision(env).status_code == 503
    assert get_review_store().get(env.task_id)["status"] == "resolved"
    assert len(env.manager.get_session("alice:s1").messages) == 2
    monkeypatch.setattr(env.manager, "add_message_once", original)
    env.restart()
    env.user[0] = CurrentUser("alice", "member", "member", 7)
    assert (
        env.tc.get(f"/api/reviews/{env.task_id}").json()["result"]["reply"]
        == "待审核草稿1"
    )
    assert len(env.manager.get_session("alice:s1").messages) == 3


def test_os_lock_is_shared_with_other_processes():
    import hashlib

    from src.engineering.session_lock import session_guard

    path = settings.workflow_lock_dir / (
        hashlib.sha256(b"alice:lock").hexdigest() + ".lock"
    )
    script = """from filelock import FileLock,Timeout
import sys
try:
 with FileLock(sys.argv[1],timeout=0): print('acquired')
except Timeout: print('blocked')
"""

    def probe():
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    with session_guard("alice:lock"):
        assert probe() == "blocked"
    assert probe() == "acquired"


def test_simultaneous_admin_retries_execute_once(env):
    admin = CurrentUser("reviewer", "admin", "vip")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(review_service.decide, env.task_id, "approve", "checked", admin)
            for _ in range(2)
        ]
        results = [future.result(timeout=10) for future in futures]
    assert results[0] == results[1]
    assert (
        env.counts["generate"] == 1
        and len(env.manager.get_session("alice:s1").messages) == 3
    )


def test_unauthenticated_and_invalid_decisions_are_rejected(env):
    _admin(env)
    assert _decision(env, "invalid").status_code == 422
    env.app.dependency_overrides.clear()
    assert env.tc.get("/api/reviews").status_code == 401
    assert _decision(env).status_code == 401
