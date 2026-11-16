"""Review UI only calls authenticated APIs; drafts never enter owner rendering."""

from types import SimpleNamespace

from src.admin import admin_ui, chat_ui, review_api


def test_admin_decision_uses_authenticated_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        review_api, "auth_headers", lambda: {"Authorization": "Bearer test"}
    )

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"status": "resolved"}
        )

    monkeypatch.setattr(review_api.requests, "request", request)
    task_id = "a" * 32
    assert (
        review_api.submit_review(task_id, "approve", "checked")["status"] == "resolved"
    )
    assert calls[0][0] == "POST" and calls[0][1].endswith(
        f"/api/reviews/{task_id}/decision"
    )
    assert calls[0][2]["headers"]["Authorization"] == "Bearer test"
    assert calls[0][2]["json"] == {"decision": "approve", "note": "checked"}


def test_pending_list_filters_before_history_limit(monkeypatch):
    statuses = []

    def request(method, path, **kwargs):
        status = kwargs["params"]["status"]
        statuses.append(status)
        return {"tasks": [{"task_id": status, "status": status}]}

    monkeypatch.setattr(review_api, "_request", request)
    assert [row["task_id"] for row in review_api.pending_reviews()["tasks"]] == [
        "pending",
        "processing",
    ]
    assert statuses == ["pending", "processing"]


def test_review_result_is_appended_once_and_pending_successor_is_hidden(monkeypatch):
    session_id = "review-ui-test"
    chat_ui._SESSION_TRACKER.pop(session_id, None)
    monkeypatch.setattr(
        review_api,
        "session_reviews",
        lambda sid: [
            {
                "task_id": "second",
                "status": "resolved",
                "result": {"reply": "已核验回复", "review_task_id": ""},
            },
            {
                "task_id": "first",
                "status": "resolved",
                "result": {"reply": "待审核草稿", "review_task_id": "second"},
            },
        ],
    )
    history, changed = chat_ui._append_review_updates([], session_id)
    assert changed and history == [{"role": "assistant", "content": "已核验回复"}]
    again, changed = chat_ui._append_review_updates(history, session_id)
    assert not changed and again == history


def test_backend_review_does_not_create_duplicate_legacy_ticket(monkeypatch):
    monkeypatch.setattr(chat_ui, "get_resolution_for_session", lambda sid: ("", ""))
    monkeypatch.setattr(chat_ui, "is_human_mode", lambda sid: False)
    monkeypatch.setattr(review_api, "session_reviews", lambda sid: [])
    monkeypatch.setattr(chat_ui, "auth_headers", lambda: {})
    monkeypatch.setattr(
        chat_ui,
        "save_hitl_request",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("duplicate ticket")),
    )
    monkeypatch.setattr(
        chat_ui.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "reply": "等待核验",
                "needs_human": True,
                "review_task_id": "a" * 32,
            },
        ),
    )
    history, status = chat_ui.chat_with_bot("礼盒问题", [], "review-ticket-test")
    assert history[-1]["content"] == "等待核验" and "核验" in status


def test_admin_blocks_include_review_controls(monkeypatch):
    monkeypatch.setattr(admin_ui, "init_hitl_db", lambda: None)
    monkeypatch.setattr(admin_ui, "get_pending_requests", lambda: [])
    app = admin_ui.create_admin_ui()
    labels = {
        component.get("props", {}).get("label")
        for component in app.config["components"]
    }
    assert "📝 回答审批" in labels and "审批任务 ID" in labels


def test_terminal_review_fallback_creates_one_manual_ticket(monkeypatch, tmp_path):
    import sqlite3

    path = tmp_path / "manual.db"
    monkeypatch.setattr(chat_ui, "HITL_DB", path)
    chat_ui.init_hitl_db()
    for _ in range(2):
        chat_ui.save_hitl_request(
            "session",
            "人工继续处理",
            "complaint",
            1.0,
            "重试已耗尽",
            "high",
            review_task_id="task-1",
        )
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM hitl_requests").fetchone()[0] == 1
