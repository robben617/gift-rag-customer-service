"""Recoverable approval -> graph continuation -> idempotent result delivery."""

import json

from src.access import is_admin, parse_access_level
from src.api import chat_service
from src.api.auth import CurrentUser
from src.conversation import Message
from src.conversation.review_store import (
    ReviewConflict,
    ReviewNotFound,
    get_review_store,
)
from src.engineering.pii_redactor import redact_text
from src.engineering.session_lock import session_guard


def register_pending(workflow, user: CurrentUser, session_id: str):
    if not hasattr(workflow, "pending_review"):
        return None  # small injected test doubles / non-graph tools
    sid = f"{user.username}:{session_id}"
    store = get_review_store()
    processing = store.processing(sid)
    if processing:
        return processing  # do not overwrite completed/partial recovery checkpoints
    snapshot = workflow.pending_review(sid)
    if snapshot is None:
        return None
    if snapshot.values.get("session_id") != sid:
        raise ReviewConflict("审批会话状态不一致")
    return store.create(
        session_key=sid, owner=user.username, session_id=session_id, snapshot=snapshot
    )


def _authorize(row: dict, user: CurrentUser):
    if is_admin(user.role):
        return
    if row["owner"] != user.username or parse_access_level(
        user.access_level
    ) < parse_access_level(row["access_level"]):
        raise ReviewNotFound("审批任务不存在")


def _deliver(row: dict):
    if row["status"] != "resolved" or not row["result_json"] or row["delivered_at"]:
        return
    response = json.loads(row["result_json"])
    if response.get("review_task_id"):
        get_review_store().mark_delivered(row["task_id"])
        return  # re-paused; no unapproved draft or duplicate waiting message
    inserted = chat_service.get_session_manager().add_message_once(
        row["session_key"],
        Message(
            role="assistant",
            content=response["reply"],
            intent=response["intent"],
            sources=[item["source_file"] for item in response["results"]],
        ),
        event_id=f"review:{row['task_id']}",
    )
    if inserted:
        # History is durable; in-memory ledger enrichment is best effort.
        try:
            chat_service.get_session_memory(row["session_key"]).record_turn(
                "", response["reply"]
            )
        except Exception:
            chat_service.logger.exception("记录审批回复记忆失败")
    get_review_store().mark_delivered(row["task_id"])


def flush_resolved(session_key: str):
    store = get_review_store()
    for row in store.undelivered(session_key):
        _deliver(row)


def list_reviews(user: CurrentUser, session_id=None, status=None):
    store = get_review_store()
    rows = store.list(
        owner=None if is_admin(user.role) else user.username,
        session_id=session_id,
        status=status,
    )
    views = []
    for row in rows:
        try:
            _authorize(row, user)
        except ReviewNotFound:
            continue
        if row["status"] == "resolved" and not row["delivered_at"]:
            with session_guard(row["session_key"]):
                _deliver(row)
        views.append(store.view(row, admin=is_admin(user.role)))
    return views


def get_review(task_id: str, user: CurrentUser):
    store = get_review_store()
    row = store.get(task_id)
    _authorize(row, user)
    if row["status"] == "resolved" and not row["delivered_at"]:
        with session_guard(row["session_key"]):
            _deliver(row)
    return store.view(row, admin=is_admin(user.role))


def decide(task_id: str, decision: str, note: str, admin: CurrentUser):
    if not is_admin(admin.role):
        raise ReviewNotFound("审批任务不存在")
    if decision not in ("approve", "reject"):
        raise ReviewConflict("无效审批决定")
    store = get_review_store()
    row = store.get(task_id)
    with session_guard(row["session_key"]):
        row = store.get(task_id)
        if row["status"] == "superseded":
            raise ReviewConflict("审批任务已过期")
        if row["decision"] and row["decision"] != decision:
            raise ReviewConflict("该任务已有不同的审批决定")
        if row["status"] == "resolved":
            _deliver(row)
            return store.view(row, admin=True)
        workflow = chat_service.get_workflow()
        snapshot = workflow.snapshot(row["session_key"])
        config = {"configurable": {"thread_id": row["session_key"]}}
        state = snapshot.values
        current_id = (
            snapshot.config.get("configurable", {}).get("checkpoint_id")
            if snapshot.config
            else None
        )
        marked = state.get("approval_task_id") == task_id
        if not marked and (
            snapshot.next != ("human_approval",) or current_id != row["checkpoint_id"]
        ):
            raise ReviewConflict("审批快照已变化，不能批准旧版本")
        # Decision is committed before graph mutation; OS lock prevents another
        # live worker from claiming it. A crashed worker's same decision can retry.
        row = store.begin_decision(
            task_id, decision, admin.username, redact_text(note)[0]
        )
        if not marked:
            workflow.app.update_state(
                config,
                {
                    "approval_task_id": task_id,
                    "human_decision": "approve"
                    if decision == "approve"
                    else "rejected",
                },
            )
            snapshot = workflow.snapshot(row["session_key"])
        consumed = snapshot.values.get("approval_consumed_id") == task_id
        # A reject may already have reached the NEXT approval checkpoint before
        # a crash. Never apply the old decision to that newly generated draft.
        if snapshot.next and not (consumed and snapshot.next == ("human_approval",)):
            resumed = workflow.resume(config)
            if resumed.get("error"):
                raise RuntimeError("工作流续接失败，请重试同一审批决定")
        snapshot = workflow.snapshot(row["session_key"])
        result = dict(snapshot.values)
        next_task = None
        if snapshot.next:
            if snapshot.next != ("human_approval",):
                raise RuntimeError("工作流尚未完成，请重试同一审批决定")
            result["__interrupt__"] = True
            next_task = {
                "session_key": row["session_key"],
                "owner": row["owner"],
                "session_id": row["session_id"],
                "snapshot": snapshot,
            }
        safe = chat_service._finalize_result(
            result, result.get("query", ""), result.get("memory_context", "")
        )
        if next_task:
            safe["review_task_id"] = store.task_id_for(row["session_key"], snapshot)
        response = chat_service._build_response(safe, result.get("query", ""))
        row = store.complete(task_id, response.model_dump(mode="json"), next_task)
        _deliver(row)
        return store.view(row, admin=True)
