"""UI adapters for backend-owned answer reviews; no local graph/DB writes."""

import re

import requests

from src.admin.api_auth import API_BASE_URL, auth_headers


def _request(method, path, **kwargs):
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}/api/reviews{path}",
            headers=auth_headers(),
            timeout=180 if method == "POST" else 10,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"error": "审批服务暂不可用或无权操作，请检查登录状态后重试"}


def pending_reviews():
    pending = _request("GET", "", params={"status": "pending"})
    processing = _request("GET", "", params={"status": "processing"})
    if "error" in pending or "error" in processing:
        return {"error": "暂时无法加载审批任务"}
    return {"tasks": pending.get("tasks", []) + processing.get("tasks", [])}


def inspect_review(task_id):
    if not re.fullmatch(r"[0-9a-f]{32}", (task_id or "").strip()):
        return {"error": "请输入有效的审批任务 ID"}
    return _request("GET", f"/{task_id.strip()}")


def submit_review(task_id, decision, note):
    if not re.fullmatch(r"[0-9a-f]{32}", (task_id or "").strip()) or decision not in (
        "approve",
        "reject",
    ):
        return {"error": "请填写任务 ID，并选择批准或拒绝"}
    return _request(
        "POST",
        f"/{task_id.strip()}/decision",
        json={"decision": decision, "note": note or ""},
    )


def session_reviews(session_id):
    return _request("GET", "", params={"session_id": session_id}).get("tasks", [])
