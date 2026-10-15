"""管理员界面 — 人工介入管理 + 系统监控

功能（5 个 Tab）：
- 人工介入管理：查看待处理请求、标记处理结果（自动刷新）
- 人工对话：转人工后同窗口与客户实时对话
- 系统监控：查询量/成本/幻觉率 + 最近查询 + 健康检查/清缓存
- 知识库管理：查看已入库文档（块数/类型/权限）+ 删除文档
- 历史记录：最近处理记录

启动: python -m src.admin.admin_ui
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import gradio as gr
import requests

# API 地址与登录鉴权同源：src.admin.api_auth 会先把 .env 注入 os.environ，
# 因此这里能取到本地 .env 的API_BASE_URL。若在本模块用 os.getenv 直接取，
# 会早于 .env 加载而落到 Docker 服务名 http://rag-api:8000，本地运行时请求全部失败。
from src.admin.api_auth import API_BASE_URL

logger = logging.getLogger(__name__)

# 人工介入数据库
HITL_DB = Path(__file__).parent.parent.parent / "data" / "hitl_requests.db"


def init_hitl_db():
    """初始化人工介入数据库"""
    HITL_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HITL_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hitl_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT DEFAULT '',
                user_query TEXT NOT NULL,
                intent TEXT DEFAULT '',
                confidence REAL DEFAULT 0,
                human_reason TEXT DEFAULT '',
                priority TEXT DEFAULT 'low',
                status TEXT DEFAULT 'pending',
                assigned_to TEXT DEFAULT '',
                resolution TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                resolved_at TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS human_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def get_pending_requests() -> list[dict]:
    """获取待处理的人工介入请求"""
    with sqlite3.connect(HITL_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""SELECT * FROM hitl_requests
               WHERE status = 'pending'
               ORDER BY
                 CASE priority
                   WHEN 'high' THEN 1
                   WHEN 'medium' THEN 2
                   WHEN 'low' THEN 3
                 END,
                 created_at DESC""").fetchall()
        return [dict(r) for r in rows]


def get_all_requests(limit: int = 50) -> list[dict]:
    """获取所有请求"""
    with sqlite3.connect(HITL_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM hitl_requests ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def add_hitl_request(
    session_id: str,
    user_query: str,
    intent: str,
    confidence: float,
    human_reason: str,
    priority: str,
):
    """添加人工介入请求"""
    with sqlite3.connect(HITL_DB) as conn:
        conn.execute(
            """INSERT INTO hitl_requests
               (session_id, user_query, intent, confidence, human_reason, priority, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                user_query,
                intent,
                confidence,
                human_reason,
                priority,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()


def resolve_request(request_id: int, resolution: str, assigned_to: str = ""):
    """处理人工介入请求"""
    with sqlite3.connect(HITL_DB) as conn:
        conn.execute(
            """UPDATE hitl_requests
               SET status = 'resolved',
                   resolution = ?,
                   assigned_to = ?,
                   resolved_at = ?
               WHERE id = ?""",
            (resolution, assigned_to, datetime.now().isoformat(), request_id),
        )
        conn.commit()


def get_human_conversation(session_id: str) -> list[dict]:
    """获取某会话的人工对话记录"""
    with sqlite3.connect(HITL_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM human_messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def send_human_reply(session_id: str, content: str) -> None:
    """客服回复客户（保存到人工对话，客户端轮询推送）"""
    with sqlite3.connect(HITL_DB) as conn:
        conn.execute(
            "INSERT INTO human_messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, "assistant", content, datetime.now().isoformat()),
        )
        conn.commit()


def format_conversation(messages: list[dict]) -> str:
    """格式化对话记录"""
    if not messages:
        return "暂无对话"
    lines = []
    for m in messages:
        who = "👤 客户" if m["role"] == "user" else "🎧 客服"
        lines.append(f"**{who}**: {m['content']}")
    return "\n\n".join(lines)


def get_system_stats() -> dict:
    """获取系统统计"""
    try:
        from src.admin.api_auth import auth_headers

        resp = requests.get(
            f"{API_BASE_URL}/api/stats/daily",
            headers=auth_headers(),
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def get_documents() -> dict:
    """获取知识库文档清单（管理员）"""
    try:
        from src.admin.api_auth import auth_headers

        resp = requests.get(
            f"{API_BASE_URL}/api/documents",
            headers=auth_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {
            "error": str(e),
            "total_documents": 0,
            "total_vectors": 0,
            "documents": [],
        }


def delete_document(source_file: str) -> dict:
    """删除某文档的全部向量（管理员）"""
    from urllib.parse import quote

    try:
        from src.admin.api_auth import auth_headers

        resp = requests.delete(
            f"{API_BASE_URL}/api/documents/{quote(source_file, safe='/')}",
            headers=auth_headers(),
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def get_recent_queries(limit: int = 20) -> list[dict]:
    """获取最近查询记录（管理员）"""
    try:
        from src.admin.api_auth import auth_headers

        resp = requests.get(
            f"{API_BASE_URL}/api/stats/recent",
            params={"limit": limit},
            headers=auth_headers(),
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def format_documents_table(docs: list[dict]) -> str:
    """格式化知识库文档表格"""
    if not docs:
        return "知识库为空。请先到入库平台（7860）添加文档。"
    lines = ["| 文档 | 块数 | 类型 | 权限 |", "|---|---|---|---|"]
    for d in docs:
        lines.append(
            f"| {d.get('source_file', '')} | {d.get('chunks', 0)} | "
            f"{d.get('doc_type', '')} | {d.get('access_level_label', '')} |"
        )
    return "\n".join(lines)


def format_recent_table(rows: list[dict]) -> str:
    """格式化最近查询表格"""
    if not rows:
        return "暂无查询记录"
    lines = ["| 时间 | 问题 | 意图 | 延迟 | 成本(元) |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {str(r.get('timestamp', ''))[:19]} | "
            f"{str(r.get('user_query', ''))[:24]} | {r.get('intent', '')} | "
            f"{round(r.get('total_time_ms', 0) or 0)} ms | "
            f"{round(r.get('cost', 0) or 0, 4)} |"
        )
    return "\n".join(lines)


def format_requests_table(requests: list[dict]) -> str:
    """格式化请求表格"""
    if not requests:
        return "暂无待处理请求"

    lines = [
        "| ID | 优先级 | 用户问题 | 意图 | 原因 | 状态 | 创建时间 |",
        "|---|--------|----------|------|------|------|----------|",
    ]

    for r in requests:
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
            r["priority"], "⚪"
        )
        lines.append(
            f"| {r['id']} | {priority_icon} {r['priority']} | {r['user_query'][:30]}... | "
            f"{r['intent']} | {r['human_reason'][:20]}... | {r['status']} | "
            f"{r['created_at'][:16]} |"
        )

    return "\n".join(lines)


def create_admin_ui() -> gr.Blocks:
    """创建管理员界面"""

    init_hitl_db()

    with gr.Blocks(
        title="管理员控制台",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# 🎛️ 管理员控制台")

        with gr.Tabs():
            # Tab 1: 人工介入管理
            with gr.Tab("📋 人工介入管理"):
                gr.Markdown("### 待处理请求")

                refresh_btn = gr.Button("🔄 刷新", variant="secondary")
                requests_display = gr.Markdown(
                    value=format_requests_table(get_pending_requests())
                )

                gr.Markdown("### 处理请求")

                with gr.Row():
                    request_id_input = gr.Number(
                        label="请求 ID",
                        precision=0,
                    )
                    resolution_input = gr.Textbox(
                        label="处理结果",
                        placeholder="输入处理结果...",
                    )
                    assigned_input = gr.Textbox(
                        label="处理人",
                        placeholder="客服姓名",
                    )

                resolve_btn = gr.Button("✅ 标记已处理", variant="primary")
                resolve_status = gr.Textbox(label="状态", interactive=False)

                def refresh_requests():
                    return format_requests_table(get_pending_requests())

                def resolve(request_id, resolution, assigned):
                    if not request_id:
                        return "请输入请求 ID"
                    if not resolution:
                        return "请输入处理结果"
                    resolve_request(int(request_id), resolution, assigned)
                    return f"✅ 请求 {request_id} 已处理"

                refresh_btn.click(
                    refresh_requests,
                    outputs=[requests_display],
                )

                resolve_btn.click(
                    resolve,
                    inputs=[request_id_input, resolution_input, assigned_input],
                    outputs=[resolve_status],
                ).then(
                    refresh_requests,
                    outputs=[requests_display],
                )

                # 自动刷新: 每 10 秒刷新待处理列表（新工单自动出现，无需手动点刷新）
                pending_timer = gr.Timer(10)
                pending_timer.tick(refresh_requests, outputs=[requests_display])

            # Tab 2: 人工对话（转人工后同窗口与客户对话）
            with gr.Tab("💬 人工对话"):
                gr.Markdown("### 与客户实时对话（转人工后同一窗口）")
                gr.Markdown(
                    "客户转人工后，消息会进入这里。客服回复后，客户侧自动收到。"
                )

                conv_refresh_btn = gr.Button("🔄 刷新会话列表", variant="secondary")
                conv_session = gr.Dropdown(
                    label="选择会话", choices=[], interactive=True
                )
                conv_state = gr.State("")
                conv_display = gr.Markdown("选择会话查看对话")

                with gr.Row():
                    conv_reply = gr.Textbox(
                        label="客服回复", placeholder="输入回复内容...", scale=4
                    )
                    conv_send = gr.Button("📤 发送给客户", variant="primary", scale=1)
                conv_status = gr.Textbox(label="状态", interactive=False)

                def load_sessions():
                    reqs = get_pending_requests()
                    choices = [
                        (f"#{r['id']} {r['user_query'][:20]}", r["session_id"])
                        for r in reqs
                    ]
                    val = choices[0][1] if choices else None
                    return gr.Dropdown(choices=choices, value=val, interactive=True)

                def show_conv(session_id):
                    conv_state.value = session_id or ""
                    if not session_id:
                        return "选择会话查看对话"
                    return format_conversation(get_human_conversation(session_id))

                def send_reply(content):
                    sid = conv_state.value
                    if not sid:
                        return "请先选择会话", gr.update(), ""
                    if not content.strip():
                        return "请输入回复内容", gr.update(), content
                    send_human_reply(sid, content.strip())
                    return (
                        f"✅ 已发送给客户: {content.strip()[:30]}",
                        format_conversation(get_human_conversation(sid)),
                        "",  # 清空输入框
                    )

                def refresh_conv():
                    """自动刷新当前会话的对话（客服界面动态显示新消息）"""
                    sid = conv_state.value
                    if not sid:
                        return gr.update()
                    return format_conversation(get_human_conversation(sid))

                conv_refresh_btn.click(load_sessions, outputs=[conv_session])
                conv_session.change(
                    show_conv, inputs=[conv_session], outputs=[conv_display]
                )
                conv_send.click(
                    send_reply,
                    inputs=[conv_reply],
                    outputs=[conv_status, conv_display, conv_reply],
                )

                # 自动刷新: 每 5 秒更新当前会话的对话（新客户消息自动出现）
                conv_timer = gr.Timer(5)
                conv_timer.tick(refresh_conv, outputs=[conv_display])

            # Tab 3: 系统监控
            with gr.Tab("📊 系统监控"):
                gr.Markdown("### 系统状态")

                stats_refresh_btn = gr.Button("🔄 刷新统计", variant="secondary")
                stats_display = gr.JSON(label="系统统计")

                gr.Markdown("### 最近查询")
                recent_refresh_btn = gr.Button("🔄 刷新最近查询", variant="secondary")
                recent_display = gr.Markdown()

                gr.Markdown("### 快捷操作")

                with gr.Row():
                    health_btn = gr.Button("检查健康状态", variant="secondary")
                    cache_btn = gr.Button("清空缓存", variant="secondary")

                action_result = gr.Textbox(label="操作结果", interactive=False)

                def refresh_stats():
                    return get_system_stats()

                def refresh_recent():
                    return format_recent_table(get_recent_queries(20))

                def check_health():
                    try:
                        from src.admin.api_auth import auth_headers

                        resp = requests.get(
                            f"{API_BASE_URL}/api/health",
                            headers=auth_headers(),
                            timeout=5,
                        )
                        resp.raise_for_status()
                        return json.dumps(resp.json(), indent=2, ensure_ascii=False)
                    except Exception as e:
                        return f"错误: {str(e)}"

                def clear_cache():
                    try:
                        from src.admin.api_auth import auth_headers

                        resp = requests.delete(
                            f"{API_BASE_URL}/api/cache",
                            headers=auth_headers(),
                            timeout=5,
                        )
                        resp.raise_for_status()
                        return "✅ 缓存已清空"
                    except Exception as e:
                        return f"错误: {str(e)}"

                stats_refresh_btn.click(
                    refresh_stats,
                    outputs=[stats_display],
                )

                recent_refresh_btn.click(
                    refresh_recent,
                    outputs=[recent_display],
                )

                health_btn.click(
                    check_health,
                    outputs=[action_result],
                )

                cache_btn.click(
                    clear_cache,
                    outputs=[action_result],
                )

            # Tab 4: 知识库管理
            with gr.Tab("📚 知识库管理"):
                gr.Markdown("### 知识库规模")
                kb_summary = gr.Markdown("点击下方「刷新知识库」加载")
                kb_refresh_btn = gr.Button("🔄 刷新知识库", variant="secondary")
                kb_table = gr.Markdown()

                gr.Markdown(
                    "### 删除文档\n"
                    "删除会移除该文档的全部向量，不可撤销。"
                    "重新入库请到入库平台 7860。"
                )
                kb_dropdown = gr.Dropdown(
                    label="选择要删除的文档", choices=[], interactive=True
                )
                kb_confirm = gr.Checkbox(
                    label="我确认删除该文档的全部向量", value=False
                )
                kb_delete_btn = gr.Button("🗑️ 删除文档", variant="stop")
                kb_status = gr.Textbox(label="操作结果", interactive=False)

                def refresh_kb():
                    data = get_documents()
                    if data.get("error"):
                        return (
                            f"❌ 加载失败: {data['error']}",
                            "",
                            gr.Dropdown(choices=[], value=None, interactive=True),
                        )
                    docs = data.get("documents", [])
                    summary = (
                        f"**已入库文档 {data.get('total_documents', 0)} 份，"
                        f"共 {data.get('total_vectors', 0)} 个向量**"
                    )
                    choices = [d["source_file"] for d in docs]
                    return (
                        summary,
                        format_documents_table(docs),
                        gr.Dropdown(choices=choices, value=None, interactive=True),
                    )

                def do_delete(source_file, confirmed):
                    if not source_file:
                        return "请先选择要删除的文档"
                    if not confirmed:
                        return "请先勾选确认框"
                    res = delete_document(source_file)
                    if res.get("error"):
                        return f"❌ 删除失败: {res['error']}"
                    return f"✅ 已删除 {source_file}（{res.get('deleted', 0)} 个向量）"

                kb_refresh_btn.click(
                    refresh_kb,
                    outputs=[kb_summary, kb_table, kb_dropdown],
                )
                kb_delete_btn.click(
                    do_delete,
                    inputs=[kb_dropdown, kb_confirm],
                    outputs=[kb_status],
                ).then(
                    refresh_kb,
                    outputs=[kb_summary, kb_table, kb_dropdown],
                )

            # Tab 5: 历史记录
            with gr.Tab("📜 历史记录"):
                gr.Markdown("### 最近处理记录")

                history_refresh_btn = gr.Button("🔄 刷新", variant="secondary")
                history_display = gr.Markdown()

                def refresh_history():
                    requests = get_all_requests(limit=20)
                    if not requests:
                        return "暂无记录"

                    lines = [
                        "| ID | 问题 | 意图 | 优先级 | 状态 | 处理人 | 处理结果 |",
                        "|---|------|------|--------|------|--------|----------|",
                    ]

                    for r in requests:
                        status_icon = "✅" if r["status"] == "resolved" else "⏳"
                        lines.append(
                            f"| {r['id']} | {r['user_query'][:20]}... | {r['intent']} | "
                            f"{r['priority']} | {status_icon} {r['status']} | "
                            f"{r['assigned_to'] or '-'} | {r['resolution'][:20] or '-'}... |"
                        )

                    return "\n".join(lines)

                history_refresh_btn.click(
                    refresh_history,
                    outputs=[history_display],
                )

    return app


def main():
    """启动管理员界面"""
    from src.admin.api_auth import gradio_auth

    app = create_admin_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        auth=gradio_auth(),
    )


if __name__ == "__main__":
    main()
