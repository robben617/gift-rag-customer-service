"""用户聊天界面 — Gradio ChatBot

面向终端用户的客服聊天界面，支持：
- 多轮对话
- 会话级隔离（每浏览器独立，不串话）
- 人工介入状态提示，客服处理结果与回复自动推送
- 自动保存人工介入请求到数据库

状态栏展示内容由「调试模式」决定：
- 关闭（默认）：只显示对客户有意义的服务状态
- 开启：额外显示意图、置信度、检索条数与来源，用于演示与排障
初始值取自环境变量 UI_DEBUG_MODE，界面上可临时切换。

启动: python -m src.admin.chat_ui
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
import requests

# API 地址与鉴权同源：src.admin.api_auth 会先把 .env 注入 os.environ，
# 因此这里的 API_BASE_URL 才能取到本地 .env 的配置。
# 若在此处直接 os.getenv 取，会早于 .env 加载而落到 Docker 服务名 http://rag-api:8000，
# 本地运行时所有请求都会连接失败（仅 Docker 内才有该服务名）。
from src.admin.api_auth import API_BASE_URL, auth_headers

logger = logging.getLogger(__name__)

# 调试模式：默认关闭。
# 关闭时界面只显示对客户有意义的服务状态；开启后额外显示意图、置信度、命中条数与检索来源，
# 用于演示与排障。面向真实客户应保持关闭——内部字段对客户是噪音，也会暴露实现细节。
# 由环境变量 UI_DEBUG_MODE 提供初始值（见 .env.example），界面上可临时切换。
DEBUG_MODE = os.getenv("UI_DEBUG_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}

# 人工介入数据库
HITL_DB = Path(__file__).parent.parent.parent / "data" / "hitl_requests.db"

# 每浏览器会话独立：会话 ID + 展示游标。
# 修复(审查): 原实现把 SESSION_ID 作为模块级常量，同一 Gradio 进程的所有用户共享同一会话，
# 会互相串话、共享人工介入状态。改为会话级隔离——session_id 由 gr.State 承载，
# 展示游标按 session_id 分开存放。
_MAX_TRACKED_SESSIONS = 1000
_SESSION_TRACKER: dict[str, dict] = {}


def _new_session_id() -> str:
    """生成一个新的会话 ID（页面加载时调用一次）"""
    return f"chat_{uuid.uuid4().hex[:10]}"


def _tracker(session_id: str) -> dict:
    """取得（或初始化）某会话的展示游标状态"""
    if (
        session_id not in _SESSION_TRACKER
        and len(_SESSION_TRACKER) >= _MAX_TRACKED_SESSIONS
    ):
        _SESSION_TRACKER.clear()  # 上限保护：避免长期运行无限增长
    return _SESSION_TRACKER.setdefault(
        session_id,
        {"shown_resolutions": set(), "last_admin_msg_id": 0},
    )


def is_human_mode(session_id: str) -> bool:
    """该会话是否已转人工（存在待处理请求）"""
    try:
        with sqlite3.connect(HITL_DB) as conn:
            row = conn.execute(
                "SELECT 1 FROM hitl_requests WHERE session_id=? AND status='pending' LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None
    except Exception:
        return False


def save_human_message(session_id: str, role: str, content: str) -> None:
    """保存人工对话消息（role: user=客户, assistant=客服）"""
    try:
        with sqlite3.connect(HITL_DB) as conn:
            conn.execute(
                "INSERT INTO human_messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, role, content, datetime.now().isoformat()),
            )
            conn.commit()
    except Exception as e:
        logger.error("保存人工消息失败: %s", e)


def get_human_assistant_messages(session_id: str, since_id: int = 0) -> list[tuple]:
    """获取管理员发给该会话的新消息（供主动推送）"""
    try:
        with sqlite3.connect(HITL_DB) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, content FROM human_messages
                   WHERE session_id=? AND role='assistant' AND id>? ORDER BY id""",
                (session_id, since_id),
            ).fetchall()
        return [(r["id"], r["content"]) for r in rows]
    except Exception:
        return []


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


def save_hitl_request(
    session_id: str,
    user_query: str,
    intent: str,
    confidence: float,
    human_reason: str,
    priority: str,
):
    """保存人工介入请求到数据库"""
    try:
        # PII 脱敏：工单库不存真实敏感信息
        from src.engineering.pii_redactor import redact_text

        safe_query, _ = redact_text(user_query or "")
        with sqlite3.connect(HITL_DB) as conn:
            conn.execute(
                """INSERT INTO hitl_requests
                   (session_id, user_query, intent, confidence, human_reason, priority, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    safe_query,
                    intent,
                    confidence,
                    human_reason,
                    priority,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            logger.info("人工介入请求已保存: %s", safe_query[:50])
    except Exception as e:
        logger.error("保存人工介入请求失败: %s", e)


def get_resolution_for_session(session_id: str) -> tuple[int, str]:
    """获取该会话已处理的人工介入结果（修复: 管理员处理后回传客户）

    Returns:
        (request_id, resolution) — 无则 ("", "")
    """
    try:
        with sqlite3.connect(HITL_DB) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT id, resolution FROM hitl_requests
                   WHERE session_id = ? AND status = 'resolved'
                     AND resolution != '' AND resolution IS NOT NULL
                   ORDER BY resolved_at DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
        if row and row["resolution"]:
            return row["id"], row["resolution"]
    except Exception as e:
        logger.error("获取处理结果失败: %s", e)
    return "", ""


# 初始化数据库
init_hitl_db()


def build_status_message(
    *,
    debug: bool,
    intent: str = "",
    confidence: float = 0.0,
    needs_human: bool = False,
    human_reason: str = "",
    result_count: int = 0,
    sources: list[str] | None = None,
) -> str:
    """构造状态栏文案。

    调试模式（debug=True）：展示意图、置信度、命中条数与来源，便于演示与排障。
    正常模式（debug=False）：只保留对客户有意义的服务状态，不暴露内部字段。
    两种模式都保留"已转人工"这一对客户必要的信息——它决定用户接下来该不该继续等。
    来源按文件名排序输出，保证同一批结果每次展示顺序一致。

    Args:
        debug: 是否调试模式
        intent: 识别到的意图
        confidence: 意图置信度
        needs_human: 是否需要人工介入
        human_reason: 转人工原因（内部字段，仅调试模式展示）
        result_count: 检索命中条数
        sources: 检索来源文件名列表

    Returns:
        状态栏展示文案
    """
    if not debug:
        if needs_human:
            return "📞 已转接人工客服，客服回复后会显示在这里"
        return "✅ 已回复，可继续提问"

    parts: list[str] = []
    if intent:
        parts.append(f"意图: {intent} ({confidence:.0%})")
    if needs_human:
        parts.append(f"⚠️ 需要人工介入: {human_reason}")
    if result_count:
        parts.append(f"检索到 {result_count} 条结果")
        if sources:
            parts.append(f"来源: {', '.join(sorted(sources))}")
    return " | ".join(parts) if parts else "正常回复"


def chat_with_bot(
    message: str, history: list, session_id: str, debug: bool = False
) -> tuple:
    """与客服机器人对话

    Args:
        message: 用户输入
        history: 对话历史 (Gradio 6.0 格式)
        session_id: 当前浏览器会话 ID（每会话独立）
        debug: 调试模式；关闭时状态栏不展示意图/置信度/检索来源等内部字段

    Returns:
        (更新后的历史, 状态信息)
    """
    if not message.strip():
        return history, "请输入问题"

    tracker = _tracker(session_id)

    # 修复: 先检查管理员是否已处理该会话的人工介入 → 回传给客户（形成闭环）
    res_id, resolution = get_resolution_for_session(session_id)
    admin_reply = ""
    if res_id and res_id not in tracker["shown_resolutions"]:
        tracker["shown_resolutions"].add(res_id)
        admin_reply = f"【客服处理结果】{resolution}\n\n"

    # 已转人工 → 客户消息直接进人工对话（同一窗口），不走 AI
    if is_human_mode(session_id):
        save_human_message(session_id, "user", message)
        history.append({"role": "user", "content": message})
        history.append(
            {
                "role": "assistant",
                "content": "📞 已转人工，客服正在处理，请稍候。您的消息已同步给客服。",
            }
        )
        return history, "🔴 人工模式：消息已转人工客服，等待回复中..."

    try:
        # 调用 API（模块33: 携带 JWT；auth_headers 已在模块顶部导入）
        resp = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={
                "query": message,
                "top_k": 5,
                "use_reranker": True,
                "session_id": session_id,
            },
            headers=auth_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # 提取结果
        reply = data.get("reply", "抱歉，暂时无法回答。")
        intent = data.get("intent", "")
        confidence = data.get("confidence", 0)
        needs_human = data.get("needs_human", False)
        human_reason = data.get("human_reason", "")
        results = data.get("results", [])

        # 构建状态信息（按调试模式决定是否暴露内部字段）
        sources = sorted(
            {r.get("source_file", "") for r in results if r.get("source_file")}
        )
        status = build_status_message(
            debug=debug,
            intent=intent,
            confidence=confidence,
            needs_human=needs_human,
            human_reason=human_reason,
            result_count=len(results),
            sources=sources,
        )

        # 如果需要人工介入，保存到数据库（用稳定会话 ID 关联）
        if needs_human:
            save_hitl_request(
                session_id=session_id,
                user_query=message,
                intent=intent,
                confidence=confidence,
                human_reason=human_reason,
                priority=data.get("human_priority", "medium"),
            )
            status += "\n✅ 已通知管理员，客服处理后将回复您"

        # 管理员处理结果前置（如已处理）
        final_reply = admin_reply + reply

        # Gradio 6.0 格式: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": final_reply})

        return history, status

    except requests.exceptions.Timeout:
        logger.warning("聊天请求超时: %s", message[:50])
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "响应超时，请稍后重试。"})
        return history, "⚠️ 响应超时，请稍后重试"
    except Exception:
        logger.exception("聊天请求失败")
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "系统暂时繁忙，请稍后重试。"})
        return history, "系统暂时繁忙，请稍后重试。"


def clear_history(session_id: str):
    """清空对话历史（同时重置本会话的展示游标）"""
    _SESSION_TRACKER.pop(session_id, None)
    return [], "对话已清空"


def poll_resolution(history: list, session_id: str):
    """主动推送客服处理结果 + 人工对话新回复（无需客户再发消息）

    由 gr.Timer 定时调用：管理员处理完/回复后 → 自动加进对话。
    """
    tracker = _tracker(session_id)
    changed = False

    # 一次性处理结果
    res_id, resolution = get_resolution_for_session(session_id)
    if res_id and res_id not in tracker["shown_resolutions"]:
        tracker["shown_resolutions"].add(res_id)
        history = list(history) + [
            {"role": "assistant", "content": f"【客服处理结果】{resolution}"}
        ]
        changed = True

    # 人工对话多轮回复（客服在管理员界面回复 → 主动推给客户）
    new_msgs = get_human_assistant_messages(
        session_id, tracker["last_admin_msg_id"]
    )
    for mid, content in new_msgs:
        history = list(history) + [
            {"role": "assistant", "content": f"【人工客服】{content}"}
        ]
        tracker["last_admin_msg_id"] = mid
        changed = True

    if changed:
        return history, "📥 客服回复已送达"
    return history, gr.update()


def create_chat_ui() -> gr.Blocks:
    """创建聊天界面"""

    with gr.Blocks(title="电商智能客服") as app:
        gr.Markdown("""
            # 🤖 电商智能客服

            欢迎使用电商智能客服系统！请输入您的问题，我会尽力为您解答。

            **我可以帮您：**
            - 商品咨询、退货退款、物流查询
            - 订单状态查询
            - 复杂或紧急问题转接人工客服
            """)

        # 技术实现说明：仅调试模式可见。面向真实客户时这些是实现细节，属噪音。
        tech_panel = gr.Markdown(
            """
            **技术实现（调试模式可见）：**
            - 6 类意图自动识别 + 条件路由
            - 双路召回 + CrossEncoder 精排，L0 至 L5 六级降级
            - 幻觉检测与自纠正，敏感问题强制人工审批
            """,
            visible=DEBUG_MODE,
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=400,
                    show_label=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入问题",
                        placeholder="请输入您的问题...",
                        show_label=False,
                        scale=4,
                        container=False,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("清空对话", variant="secondary")
                    retry_btn = gr.Button("重新回答", variant="secondary")

            with gr.Column(scale=1):
                status_display = gr.Textbox(
                    label="状态",
                    value="等待输入...",
                    interactive=False,
                    lines=3,
                )

                debug_toggle = gr.Checkbox(
                    label="调试模式",
                    value=DEBUG_MODE,
                    info="显示意图、置信度与检索来源，仅演示与排障使用",
                )

                gr.Markdown("""
                    ### 📋 快捷问题

                    - 怎么退货？
                    - 退款多久到账？
                    - 快递到哪了？
                    - 这个商品怎么样？
                    - 我要投诉
                    """)

                # 系统说明同属实现细节，跟随调试模式显隐
                sys_panel = gr.Markdown(
                    """
                    ### ℹ️ 系统说明

                    - 意图分类：6 类意图自动识别
                    - 检索策略：稠密向量召回 + CrossEncoder 精排
                    - 降级策略：L0 至 L5 六级降级
                    - 质量保证：幻觉检测 + 自纠正
                    """,
                    visible=DEBUG_MODE,
                )

        # 每浏览器会话独立的会话 ID（页面加载时生成，避免多用户串话）
        session_state = gr.State("")
        app.load(_new_session_id, outputs=session_state)

        # 调试开关切换 → 同步技术说明与系统说明的显隐
        debug_toggle.change(
            lambda on: (gr.update(visible=bool(on)), gr.update(visible=bool(on))),
            inputs=[debug_toggle],
            outputs=[tech_panel, sys_panel],
        )

        # 事件绑定
        def respond(message, history, session_id, debug):
            return chat_with_bot(message, history, session_id, debug)

        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot, session_state, debug_toggle],
            outputs=[chatbot, status_display],
        ).then(
            lambda: "",
            outputs=[msg_input],
        )

        submit_btn.click(
            respond,
            inputs=[msg_input, chatbot, session_state, debug_toggle],
            outputs=[chatbot, status_display],
        ).then(
            lambda: "",
            outputs=[msg_input],
        )

        clear_btn.click(
            clear_history,
            inputs=[session_state],
            outputs=[chatbot, status_display],
        )

        def retry_last(history, session_id, debug):
            if history:
                # 获取最后一条用户消息
                last_message = None
                for msg in reversed(history):
                    if msg.get("role") == "user":
                        last_message = msg.get("content")
                        break
                if last_message:
                    # 移除最后一轮对话
                    history = history[:-2]
                    return chat_with_bot(last_message, history, session_id, debug)
            return history, "没有可重试的消息"

        retry_btn.click(
            retry_last,
            inputs=[chatbot, session_state, debug_toggle],
            outputs=[chatbot, status_display],
        )

        # 主动推送: 每 5 秒检查管理员处理结果（无需客户再发消息）
        timer = gr.Timer(5)
        timer.tick(
            poll_resolution,
            inputs=[chatbot, session_state],
            outputs=[chatbot, status_display],
        )

    return app


def main():
    """启动聊天界面"""
    from src.admin.api_auth import gradio_auth

    app = create_chat_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        auth=gradio_auth(),
    )


if __name__ == "__main__":
    main()
