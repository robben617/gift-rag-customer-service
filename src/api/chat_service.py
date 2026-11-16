"""Shared chat application service: workflow, verified reply, memory and monitoring."""

import asyncio
import logging
import time

from src.access import is_admin
from src.api.auth import CurrentUser
from src.api.models import ChatRequest, ChatResponse, SearchResultItem
from src.config import settings
from src.conversation import Message, get_session_manager, get_session_memory
from src.engineering.pii_redactor import get_pii_redactor
from src.engineering.session_lock import session_guard
from src.graph import get_workflow

logger = logging.getLogger(__name__)

ERROR_REPLY = "系统暂时繁忙，请稍后重试。"
REVIEW_REPLY = "回答需人工核验，正在为您转接人工客服。"


async def execute_chat(req: ChatRequest, current_user: CurrentUser) -> ChatResponse:
    """Keep synchronous model, graph and SQLite calls off the event loop."""
    return await asyncio.to_thread(run_chat, req, current_user)


def run_chat(req: ChatRequest, current_user: CurrentUser) -> ChatResponse:
    """Serialize turns within one user/session, across both transport adapters."""
    sid = f"{current_user.username}:{req.session_id}"
    with session_guard(sid):
        try:
            return _run_chat(req, current_user)
        except Exception:
            logger.exception("对话服务执行失败: session=%s", sid)
            # Initialization failures must not echo raw PII or internal exceptions.
            return _build_response({"answer": ERROR_REPLY}, "")


def _finalize_result(result: dict, safe_query: str, memory_context: str) -> dict:
    """Only release/store a final safe answer; keep unapproved drafts in the graph."""
    result = dict(result)
    result["query"] = safe_query
    needs_review = "__interrupt__" in result or result.get("evaluation_passed") is False
    if needs_review:
        result.update(
            answer=REVIEW_REPLY,
            needs_human=True,
            human_reason=result.get("human_reason") or "回答质量评估未通过，需人工核验",
            human_priority=result.get("human_priority") or "medium",
        )
        if not result.get("handoff_payload"):
            result["handoff_payload"] = {
                "user_query": safe_query,
                "intent": result.get("intent", ""),
                "emotion": result.get("emotion", "calm"),
                "entities": result.get("entities", {}),
                "memory_summary": memory_context[:500],
                "attempted_actions": (
                    f"degradation_level={result.get('degradation_level', 1)}, "
                    f"correction_rounds={result.get('correction_rounds', 0)}"
                ),
                "unresolved_reason": result["human_reason"],
            }
    elif result.get("error"):
        result["answer"] = ERROR_REPLY
    return result


def _run_chat(req: ChatRequest, current_user: CurrentUser) -> ChatResponse:
    # ── 模块33: 会话按用户隔离（用户 A 永远只能触达 "{username}:" 前缀的 key）──
    started = time.perf_counter()
    sid = f"{current_user.username}:{req.session_id}"

    # ── 第0步: PII 脱敏 ──
    redactor = get_pii_redactor()
    safe_query, pii_found = redactor.redact(req.query)
    if pii_found:
        logger.warning(
            "PII 脱敏: 输入包含 %d 项敏感信息 (%s)",
            len(pii_found),
            ", ".join(f["type"] for f in pii_found),
        )

    workflow = get_workflow()

    from src.api.review_service import flush_resolved, register_pending

    flush_resolved(sid)
    pending = register_pending(workflow, current_user, req.session_id)
    if pending:
        result = _finalize_result({"__interrupt__": True}, safe_query, "")
        result.update(
            answer="上一条回答仍在人工核验，请待处理后重试本次问题。",
            review_task_id=pending["task_id"],
        )
        return _build_response(result, safe_query)

    # 重置 LLM token 累加器，统计本次请求真实用量（而非字符数估算）
    from src.engineering.llm_client import get_llm_client

    get_llm_client().reset_usage()

    # ── 多轮对话: 加载会话历史（指代消解用）──
    session_manager = get_session_manager()
    session = session_manager.refresh_session(sid)
    if not session:
        session = session_manager.create_session(sid)
    history = [{"role": m.role, "content": m.content} for m in session.messages[-6:]]

    # 三层记忆: 组装"最小有用上下文"（实体ledger/滚动摘要/历史片段）
    memory_context = get_session_memory(sid).build_context(safe_query)

    try:
        result = workflow.run(
            query=safe_query,
            session_id=sid,
            top_k=req.top_k,
            use_reranker=req.use_reranker,
            history=history,
            memory_context=memory_context,
            access_level=current_user.access_level,
            # 修复(审查): 订单查询按用户隔离，防枚举单号越权
            user_id=current_user.seed_user_id,
            is_admin=is_admin(current_user.role),
        )
    except Exception:
        logger.exception("对话工作流执行失败: query=%s", safe_query[:100])
        result = {
            "query": safe_query,
            "answer": "系统暂时繁忙，请稍后重试。",
            "error": "workflow_execution_failed",
        }

    pending = register_pending(workflow, current_user, req.session_id)
    if pending:
        result["review_task_id"] = pending["task_id"]
    result = _finalize_result(result, safe_query, memory_context)

    # ── 多轮对话: 保存本轮消息到会话历史（用命名空间 sid，按用户隔离）──
    try:
        session_manager.add_message(
            sid,
            Message(
                role="user",
                content=safe_query,
                intent=result.get("intent", ""),
            ),
        )
        session_manager.add_message(
            sid,
            Message(
                role="assistant",
                content=result.get("answer", ""),
                sources=[
                    doc.get("source_file", "")
                    for doc in result.get("retrieved_docs", [])
                ],
            ),
        )
    except Exception:
        logger.warning("保存会话历史失败: %s", sid)

    # 三层记忆: 记录本轮（抽实体 + 更新滚动摘要）
    try:
        get_session_memory(sid).record_turn(safe_query, result.get("answer", ""))
    except Exception:
        logger.warning("记录会话记忆失败: %s", sid)

    # 成本监控: 记录本次查询（token 数按长度粗略估算，成本用官方价格折算）
    try:
        from src.engineering import get_monitor
        from src.engineering.llm_client import get_llm_client
        from src.engineering.monitor import QueryRecord, estimate_cost

        _answer = result.get("answer", "")
        _usage = get_llm_client().total_usage
        _prompt_tokens = _usage.get("prompt_tokens") or max(1, len(safe_query))
        _completion_tokens = _usage.get("completion_tokens") or max(
            1, len(_answer) // 2
        )
        _total_tokens = _usage.get("total_tokens") or (
            _prompt_tokens + _completion_tokens
        )
        get_monitor().record(
            QueryRecord(
                user_query=safe_query,
                intent=result.get("intent", ""),
                retrieval_method=result.get("degradation_method", "dense"),
                degradation_level=result.get("degradation_level", 1),
                retrieval_docs_count=len(result.get("retrieved_docs", [])),
                retrieval_time_ms=result.get("search_time_ms", 0),
                hallucination_detected=result.get("was_corrected", False),
                self_correction_rounds=result.get("correction_rounds", 0),
                faithfulness=result.get("faithfulness", 0),
                prompt_tokens=_prompt_tokens,
                completion_tokens=_completion_tokens,
                total_tokens=_total_tokens,
                llm_cost=estimate_cost(
                    settings.default_model, _prompt_tokens, _completion_tokens
                ),
                total_time_ms=(time.perf_counter() - started) * 1000,
                final_answer_length=len(_answer),
                session_id=sid,
                model_used=settings.default_model,
            )
        )
    except Exception as e:
        logger.warning("成本监控记录失败: %s", e)

    return _build_response(result, safe_query)


def _build_response(result: dict, safe_query: str) -> ChatResponse:
    # 统一映射结果，两种传输方式使用完全相同的业务字段。
    # 修复(审查): 评估不通过(忠实度低/检索降级/诚实兜底)应上抛人工核验标志，
    # 否则"无法确认"类低质量答案被静默返回，绕过"高敏承诺需 ≥0.85 否则转人工"的护栏。
    needs_human = result.get("needs_human", False)
    human_reason = result.get("human_reason", "")
    if not needs_human and result.get("evaluation_passed") is False:
        needs_human = True
        human_reason = human_reason or "回答质量评估未通过，建议人工核验"

    # 构建响应
    response = ChatResponse(
        query=result.get("query", safe_query),
        intent=result.get("intent", ""),
        confidence=result.get("confidence", 0.0),
        target=result.get("target", ""),
        rewritten_query=result.get("rewritten_query", safe_query),
        reasoning=result.get("reasoning", ""),
        results=[
            SearchResultItem(
                chunk_id=doc.get("chunk_id", ""),
                text=doc.get("text", ""),
                score=doc.get("score", 0.0),
                doc_type=doc.get("doc_type", ""),
                source_file=doc.get("source_file", ""),
            )
            for doc in result.get("retrieved_docs", [])
        ],
        reply=result.get("answer", ""),
        search_time_ms=result.get("search_time_ms", 0),
        degradation_level=result.get("degradation_level", 1),
        degradation_method=result.get("degradation_method", "dense"),
        faithfulness=result.get("faithfulness", 0.0),
        correction_rounds=result.get("correction_rounds", 0),
        was_corrected=result.get("was_corrected", False),
        needs_human=needs_human,
        human_reason=human_reason,
        human_priority=result.get("human_priority", ""),
        emotion=result.get("emotion", "calm"),
        handoff_payload=result.get("handoff_payload", {}),
        review_task_id=result.get("review_task_id", ""),
        policy_status=result.get("policy_status", ""),
        policy_missing_fields=result.get("policy_missing_fields", []),
        policy_version=result.get("policy_version", ""),
        evidence_status=result.get("evidence_status", ""),
        answer_status=result.get("answer_status", ""),
    )

    return response
