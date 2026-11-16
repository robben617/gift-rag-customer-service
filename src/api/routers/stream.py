"""POST /api/chat/stream — SSE transport for the shared, verified chat result.

No provisional model tokens are sent: the complete workflow, including quality
checks and any human interruption, finishes before answer chunks are emitted.
"""

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from src.api import chat_service
from src.api.auth import CurrentUser
from src.api.deps import get_current_user
from src.api.models import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["流式对话"])
HEARTBEAT_SECONDS = 10
ANSWER_CHUNK_SIZE = 32


def _event(name: str, data) -> str:
    # Preserve the existing data: {event, data} wire format and [DONE] sentinel.
    return f"data: {json.dumps({'event': name, 'data': data}, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    query: str = Query(..., min_length=1, max_length=1000),
    session_id: str = Query(default="default"),
    top_k: int = Query(default=5, ge=1, le=20),
    use_reranker: bool = Query(default=True),
    current_user: CurrentUser = Depends(get_current_user),
):
    """状态先行，完整核验后分段回复；query 参数保持原接口兼容。"""
    req = ChatRequest(
        query=query, session_id=session_id, top_k=top_k, use_reranker=use_reranker
    )

    async def event_generator():
        yield _event("status", "正在处理并核验回答...")
        task = asyncio.create_task(chat_service.execute_chat(req, current_user))
        try:
            while not task.done():
                completed, _ = await asyncio.wait({task}, timeout=HEARTBEAT_SECONDS)
                if not completed:
                    yield ": keep-alive\n\n"
            response = await task
            sources = sorted(
                {item.source_file for item in response.results if item.source_file}
            )
            yield _event("sources", sources[:3])
            yield _event("emotion", response.emotion)
            if response.needs_human:
                yield _event(
                    "handoff",
                    {
                        "needs_human": True,
                        "reason": response.human_reason,
                        "priority": response.human_priority,
                        "payload": response.handoff_payload,
                    },
                )
            for offset in range(0, len(response.reply), ANSWER_CHUNK_SIZE):
                yield _event(
                    "token", response.reply[offset : offset + ANSWER_CHUNK_SIZE]
                )
                await asyncio.sleep(0)
            # Exact JSON response parity, including tools, evidence and handoff flags.
            yield _event("result", response.model_dump(mode="json"))
        except Exception:
            logger.exception("SSE 对话服务执行失败")
            yield _event("error", chat_service.ERROR_REPLY)
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                # asyncio cannot stop a worker already running in to_thread.
                # Its workflow may finish/persist, but no draft is sent to the client.
        yield _event("done", "[DONE]")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
