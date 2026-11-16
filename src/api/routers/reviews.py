"""Authenticated review lookup and administrator-only decisions."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api import review_service
from src.api.auth import CurrentUser
from src.api.deps import get_current_user, require_admin
from src.api.models import ReviewDecisionRequest
from src.conversation.review_store import ReviewConflict, ReviewNotFound

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reviews", tags=["回答审批"])


async def _call(function, *args):
    try:
        return await asyncio.to_thread(function, *args)
    except ReviewNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ReviewConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        logger.exception("审批服务失败")
        raise HTTPException(503, "审批服务暂不可用，请稍后重试同一操作") from exc


@router.get("")
async def list_reviews(
    session_id: str | None = None,
    status: str | None = Query(
        None, pattern="^(pending|processing|resolved|superseded)$"
    ),
    user: CurrentUser = Depends(get_current_user),
):
    return {"tasks": await _call(review_service.list_reviews, user, session_id, status)}


@router.get("/{task_id}")
async def get_review(task_id: str, user: CurrentUser = Depends(get_current_user)):
    return await _call(review_service.get_review, task_id, user)


@router.post("/{task_id}/decision")
async def decide_review(
    task_id: str,
    req: ReviewDecisionRequest,
    admin: CurrentUser = Depends(require_admin),
):
    return await _call(review_service.decide, task_id, req.decision, req.note, admin)
