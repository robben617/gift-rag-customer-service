"""POST /api/chat — JSON transport for the shared chat service."""

from fastapi import APIRouter, Depends

from src.api import chat_service
from src.api.auth import CurrentUser
from src.api.deps import get_current_user
from src.api.models import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["对话"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    return await chat_service.execute_chat(req, current_user)
