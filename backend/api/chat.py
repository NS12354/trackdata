"""Chatbot endpoint (Phase 6)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from pipeline.chat import answer_question
from .security import require_api_key

router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(require_api_key)])


class ChatRequest(BaseModel):
    question: str
    video_id: Optional[str] = None


@router.post("")
def chat(req: ChatRequest):
    """Answer a question grounded in the activity commentary + metrics."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="empty question")
    try:
        return answer_question(req.question, req.video_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"chat backend error: {exc}")
