"""사용량 조회 라우터."""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthenticatedUser, get_current_user, get_settings
from app.api.schemas import UsageResponse
from app.rate_limit.upstash import get_rate_limiter
from app.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get(
    "/usage",
    response_model=UsageResponse,
    responses={
        200: {
            "description": "현재 사용자 일일 사용량 조회",
            "content": {"application/json": {"example": {"limit": 3, "remaining": 2, "bypass": False}}},
        },
        401: {"description": "인증 필요 (Authorization: Bearer <JWT>)"},
        503: {"description": "레이트리밋 백엔드 오류"},
    },
    summary="일일 사용량 조회",
)
async def get_usage(user: AuthenticatedUser = Depends(get_current_user)):
    """현재 사용자 남은 일일 사용량을 조회한다."""

    settings = get_settings()
    if user.get("bypass"):
        return {"limit": settings.daily_usage_limit, "remaining": None, "bypass": True}

    limiter = get_rate_limiter()
    key = f"usage:{user['sub']}:{datetime.now(timezone.utc).date().isoformat()}"
    try:
        count = await limiter.get(key)
    except Exception as exc:
        logger.error("usage 조회 실패: %s", exc)
        raise HTTPException(status_code=503, detail="rate limit backend unavailable")

    remaining = max(0, settings.daily_usage_limit - count)
    return {"limit": settings.daily_usage_limit, "remaining": remaining, "bypass": False}


__all__ = ["router"]
