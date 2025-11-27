"""레이트 리밋 관련 FastAPI dependency."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status

from app.logger import get_logger

from .upstash import get_rate_limiter

logger = get_logger(__name__)


def _seconds_until_midnight_utc() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((tomorrow - now).total_seconds())


async def enforce_daily_limit(user_id: str, limit: int) -> None:
    """사용자별 일일 호출 제한을 적용한다."""

    try:
        client = get_rate_limiter()
        key = f"usage:{user_id}:{datetime.now(timezone.utc).date().isoformat()}"
        count = await client.incr_with_expiry(key, _seconds_until_midnight_utc())
        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="daily usage limit exceeded",
            )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - 백엔드 장애 시 우회
        logger.warning("레이트리밋 백엔드 오류, 우회 적용: %s", exc)
        return


__all__ = ["enforce_daily_limit"]
