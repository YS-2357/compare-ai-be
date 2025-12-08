"""공통 응답 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """FastAPI에서 사용하는 에러 응답 포맷."""

    detail: str = Field(..., description="에러 메시지.")


class HealthResponse(BaseModel):
    """헬스 체크 응답."""

    status: str = Field(..., description="서비스 상태. 정상일 경우 'ok'.", examples=["ok"])


class UsageResponse(BaseModel):
    """일일 사용량 조회 응답."""

    limit: int = Field(..., description="일일 허용 호출 횟수.", examples=[3])
    remaining: int | None = Field(
        None,
        description="남은 호출 가능 횟수. 관리자가 우회 권한을 가진 경우 None.",
        examples=[2, None],
    )
    bypass: bool = Field(..., description="우회 계정 여부.", examples=[False])


__all__ = ["ErrorResponse", "HealthResponse", "UsageResponse"]
