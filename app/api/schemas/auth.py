"""인증 요청/응답 스키마."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """회원가입 요청 스키마."""

    email: EmailStr = Field(..., description="회원가입 이메일", examples=["user@example.com"])
    password: str = Field(
        min_length=6,
        max_length=128,
        description="비밀번호(6~128자)",
        examples=["secret123"],
    )


class LoginRequest(BaseModel):
    """로그인 요청 스키마."""

    email: EmailStr = Field(..., description="로그인 이메일", examples=["user@example.com"])
    password: str = Field(..., description="비밀번호", examples=["secret123"])


class LoginResponse(BaseModel):
    """Supabase Auth 토큰 응답을 래핑한다."""

    access_token: str = Field(..., description="JWT 액세스 토큰", examples=["eyJhbGciOi..."])
    token_type: str = Field(..., description="토큰 타입", examples=["bearer"])
    expires_in: int | None = Field(None, description="만료까지 남은 초", examples=[3600])
    refresh_token: str | None = Field(None, description="리프레시 토큰", examples=["eyJhbGciOi..."])
    user: dict | None = Field(None, description="Supabase 사용자 정보", examples=[{"id": "uuid", "email": "user@example.com"}])


class RegisterResponse(BaseModel):
    """회원가입 후 전달할 최소 유저 정보."""

    id: str | None = Field(None, description="사용자 ID", examples=["uuid"])
    email: str | None = Field(None, description="사용자 이메일", examples=["user@example.com"])
    confirmation_sent_at: str | None = Field(None, description="인증 메일 발송 시각", examples=["2025-12-06T12:00:00Z"])


__all__ = ["RegisterRequest", "LoginRequest", "LoginResponse", "RegisterResponse"]
