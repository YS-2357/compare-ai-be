"""인증 요청/응답 스키마."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """회원가입 요청 스키마."""

    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    """로그인 요청 스키마."""

    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Supabase Auth 토큰 응답을 래핑한다."""

    access_token: str
    token_type: str
    expires_in: int | None = None
    refresh_token: str | None = None
    user: dict | None = None


class RegisterResponse(BaseModel):
    """회원가입 후 전달할 최소 유저 정보."""

    id: str | None = None
    email: str | None = None
    confirmation_sent_at: str | None = None


__all__ = ["RegisterRequest", "LoginRequest", "LoginResponse", "RegisterResponse"]

