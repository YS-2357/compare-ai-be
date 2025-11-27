"""회원가입/로그인 엔드포인트."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.auth import get_auth_client
from app.api.schemas import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse
from app.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/register", response_model=RegisterResponse)
async def register(payload: RegisterRequest) -> RegisterResponse:
    """이메일+비밀번호 회원가입을 수행한다."""

    client = get_auth_client()
    try:
        result = await client.signup(payload.email, payload.password)
    except HTTPException as exc:
        logger.warning("회원가입 실패: %s", exc.detail)
        raise

    user = result.get("user") or {}
    return RegisterResponse(
        id=user.get("id"),
        email=user.get("email"),
        confirmation_sent_at=result.get("confirmation_sent_at"),
    )


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    """이메일+비밀번호 로그인 후 Supabase 토큰을 반환한다."""

    client = get_auth_client()
    try:
        result = await client.signin(payload.email, payload.password)
    except HTTPException as exc:
        logger.warning("로그인 실패: %s", exc.detail)
        raise

    return LoginResponse(
        access_token=result.get("access_token"),
        token_type=result.get("token_type", "bearer"),
        expires_in=result.get("expires_in"),
        refresh_token=result.get("refresh_token"),
        user=result.get("user"),
    )


__all__ = ["router"]

