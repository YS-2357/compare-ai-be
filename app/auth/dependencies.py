"""인증 관련 FastAPI dependency."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Header, status

from app.config import get_settings
from .supabase import AuthenticatedUser, get_verifier


async def get_current_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
    admin_bypass: str | None = Header(default=None, alias="x-admin-bypass"),
) -> AuthenticatedUser:
    """JWT를 검증하고 사용자 정보를 반환한다. 관리자 바이패스를 지원한다."""

    settings = get_settings()

    # 관리자 바이패스 토큰 우선
    if settings.admin_bypass_token and admin_bypass and admin_bypass == settings.admin_bypass_token:
        return AuthenticatedUser(sub="admin-bypass", email=None, role="admin", bypass=True)

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authorization header missing")

    token = authorization.split(" ", 1)[1].strip()
    verifier = get_verifier()
    return await verifier.verify(token)


__all__ = ["get_current_user", "AuthenticatedUser"]

