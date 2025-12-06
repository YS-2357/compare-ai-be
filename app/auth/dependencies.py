"""인증 관련 FastAPI dependency."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Header, status

from app.config import get_settings
from .supabase import AuthenticatedUser, get_verifier


async def get_current_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> AuthenticatedUser:
    """JWT를 검증하고 사용자 정보를 반환한다."""

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authorization header missing")

    token = authorization.split(" ", 1)[1].strip()
    verifier = get_verifier()
    user = await verifier.verify(token)

    settings = get_settings()
    admin_email = (settings.admin_email or "").lower()
    user_email = (user.get("email") or "").lower()
    if admin_email and user_email and user_email == admin_email:
        user["bypass"] = True
        user["role"] = user.get("role") or "admin"

    return user


__all__ = ["get_current_user", "AuthenticatedUser"]
