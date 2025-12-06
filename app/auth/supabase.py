"""Supabase JWT 검증 및 JWKS 캐시 로직을 제공한다."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, TypedDict

import httpx
from fastapi import HTTPException, status
from jose import JWTError, jwt

from app.config import get_settings


class AuthenticatedUser(TypedDict):
    """검증된 사용자 정보를 표현한다."""

    sub: str
    email: str | None
    role: str | None
    bypass: bool


class _JWKSCache:
    """Supabase JWKS를 주기적으로 캐시한다."""

    def __init__(
        self,
        jwks_url: str,
        cache_ttl: int = 300,
        api_key: str | None = None,
        http_timeout: float = 5.0,
    ) -> None:
        self.jwks_url = jwks_url
        self.cache_ttl = cache_ttl
        self.api_key = api_key
        self._keys: dict[str, dict[str, Any]] | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()
        self._http_timeout = http_timeout

    async def get_key(self, kid: str) -> dict[str, Any]:
        async with self._lock:
            if self._keys is None or self._expires_at <= asyncio.get_event_loop().time():
                await self._refresh()
            if not self._keys or kid not in self._keys:
                raise KeyError("JWKS key not found")
            return self._keys[kid]

    async def _refresh(self) -> None:
        headers = None
        params = None
        if self.api_key:
            headers = {"apikey": self.api_key, "Authorization": f"Bearer {self.api_key}"}
            params = {"apikey": self.api_key}

        base = self.jwks_url.rstrip("/")
        root, _, last = base.rpartition("/")
        urls = [base]
        if root:
            urls.append(f"{root}/keys")
            urls.append(f"{root}/tenants/default/jwks")
            urls.append(f"{root}/.well-known/jwks.json")
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            last_error: Exception | None = None
            for url in urls:
                try:
                    resp = await client.get(url, headers=headers, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    keys = {item["kid"]: item for item in data.get("keys", []) if "kid" in item}
                    self._keys = keys
                    self._expires_at = asyncio.get_event_loop().time() + self.cache_ttl
                    return
                except Exception as exc:  # pragma: no cover - fallback 케이스
                    last_error = exc
                    continue
            if last_error:
                raise last_error


@dataclass(frozen=True)
class SupabaseVerifier:
    """Supabase JWT 검증기."""

    jwks_cache: _JWKSCache
    audience: str
    issuer: str

    async def verify(self, token: str) -> AuthenticatedUser:
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if not kid:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token header")
            key = await self.jwks_cache.get_key(kid)
            claims = jwt.decode(
                token,
                key,
                algorithms=[key.get("alg", "RS256")],
                audience=self.audience,
                issuer=self.issuer,
            )
        except (JWTError, KeyError) as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token") from exc

        return AuthenticatedUser(
            sub=str(claims.get("sub")),
            email=claims.get("email"),
            role=claims.get("role"),
            bypass=False,
        )


_verifier: SupabaseVerifier | None = None
_auth_client: "SupabaseAuthClient" | None = None


def get_verifier() -> SupabaseVerifier:
    """환경 변수를 기반으로 Supabase 검증기를 싱글턴으로 반환한다."""

    global _verifier
    if _verifier is not None:
        return _verifier

    settings = get_settings()
    jwks_url = settings.supabase_jwks_url
    if not jwks_url:
        if not settings.supabase_url:
            raise RuntimeError("Supabase URL이 설정되지 않았습니다.")
        jwks_url = settings.supabase_url.rstrip("/") + "/auth/v1/jwks"
    issuer = (settings.supabase_url or "").rstrip("/") + "/auth/v1"
    api_key = settings.supabase_anon_key or settings.supabase_service_role_key
    jwks_cache = _JWKSCache(
        jwks_url,
        cache_ttl=settings.supabase_jwks_cache_ttl,
        api_key=api_key,
        http_timeout=settings.supabase_http_timeout,
    )
    _verifier = SupabaseVerifier(jwks_cache, audience=settings.supabase_aud, issuer=issuer)
    return _verifier


class SupabaseAuthClient:
    """Supabase Auth REST 호출 래퍼."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        return {"apikey": self.api_key, "Authorization": f"Bearer {self.api_key}"}

    async def signup(self, email: str, password: str) -> dict[str, Any]:
        url = f"{self.base_url}/auth/v1/signup"
        try:
            resp = await self._client.post(url, json={"email": email, "password": password}, headers=self._headers())
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json() if exc.response else {"message": str(exc)}
            raise HTTPException(status_code=exc.response.status_code if exc.response else 400, detail=detail)

    async def signin(self, email: str, password: str) -> dict[str, Any]:
        url = f"{self.base_url}/auth/v1/token?grant_type=password"
        try:
            resp = await self._client.post(
                url, json={"email": email, "password": password}, headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json() if exc.response else {"message": str(exc)}
            raise HTTPException(status_code=exc.response.status_code if exc.response else 400, detail=detail)

    async def aclose(self) -> None:
        """재사용 중인 HTTP 클라이언트를 종료한다."""

        await self._client.aclose()


def get_auth_client() -> "SupabaseAuthClient":
    """회원가입/로그인을 위한 Supabase Auth 클라이언트를 반환한다."""

    global _auth_client
    if _auth_client is not None:
        return _auth_client

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise RuntimeError("Supabase Auth 설정이 없습니다.")
    _auth_client = SupabaseAuthClient(
        settings.supabase_url,
        settings.supabase_anon_key,
        timeout=settings.supabase_http_timeout,
    )
    return _auth_client


async def shutdown_auth_client() -> None:
    """FastAPI lifespan에서 Auth 클라이언트를 정리한다."""

    global _auth_client
    if _auth_client is None:
        return
    try:
        await _auth_client.aclose()
    finally:
        _auth_client = None
