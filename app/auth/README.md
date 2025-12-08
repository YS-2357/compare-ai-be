# app/auth

> 최종 업데이트: 2025-12-05 · 관리자 이메일 우회 + Supabase 타임아웃 정리

- Supabase JWT 검증 및 Auth REST 클라이언트 유틸.
- `dependencies.py`: `get_current_user`가 Bearer JWT를 검증하고 `ADMIN_EMAIL` 사용자에 bypass 플래그 부여.
- `supabase.py`: JWKS 캐시/검증, signup/signin 클라이언트(타임아웃 설정, 종료 훅 포함).

환경 변수:
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_JWKS_URL` (기본: `https://<project>.supabase.co/auth/v1/.well-known/jwks.json`)
- `SUPABASE_JWT_AUD` (기본: `authenticated`)
- `SUPABASE_HTTP_TIMEOUT`, `SUPABASE_JWKS_CACHE_TTL`
- `ADMIN_EMAIL` (선택, 지정 시 해당 계정은 사용량/턴 제한을 우회)
