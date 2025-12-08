# app/api/

> 최종 업데이트: 2025-12-05 · 모델 오버라이드 + 관리자 이메일 우회

- `routes.py`: `/health`, `/api/ask`, `/usage` 엔드포인트. 모델 덮어쓰기와 관리자 우회 플래그를 LangGraph로 전달하고, 사용량 헤더를 스트리밍 응답에 포함.
- `auth_routes.py`: `/auth/register`, `/auth/login` (Supabase Auth REST 연동).
- `deps.py`: 공용 Depends(현재 사용자, 설정, 레이트리밋) — JWT 클레임과 `ADMIN_EMAIL` 비교.
- `schemas/`: 요청/응답 Pydantic 모델(`ask.py`, `auth.py` 등).

비고:
- Authorization: Bearer <token> 필수. `ADMIN_EMAIL` 계정은 `remaining=null`, 턴 제한 우회.
- `/api/ask` payload에 `models` 매핑을 넣으면 공급자별 기본 모델을 덮어씀.
- Upstash 연결 시 응답 헤더에 `X-Usage-Limit` / `X-Usage-Remaining` 포함.

