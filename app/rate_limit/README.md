# app/rate_limit/

> 최종 업데이트: 2025-12-05 · Upstash 타임아웃 + 종료 훅

- Upstash Redis 기반 일일 사용량 제한기.
- `upstash.py`: HTTP 타임아웃 설정 및 종료 훅을 가진 비동기 클라이언트(INCR + EXPIRE).
- `dependencies.py`: `enforce_daily_limit`로 호출 전 한도 검사(백엔드 장애는 HTTP 오류로 전파).

환경 변수:
- `UPSTASH_REDIS_URL` / `UPSTASH_REDIS_TOKEN`
- `DAILY_USAGE_LIMIT` (기본 3)
- `UPSTASH_HTTP_TIMEOUT` (초 단위, 기본 5)

비고:
- Upstash 연결 가능 시 카운터는 원격에 저장되며, 장애 시 FastAPI가 503으로 백엔드 이슈를 노출합니다.
