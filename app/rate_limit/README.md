# app/rate_limit/

- Upstash Redis 기반 일일 사용량 제한.
- `upstash.py`: 클라이언트(`INCR` + `EXPIRE` 파이프라인).
- `dependencies.py`: `enforce_daily_limit` (Upstash 실패 시 로컬 메모리 캐시 폴백).

환경변수:
- `UPSTASH_REDIS_URL`
- `UPSTASH_REDIS_TOKEN`
- `DAILY_USAGE_LIMIT` (기본 3)

메모:
- Upstash 연결이 되면 Redis 기준으로 카운트, 실패 시 프로세스 메모리로 3회 제한 유지(기본 3회 제한은 검증 완료).
