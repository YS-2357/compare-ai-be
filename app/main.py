"""FastAPI 애플리케이션 팩토리."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .config import Settings, get_settings
from .auth.supabase import shutdown_auth_client
from .rate_limit.upstash import shutdown_rate_limiter


FASTAPI_DESCRIPTION = (
    "LangGraph 기반 Compare-AI 백엔드 API.\n\n"
    "- `/health`: 서비스 상태 확인.\n"
    "- `/auth/register`, `/auth/login`: 이메일/비밀번호 회원관리 (Supabase 연동).\n"
    "- `/api/ask`: LangGraph 워크플로우 스트리밍(NDJSON) — partial/summary 이벤트, 모델 오버라이드 및 사용량 헤더 포함.\n"
    "- `/usage`: 일일 호출 제한 조회 (관리자 bypass 시 남은 횟수 null).\n\n"
    "Swagger UI(`/docs`)와 ReDoc(`/redoc`)에서 요청/응답 예시, 스키마, 오류 포맷을 확인하세요."
)

TAGS_METADATA = [
    {"name": "system", "description": "헬스 체크 및 공통 시스템 엔드포인트"},
    {"name": "auth", "description": "Supabase 기반 회원가입/로그인 API"},
    {"name": "questions", "description": "LangGraph 질의 처리 스트리밍 API"},
    {"name": "usage", "description": "일일 사용량 및 제한 정보"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup/shutdown에서 공용 리소스를 정리한다."""

    try:
        yield
    finally:
        await shutdown_auth_client()
        await shutdown_rate_limiter()


def create_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI 애플리케이션을 구성해 반환한다.

    Args:
        settings: 외부에서 주입할 `Settings` 인스턴스. 생략 시 `.env`/환경변수를 읽어 생성한다.

    Returns:
        FastAPI: 라우터와 미들웨어가 등록된 FastAPI 인스턴스.
    """

    settings = settings or get_settings()
    app = FastAPI(
        title=(settings.fastapi_title or "").strip('"'),
        version=settings.fastapi_version,
        description=FASTAPI_DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.state.settings = settings
    return app


app = create_app()
