"""로컬 실행 엔트리포인트.

- 기본: FastAPI + Streamlit(run_app) 동시 실행
- 환경변수 `APP_MODE=api`일 때는 FastAPI만 단독 실행
"""

from __future__ import annotations

import os

import uvicorn

from app.config import get_settings
from app.main import app as fastapi_app
from scripts.run_app import main as run_main


def main() -> None:
    mode = os.getenv("APP_MODE", "").lower()
    settings = get_settings()
    host = os.getenv("FASTAPI_HOST", settings.fastapi_host)
    port = int(os.getenv("PORT") or settings.fastapi_port)

    if mode == "api":
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
    else:
        run_main()


if __name__ == "__main__":
    main()
