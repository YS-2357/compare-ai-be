"""애플리케이션 설정 모듈."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """환경 변수를 통해 주입되는 기본 설정."""

    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = 8000
    streamlit_port: int = 8501
    streamlit_headless: bool = True

    # 멀티턴/컨텍스트 제어
    max_turns_default: int = 3
    max_context_messages: int = 10  # 최근 메시지 유지 개수 (약 5턴: user/assistant 합산)

    env: Literal["local", "test", "prod"] = "local"
    langsmith_project: str = "Compare-AI-BE"
    supabase_url: str | None = None
    supabase_jwks_url: str | None = None
    supabase_aud: str = "authenticated"
    supabase_anon_key: str | None = None
    supabase_service_role_key: str | None = None
    admin_bypass_token: str | None = None
    upstash_redis_url: str | None = None
    upstash_redis_token: str | None = None
    daily_usage_limit: int = 3
    model_openai: str = "gpt-4o-mini"
    model_gemini: str = "gemini-2.5-flash-lite"
    model_anthropic: str = "claude-haiku-4-5-20251001"
    model_upstage: str = "solar-mini"
    model_perplexity: str = "sonar"
    model_mistral: str = "mistral-large-latest"
    model_groq: str = "llama3-70b-8192"
    model_cohere: str = "command-r-plus"

    @staticmethod
    def from_env() -> Settings:
        """환경 변수와 `.env` 값을 기반으로 Settings를 생성한다.

        Returns:
            Settings: 현재 실행 환경에 맞춘 설정 인스턴스.
        """

        return Settings(
            fastapi_host=os.getenv("FASTAPI_HOST", "127.0.0.1"),
            fastapi_port=int(os.getenv("FASTAPI_PORT", "8000")),
            streamlit_port=int(os.getenv("STREAMLIT_SERVER_PORT", "8501")),
            streamlit_headless=os.getenv("STREAMLIT_SERVER_HEADLESS", "true").lower() == "true",
            env=os.getenv("APP_ENV", "local"),  # type: ignore[assignment]
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "Compare-AI-BE"),
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_jwks_url=os.getenv("SUPABASE_JWKS_URL"),
            supabase_aud=os.getenv("SUPABASE_JWT_AUD", "authenticated"),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY"),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            admin_bypass_token=os.getenv("ADMIN_BYPASS_TOKEN"),
            upstash_redis_url=(os.getenv("UPSTASH_REDIS_URL") or "").strip() or None,
            upstash_redis_token=(os.getenv("UPSTASH_REDIS_TOKEN") or "").strip() or None,
            daily_usage_limit=int(os.getenv("DAILY_USAGE_LIMIT", "3")),
            model_openai=os.getenv("MODEL_OPENAI", "gpt-4o-mini"),
            model_gemini=os.getenv("MODEL_GEMINI", "gemini-2.5-flash-lite"),
            model_anthropic=os.getenv("MODEL_ANTHROPIC", "claude-haiku-4-5-20251001"),
            model_upstage=os.getenv("MODEL_UPSTAGE", "solar-mini"),
            model_perplexity=os.getenv("MODEL_PERPLEXITY", "sonar"),
            model_mistral=os.getenv("MODEL_MISTRAL", "mistral-large-latest"),
            model_groq=os.getenv("MODEL_GROQ", "llama3-70b-8192"),
            model_cohere=os.getenv("MODEL_COHERE", "command-r-plus"),
            max_turns_default=int(os.getenv("MAX_TURNS_DEFAULT", "3")),
            max_context_messages=int(os.getenv("MAX_CONTEXT_MESSAGES", "10")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """전역적으로 재사용 가능한 Settings 인스턴스를 반환한다.

    Returns:
        Settings: 캐싱된 설정 인스턴스.
    """

    return Settings.from_env()
