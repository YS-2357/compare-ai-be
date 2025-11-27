"""LLM 클라이언트 및 LangSmith 환경 설정."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_perplexity import ChatPerplexity
from langchain_upstage import ChatUpstage

try:
    from langchain_mistralai.chat_models import ChatMistralAI
except ImportError:  # pragma: no cover
    ChatMistralAI = None  # type: ignore[assignment]

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None  # type: ignore[assignment]

try:
    from langchain_cohere import ChatCohere
except ImportError:  # pragma: no cover
    ChatCohere = None  # type: ignore[assignment]

# LangSmith UUID v7 지원
try:
    from langsmith import uuid7 as create_uuid
except ImportError:
    from uuid import uuid4 as create_uuid

# 환경변수 로드
load_dotenv()

# LangSmith 설정 (환경변수 기반)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Compare-AI-BE")

__all__ = [
    "ChatAnthropic",
    "ChatGoogleGenerativeAI",
    "ChatOpenAI",
    "ChatPerplexity",
    "ChatUpstage",
    "ChatMistralAI",
    "ChatGroq",
    "ChatCohere",
    "create_uuid",
]

