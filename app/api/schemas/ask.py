"""질문 요청 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """질문을 포함하는 요청 스키마."""

    question: str = Field(..., description="사용자 질문", examples=["LangGraph란 무엇인가요?"])
    turn: int | None = Field(None, description="현재 턴(1부터 시작)", examples=[1])
    max_turns: int | None = Field(None, description="허용 최대 턴", examples=[3])
    history: list[dict[str, str]] | None = Field(
        None,
        description="이전 대화 이력 목록",
        examples=[[{"role": "assistant", "content": "이전 답변"}]],
    )
    models: dict[str, str] | None = Field(
        None,
        description="공급자별 모델 덮어쓰기 (provider:model)",
        examples=[{"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash-lite"}],
    )


__all__ = ["AskRequest"]
