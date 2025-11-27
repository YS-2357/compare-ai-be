"""질문 요청 스키마."""

from __future__ import annotations

from pydantic import BaseModel


class AskRequest(BaseModel):
    """질문을 포함하는 요청 스키마."""

    question: str
    turn: int | None = None
    max_turns: int | None = None
    history: list[dict[str, str]] | None = None


__all__ = ["AskRequest"]

