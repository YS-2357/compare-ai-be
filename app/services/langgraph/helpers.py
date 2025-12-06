"""LangGraph 프롬프트/메시지 유틸."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.logger import get_logger
from app.config import get_settings
from .summaries import _preview

logger = get_logger(__name__)
settings_cache = get_settings()


class Answer(BaseModel):
    """LCEL 체인 출력 스키마."""

    content: str = Field(..., description="짧게 요약된 답변")
    source: str | None = Field(default=None, description="출처 URL 또는 참고 정보(옵션)")


def _message_to_text(message: Any) -> str | None:
    """여러 형태의 메시지를 role: content 문자열로 변환한다."""

    if isinstance(message, (list, tuple)) and len(message) == 2:
        role, content = message
        return f"{role}: {content}"
    if isinstance(message, dict):
        return f"{message.get('role')}: {message.get('content')}"
    if isinstance(message, BaseMessage):
        role = getattr(message, "type", message.__class__.__name__)
        content = message.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            content = " ".join([p for p in parts if p])
        return f"{role}: {content}"
    return None


def _build_prompt() -> ChatPromptTemplate:
    parser = PydanticOutputParser(pydantic_object=Answer)
    instructions = parser.get_format_instructions()
    system = (
        "You are a helpful multi-turn assistant. Use the conversation history to stay on topic; if there is none, treat the input as a new question. "
        "Respond only in Korean, write clear and natural paragraphs, and provide enough detail to fully answer the question without being overly terse. "
        "If additional context is needed, briefly note the limitation. "
        "{format_instructions}"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", "{question}"),
        ]
    ).partial(format_instructions=instructions)


def _render_history_for_model(state: Any, label: str, max_messages: int = 10) -> str:
    """모델별로 최근 히스토리와 요약을 합쳐 프롬프트용 문자열을 만든다."""

    user_msgs = state.get("user_messages") or []
    model_histories = state.get("model_messages") or {}
    model_msgs = model_histories.get(label, []) or []
    summary_present = bool((state.get("model_summaries") or {}).get(label))
    logger.info(
        "history/render label=%s user_msgs=%d model_msgs=%d summary=%s",
        label,
        len(user_msgs),
        len(model_msgs),
        summary_present,
    )

    lines: list[str] = []
    summaries = state.get("model_summaries") or {}
    summary = summaries.get(label)
    if summary:
        lines.append(f"[이전 요약] {summary}")

    combined: list[str] = []
    max_len = max(len(user_msgs), len(model_msgs))
    for idx in range(max_len):
        if idx < len(user_msgs):
            msg_text = _message_to_text(user_msgs[idx])
            if msg_text:
                combined.append(msg_text)
        if idx < len(model_msgs):
            msg_text = _message_to_text(model_msgs[idx])
            if msg_text:
                combined.append(msg_text)

    lines.extend(combined[-max_messages:])

    history_text = "\n".join(lines)
    logger.info(
        "history/rendered label=%s lines=%d chars=%d preview=%s",
        label,
        len(lines),
        len(history_text),
        _preview(history_text),
    )
    return history_text


def _build_prompt_input(state: Any, label: str) -> str:
    """
    대화 이력과 현재 질문을 분리해 전달한다.
    - history: 모델별 인터리브된 최근 히스토리
    - question: 현재 질문
    """

    max_context = settings_cache.max_context_messages
    history_text = _render_history_for_model(state, label, max_messages=max_context)
    user_msgs = state.get("user_messages") or []
    current_question = ""
    if user_msgs:
        for msg in reversed(user_msgs):
            if isinstance(msg, (list, tuple)) and len(msg) == 2 and msg[0] == "user":
                current_question = msg[1]
                break
            if isinstance(msg, dict) and msg.get("role") == "user":
                current_question = msg.get("content", "")
                break
    if history_text.strip():
        prompt = (
            "[Conversation History]\n"
            f"{history_text}\n\n"
            "[Current Question]\n"
            f"{current_question}\n\n"
            "If the wording is ambiguous, prefer the most recent topic or flow. Respond only in Korean."
        )
        mode = "with_history"
    else:
        prompt = (
            "[Current Question]\n"
            f"{current_question}\n\n"
            "This is the first turn; there is no prior conversation. Answer clearly and only in Korean."
        )
        mode = "first_turn"
    logger.info(
        "prompt/built label=%s turn=%s mode=%s chars=%d preview=%s",
        label,
        state.get("turn") or 1,
        mode,
        len(prompt),
        _preview(prompt),
    )
    return prompt


__all__ = ["Answer", "_message_to_text", "_build_prompt", "_render_history_for_model", "_build_prompt_input"]
