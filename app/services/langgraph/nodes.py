"""LangGraph 노드 정의 및 LLM 호출 래퍼."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph.message import add_messages
from langgraph.types import Send

from app.config import get_settings
from app.logger import get_logger
from .errors import build_status_from_error, build_status_from_response
from .helpers import Answer, _build_prompt, _build_prompt_input, _message_to_text, _render_history_for_model

from .llm_registry import (
    ChatAnthropic,
    ChatCohere,
    ChatGoogleGenerativeAI,
    ChatGroq,
    ChatMistralAI,
    ChatOpenAI,
    ChatPerplexity,
    ChatUpstage,
)
from .summaries import _preview, _summarize_content

logger = get_logger(__name__)
settings_cache = get_settings()
DEFAULT_MAX_TURNS = settings_cache.max_turns_default
MAX_CONTEXT_MESSAGES = settings_cache.max_context_messages  # 최근 메시지 유지 개수 (약 5턴: user/assistant 합산)


def merge_dicts(existing: dict | None, new: dict | None) -> dict:
    """LangGraph 상태 병합 시 딕셔너리를 병합한다."""

    merged: dict = dict(existing or {})
    merged.update(new or {})
    return merged


def merge_model_messages(existing: dict | None, new: dict | None) -> dict:
    """모델별 메시지 딕셔너리를 병합한다."""

    merged: dict[str, list] = dict(existing or {})
    for model, messages in (new or {}).items():
        merged[model] = add_messages(merged.get(model, []), messages or [])
    return merged


class GraphState(TypedDict, total=False):
    """LangGraph 실행 시 공유되는 상태 정의."""

    max_turns: Annotated[int | None, "최대 턴 수"]
    turn: Annotated[int | None, "현재 턴 인덱스"]
    active_models: Annotated[list[str] | None, "활성화된 모델 목록"]

    # 공통 유저 발화 + 모델별 히스토리/요약
    user_messages: Annotated[list, add_messages]
    model_messages: Annotated[dict[str, list], merge_model_messages]
    model_summaries: Annotated[dict[str, str] | None, merge_dicts]

    # 호출 메타(필요 시 사용)
    raw_responses: Annotated[dict[str, str] | None, merge_dicts]
    raw_sources: Annotated[dict[str, str | None] | None, merge_dicts]
    api_status: Annotated[dict[str, Any] | None, merge_dicts]


def _default_active_models() -> list[str]:
    return list(NODE_CONFIG.keys())


def _model_label(node_name: str) -> str:
    meta = NODE_CONFIG.get(node_name)
    return meta["label"] if meta else node_name


async def _call_model_common(
    label: str,
    state: GraphState,
    llm_factory: Callable[[], Any],
    *,
    message_transform: Callable[[str, str | None], str] | None = None,
) -> GraphState:
    """모델 호출 공통 루틴."""

    prompt_input = _build_prompt_input(state, label)
    logger.debug("%s 호출 시작", label)
    try:
        llm = llm_factory()
        content, source, status = await _invoke_parsed(llm, prompt_input, label)
        msg_payload = message_transform(content, source) if message_transform else content
        updated_msgs, summaries = await _maybe_summarize_history(
            llm, state, label, [("assistant", msg_payload)], state.get("turn")
        )
        logger.info("%s 응답 완료", label)
        return GraphState(
            model_messages={label: updated_msgs},
            model_summaries=summaries,
            api_status={label: status},
            raw_responses={label: content},
            raw_sources={label: source},
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("%s 호출 실패: %s", label, exc)
        return GraphState(
            api_status={label: status},
            model_messages={label: [format_response_message(f"{label} 오류", exc)]},
        )


async def _maybe_summarize_history(
    llm: Any, state: GraphState, label: str, new_messages: list, turn: int | None
) -> tuple[list, dict[str, str]]:
    """
    모델별 히스토리를 갱신하고 필요 시 요약한다.
    - turn >= 2 또는 메시지가 MAX_CONTEXT_MESSAGES 초과 시 요약
    - 요약은 model_summaries[label]에 1개만 유지
    """

    existing = (state.get("model_messages") or {}).get(label, [])
    updated = add_messages(existing, new_messages)
    summaries = state.get("model_summaries") or {}

    should_summarize = (turn or 1) >= 2 and len(updated) > MAX_CONTEXT_MESSAGES
    if should_summarize:
        # 히스토리를 문자열로 합쳐 요약
        text_lines = []
        for msg in updated:
            msg_text = _message_to_text(msg)
            if msg_text:
                text_lines.append(msg_text)
        history_text = "\n".join(text_lines)
        summary = await _summarize_content(llm, history_text, label)
        summaries[label] = summary
        # 최근 메시지만 남기되 요약은 별도 필드에 저장
        updated = updated[-MAX_CONTEXT_MESSAGES:]
        logger.info(
            "history/summarize label=%s turn=%s msgs_before=%d msgs_after=%d summary_len=%d preview=%s",
            label,
            turn,
            len(existing),
            len(updated),
            len(summary),
            _preview(summary),
        )
    return updated, summaries


def _build_prompt_input(state: GraphState, label: str) -> str:
    """
    대화 이력과 현재 질문을 분리해 전달한다.
    - history: 모델별 인터리브된 최근 히스토리
    - question: 현재 질문
    """
    history_text = _render_history_for_model(state, label, max_messages=MAX_CONTEXT_MESSAGES)
    current_question = state.get("question", "")
    return (
        "[대화 이력]\n"
        f"{history_text}\n\n"
        "[현재 질문]\n"
        f"{current_question}\n\n"
        "모호한 표현은 직전 주제나 대화 흐름을 이어서 해석하세요."
    )


def _extract_source(extras: dict[str, Any] | None) -> str | None:
    """추가 메타에서 출처 URL을 추출한다."""

    if not extras:
        return None
    citations = extras.get("citations")
    if isinstance(citations, list) and citations:
        first = citations[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            url = first.get("url") or first.get("source")
            if url:
                return str(url)
    search_results = extras.get("search_results")
    if isinstance(search_results, list) and search_results:
        item = search_results[0]
        if isinstance(item, dict):
            url = item.get("url")
            if url:
                return str(url)
    return None


async def _invoke_parsed(llm: Any, prompt_input: str, label: str) -> tuple[str, str | None, dict[str, Any]]:
    """LLM을 한 번 호출한 뒤 파서를 적용하고, 실패하면 원문을 그대로 사용한다."""

    parser = PydanticOutputParser(pydantic_object=Answer)
    prompt = _build_prompt()
    chain = prompt | llm
    response = await chain.ainvoke({"question": prompt_input})
    status = build_status_from_response(response)
    raw_text = response.content if hasattr(response, "content") else str(response)
    if raw_text is None:
        raw_text = ""
    try:
        parsed: Answer = parser.parse(raw_text)
        content = parsed.content or raw_text
        source = parsed.source or _extract_source(getattr(parsed, "model_extra", None))
        if not source:
            source = _extract_source(getattr(response, "response_metadata", None))
    except Exception:
        content = raw_text
        source = _extract_source(getattr(response, "response_metadata", None))
    return content, source, status


def format_response_message(label: str, payload: Any) -> tuple[str, str]:
    """메시지 로그에 저장할 간단한 (role, content) 튜플을 생성한다."""

    return ("assistant", f"[{label}] {payload}")


def init_question(state: GraphState) -> GraphState:
    """그래프 초기 상태를 검증하고 기본 메시지를 설정한다."""

    max_turns = state.get("max_turns") or DEFAULT_MAX_TURNS
    active_models = state.get("active_models") or list(NODE_CONFIG.keys())
    # 유저/모델 메시지 초기화
    user_messages = state.get("user_messages") or []
    if not user_messages:
        raise ValueError("질문이 비어 있습니다.")
    model_messages = state.get("model_messages") or {}
    model_summaries = state.get("model_summaries") or {}
    turn_value = state.get("turn") or 1

    last_user = next((msg for msg in reversed(user_messages) if isinstance(msg, (list, tuple, dict))), None)
    preview_target = ""
    if isinstance(last_user, (list, tuple)) and len(last_user) == 2:
        preview_target = last_user[1]
    elif isinstance(last_user, dict):
        preview_target = last_user.get("content", "")
    logger.debug("질문 초기화: %s", _preview(preview_target))
    return GraphState(
        max_turns=max_turns,
        turn=turn_value,
        active_models=active_models,
        raw_responses=state.get("raw_responses") or {},
        raw_sources=state.get("raw_sources") or {},
        api_status=state.get("api_status") or {},
        user_messages=user_messages,
        model_messages=model_messages,
        model_summaries=model_summaries,
    )


async def _ainvoke(llm: Any, question: str) -> Any:
    """주어진 LLM에서 비동기 호출을 수행한다."""

    if hasattr(llm, "ainvoke"):
        return await llm.ainvoke(question)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, llm.invoke, question)


async def call_openai(state: GraphState) -> GraphState:
    """OpenAI 모델을 호출하고 응답/상태를 반환한다."""

    settings = get_settings()
    llm_factory = lambda: ChatOpenAI(model=settings.model_openai)
    return await _call_model_common("OpenAI", state, llm_factory)


async def call_gemini(state: GraphState) -> GraphState:
    """Google Gemini 모델을 호출한다."""

    settings = get_settings()
    llm_factory = lambda: ChatGoogleGenerativeAI(model=settings.model_gemini, temperature=0)
    return await _call_model_common("Gemini", state, llm_factory)


async def call_anthropic(state: GraphState) -> GraphState:
    """Anthropic Claude 모델을 호출한다."""

    settings = get_settings()
    llm_factory = lambda: ChatAnthropic(model=settings.model_anthropic, temperature=0)
    return await _call_model_common("Anthropic", state, llm_factory)


async def call_upstage(state: GraphState) -> GraphState:
    """Upstage Solar 모델을 호출한다."""

    settings = get_settings()
    llm_factory = lambda: ChatUpstage(model=settings.model_upstage)
    return await _call_model_common("Upstage", state, llm_factory)


async def call_perplexity(state: GraphState) -> GraphState:
    """Perplexity Sonar 모델을 호출한다."""

    def llm_factory() -> Any:
        pplx_api_key = os.getenv("PPLX_API_KEY")
        if not pplx_api_key:
            raise RuntimeError("PPLX_API_KEY is missing")
        settings = get_settings()
        return ChatPerplexity(temperature=0, model=settings.model_perplexity, pplx_api_key=pplx_api_key)

    msg_transform = lambda content, source: content if not source else f"{content} (src: {source})"
    return await _call_model_common("Perplexity", state, llm_factory, message_transform=msg_transform)


async def call_mistral(state: GraphState) -> GraphState:
    """Mistral AI 모델을 호출한다."""

    def llm_factory() -> Any:
        if ChatMistralAI is None:
            raise RuntimeError("langchain-mistralai 패키지가 설치되어 있지 않습니다.")
        settings = get_settings()
        return ChatMistralAI(model=settings.model_mistral, temperature=0)

    return await _call_model_common("Mistral", state, llm_factory)


async def call_groq(state: GraphState) -> GraphState:
    """Groq 기반 모델을 호출한다."""

    def llm_factory() -> Any:
        if ChatGroq is None:
            raise RuntimeError("langchain-groq 패키지가 설치되어 있지 않습니다.")
        settings = get_settings()
        return ChatGroq(model=settings.model_groq, temperature=0)

    return await _call_model_common("Groq", state, llm_factory)


async def call_cohere(state: GraphState) -> GraphState:
    """Cohere Command 모델을 호출한다."""

    def llm_factory() -> Any:
        if ChatCohere is None:
            raise RuntimeError("langchain-cohere 패키지가 설치되어 있지 않습니다.")
        settings = get_settings()
        return ChatCohere(model=settings.model_cohere, temperature=0)

    return await _call_model_common("Cohere", state, llm_factory)


NODE_CONFIG: dict[str, dict[str, str]] = {
    "call_openai": {"label": "OpenAI", "answer_key": "openai_answer", "status_key": "openai_status"},
    "call_gemini": {"label": "Gemini", "answer_key": "gemini_answer", "status_key": "gemini_status"},
    "call_anthropic": {"label": "Anthropic", "answer_key": "anthropic_answer", "status_key": "anthropic_status"},
    "call_perplexity": {"label": "Perplexity", "answer_key": "perplexity_answer", "status_key": "perplexity_status"},
    "call_upstage": {"label": "Upstage", "answer_key": "upstage_answer", "status_key": "upstage_status"},
    "call_mistral": {"label": "Mistral", "answer_key": "mistral_answer", "status_key": "mistral_status"},
    "call_groq": {"label": "Groq", "answer_key": "groq_answer", "status_key": "groq_status"},
    "call_cohere": {"label": "Cohere", "answer_key": "cohere_answer", "status_key": "cohere_status"},
}


def dispatch_llm_calls(state: GraphState) -> list[Send]:
    """Send API를 활용해 각 LLM 노드를 동시에 실행할 태스크 목록을 생성한다."""

    user_messages = state.get("user_messages") or []
    if not user_messages:
        raise ValueError("질문이 비어 있습니다.")
    active_models = state.get("active_models") or _default_active_models()
    preview_target = ""
    last_user = next((msg for msg in reversed(user_messages) if isinstance(msg, (list, tuple, dict))), None)
    if isinstance(last_user, (list, tuple)) and len(last_user) == 2:
        preview_target = last_user[1]
    elif isinstance(last_user, dict):
        preview_target = last_user.get("content", "")
    logger.info("LLM fan-out 실행: %s | 질문: %s", ", ".join(active_models), _preview(preview_target))
    return [Send(node_name, state) for node_name in active_models]


__all__ = [
    "GraphState",
    "NODE_CONFIG",
    "DEFAULT_MAX_TURNS",
    "dispatch_llm_calls",
    "init_question",
    "_ainvoke",
    "_preview",
    "call_openai",
    "call_gemini",
    "call_anthropic",
    "call_upstage",
    "call_perplexity",
    "call_mistral",
    "call_groq",
    "call_cohere",
    "format_response_message",
]
