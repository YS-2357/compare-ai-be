"""LangGraph 노드 정의 및 LLM 호출 래퍼."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated, Any, TypedDict, cast

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import BaseModel, Field

from app.config import get_settings
from app.logger import get_logger

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
DEFAULT_MAX_TURNS = 3


def merge_dicts(existing: dict | None, new: dict | None) -> dict:
    """LangGraph 상태 병합 시 딕셔너리를 병합한다."""

    merged: dict = dict(existing or {})
    merged.update(new or {})
    return merged


class GraphState(TypedDict, total=False):
    """LangGraph 실행 시 공유되는 상태 정의."""

    question: Annotated[str, "Question"]
    max_turns: Annotated[int | None, "최대 턴 수"]
    conversation_history: Annotated[list[dict[str, str]] | None, "전체 대화 히스토리"]
    history_summary: Annotated[str | None, "요약된 대화"]
    turn: Annotated[int | None, "현재 턴 인덱스"]
    current_inputs: Annotated[dict[str, str] | None, merge_dicts]
    active_models: Annotated[list[str] | None, "활성화된 모델 목록"]

    openai_answer: Annotated[str | None, "OpenAI 응답"]
    gemini_answer: Annotated[str | None, "Google Gemini 응답"]
    anthropic_answer: Annotated[str | None, "Anthropic Claude 응답"]
    upstage_answer: Annotated[str | None, "Upstage 응답"]
    perplexity_answer: Annotated[str | None, "Perplexity 응답"]
    mistral_answer: Annotated[str | None, "Mistral AI 응답"]
    groq_answer: Annotated[str | None, "Groq 응답"]
    cohere_answer: Annotated[str | None, "Cohere 응답"]

    raw_responses: Annotated[dict[str, str] | None, merge_dicts]
    self_summaries: Annotated[dict[str, str] | None, merge_dicts]
    raw_sources: Annotated[dict[str, str | None] | None, merge_dicts]
    openai_summary: Annotated[str | None, "OpenAI 자기 요약"]
    gemini_summary: Annotated[str | None, "Gemini 자기 요약"]
    anthropic_summary: Annotated[str | None, "Anthropic 자기 요약"]
    upstage_summary: Annotated[str | None, "Upstage 자기 요약"]
    perplexity_summary: Annotated[str | None, "Perplexity 자기 요약"]
    mistral_summary: Annotated[str | None, "Mistral 자기 요약"]
    groq_summary: Annotated[str | None, "Groq 자기 요약"]
    cohere_summary: Annotated[str | None, "Cohere 자기 요약"]

    openai_status: Annotated[dict[str, Any] | None, "OpenAI 호출 상태"]
    gemini_status: Annotated[dict[str, Any] | None, "Gemini 호출 상태"]
    anthropic_status: Annotated[dict[str, Any] | None, "Anthropic 호출 상태"]
    upstage_status: Annotated[dict[str, Any] | None, "Upstage 호출 상태"]
    perplexity_status: Annotated[dict[str, Any] | None, "Perplexity 호출 상태"]
    mistral_status: Annotated[dict[str, Any] | None, "Mistral 호출 상태"]
    groq_status: Annotated[dict[str, Any] | None, "Groq 호출 상태"]
    cohere_status: Annotated[dict[str, Any] | None, "Cohere 호출 상태"]
    summary_model: Annotated[str | None, "요약에 사용한 모델"]

    messages: Annotated[list, add_messages]


def _default_active_models() -> list[str]:
    return list(NODE_CONFIG.keys())


def _model_label(node_name: str) -> str:
    meta = NODE_CONFIG.get(node_name)
    return meta["label"] if meta else node_name


def _simplify_error_message(error: Any) -> str:
    """예외 객체에서 핵심 메시지만 추출한다."""

    def from_mapping(data: dict[str, Any]) -> str | None:
        for key in ("detail", "message", "error"):
            value = data.get(key)
            if value:
                return str(value)
        body = data.get("body")
        if isinstance(body, dict):
            nested = from_mapping(body)
            if nested:
                return nested
        elif body:
            return str(body)
        return None

    response = getattr(error, "response", None)
    if response is not None:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                text = from_mapping(payload)
                if text:
                    return text
        except Exception:
            pass
        try:
            body = response.text
            if body:
                return body.strip().splitlines()[0][:200]
        except Exception:
            pass

    body_attr = getattr(error, "body", None)
    if body_attr:
        if isinstance(body_attr, dict):
            text = from_mapping(body_attr)
            if text:
                return text
        if isinstance(body_attr, str):
            return body_attr.strip().splitlines()[0][:200]

    if hasattr(error, "args") and error.args:
        candidate = error.args[0]
        if isinstance(candidate, dict):
            text = from_mapping(candidate)
            if text:
                return text
        if isinstance(candidate, str):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    text = from_mapping(parsed)
                    if text:
                        return text
            except Exception:
                pass
            return candidate.strip().splitlines()[0][:200]

    return str(error).strip().splitlines()[0][:200]


class Answer(BaseModel):
    """LCEL 체인 출력 스키마."""

    content: str = Field(..., description="짧게 요약된 답변")
    source: str | None = Field(default=None, description="출처 URL 또는 참고 정보(옵션)")


def _build_prompt() -> ChatPromptTemplate:
    parser = PydanticOutputParser(pydantic_object=Answer)
    instructions = parser.get_format_instructions()
    system = (
        "사용자 질문에 대해 가능한 한 짧게 핵심만 답변하세요. "
        "5문장 이하, 400자 이하로 요약하며 모르면 모른다고 답변합니다.\n"
        "{format_instructions}"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", "{question}"),
        ]
    ).partial(format_instructions=instructions)


def _extract_source(extras: dict[str, Any] | None) -> str | None:
    """추가 메타에서 출처 URL을 추출한다."""

    if not extras:
        return None
    citations = extras.get("citations")
    if isinstance(citations, list) and citations:
        first = citations[0]
        if isinstance(first, str):
            return first
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
    try:
        parsed: Answer = parser.parse(raw_text)
        content = parsed.content
        source = parsed.source or _extract_source(getattr(parsed, "model_extra", None))
    except Exception:
        content = raw_text
        source = _extract_source(getattr(response, "response_metadata", None))
    return content, source, status


def build_status_from_response(
    response: Any, default_status: int = 200, detail: str = "success"
) -> dict[str, Any]:
    """LLM 응답 객체에서 상태 메타데이터를 추출한다."""

    metadata = getattr(response, "response_metadata", None) or {}
    status = metadata.get("status_code") or metadata.get("status") or metadata.get("http_status")
    detail_text = metadata.get("finish_reason") or metadata.get("reason") or detail
    return {"status": status or default_status, "detail": detail_text}


def build_status_from_error(error: Exception) -> dict[str, Any]:
    """예외 객체를 API 상태 표현으로 변환한다."""

    status = cast(int | None, getattr(error, "status_code", None))
    if status is None:
        response = getattr(error, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    detail = _simplify_error_message(error)
    return {"status": status or "error", "detail": detail}


def format_response_message(label: str, payload: Any) -> tuple[str, str]:
    """메시지 로그에 저장할 간단한 (role, content) 튜플을 생성한다."""

    return ("assistant", f"[{label}] {payload}")


def init_question(state: GraphState) -> GraphState:
    """그래프 초기 상태를 검증하고 기본 메시지를 설정한다."""

    question = state.get("question")
    if not question:
        raise ValueError("질문이 비어 있습니다.")

    max_turns = state.get("max_turns") or DEFAULT_MAX_TURNS
    active_models = state.get("active_models") or list(NODE_CONFIG.keys())
    current_inputs = state.get("current_inputs") or {
        NODE_CONFIG[node]["label"]: question for node in active_models
    }
    history = state.get("conversation_history")
    if not history:
        history = [{"role": "user", "content": question}]
    turn_value = state.get("turn") or 1

    logger.debug("질문 초기화: %s", _preview(question))
    return GraphState(
        question=question,
        max_turns=max_turns,
        turn=turn_value,
        conversation_history=history,
        history_summary=state.get("history_summary"),
        current_inputs=current_inputs,
        active_models=active_models,
        raw_responses=state.get("raw_responses") or {},
        self_summaries=state.get("self_summaries") or {},
        messages=state.get("messages") or [("user", question)],
    )


async def _ainvoke(llm: Any, question: str) -> Any:
    """주어진 LLM에서 비동기 호출을 수행한다."""

    if hasattr(llm, "ainvoke"):
        return await llm.ainvoke(question)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, llm.invoke, question)


async def call_openai(state: GraphState) -> GraphState:
    """OpenAI 모델을 호출하고 응답/상태를 반환한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt_input = inputs.get("OpenAI") or question
    logger.debug("OpenAI 호출 시작")
    try:
        settings = get_settings()
        llm = ChatOpenAI(model=settings.model_openai)
        content, source, status = await _invoke_parsed(llm, prompt_input, "OpenAI")
        summary = await _summarize_content(llm, content, "OpenAI")
        logger.info("OpenAI 응답 완료")
        return GraphState(
            openai_answer=content,
            openai_status=status,
            openai_summary=summary,
            raw_responses={"OpenAI": content},
            raw_sources={"OpenAI": source},
            self_summaries={"OpenAI": summary},
            messages=[format_response_message("OpenAI", content)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("OpenAI 호출 실패: %s", exc)
        return GraphState(
            openai_status=status,
            messages=[format_response_message("OpenAI 오류", exc)],
        )


async def call_gemini(state: GraphState) -> GraphState:
    """Google Gemini 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt_input = inputs.get("Gemini") or question
    logger.debug("Gemini 호출 시작")
    try:
        settings = get_settings()
        llm = ChatGoogleGenerativeAI(model=settings.model_gemini, temperature=0)
        content, source, status = await _invoke_parsed(llm, prompt_input, "Gemini")
        summary = await _summarize_content(llm, content, "Gemini")
        logger.info("Gemini 응답 완료")
        return GraphState(
            gemini_answer=content,
            gemini_status=status,
            gemini_summary=summary,
            raw_responses={"Gemini": content},
            raw_sources={"Gemini": source},
            self_summaries={"Gemini": summary},
            messages=[format_response_message("Gemini", content)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Gemini 호출 실패: %s", exc)
        return GraphState(
            gemini_status=status,
            messages=[format_response_message("Gemini 오류", exc)],
        )


async def call_anthropic(state: GraphState) -> GraphState:
    """Anthropic Claude 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt_input = inputs.get("Anthropic") or question
    logger.debug("Anthropic 호출 시작")
    try:
        settings = get_settings()
        llm = ChatAnthropic(model=settings.model_anthropic, temperature=0)
        content, source, status = await _invoke_parsed(llm, prompt_input, "Anthropic")
        summary = await _summarize_content(llm, content, "Anthropic")
        logger.info("Anthropic 응답 완료")
        return GraphState(
            anthropic_answer=content,
            anthropic_status=status,
            anthropic_summary=summary,
            raw_responses={"Anthropic": content},
            raw_sources={"Anthropic": source},
            self_summaries={"Anthropic": summary},
            messages=[format_response_message("Anthropic", content)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Anthropic 호출 실패: %s", exc)
        return GraphState(
            anthropic_status=status,
            messages=[format_response_message("Anthropic 오류", exc)],
        )


async def call_upstage(state: GraphState) -> GraphState:
    """Upstage Solar 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt_input = inputs.get("Upstage") or question
    logger.debug("Upstage 호출 시작")
    try:
        settings = get_settings()
        llm = ChatUpstage(model=settings.model_upstage)
        content, source, status = await _invoke_parsed(llm, prompt_input, "Upstage")
        summary = await _summarize_content(llm, content, "Upstage")
        logger.info("Upstage 응답 완료")
        return GraphState(
            upstage_answer=content,
            upstage_status=status,
            upstage_summary=summary,
            raw_responses={"Upstage": content},
            raw_sources={"Upstage": source},
            self_summaries={"Upstage": summary},
            messages=[format_response_message("Upstage", content)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Upstage 호출 실패: %s", exc)
        return GraphState(
            upstage_status=status,
            messages=[format_response_message("Upstage 오류", exc)],
        )


async def call_perplexity(state: GraphState) -> GraphState:
    """Perplexity Sonar 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt_input = inputs.get("Perplexity") or question
    logger.debug("Perplexity 호출 시작")
    pplx_api_key = os.getenv("PPLX_API_KEY")
    if not pplx_api_key:
        error = RuntimeError("PPLX_API_KEY is missing")
        status = build_status_from_error(error)
        return GraphState(
            perplexity_status=status,
            messages=[format_response_message("Perplexity 오류", error)],
        )
    try:
        settings = get_settings()
        llm = ChatPerplexity(temperature=0, model=settings.model_perplexity, pplx_api_key=pplx_api_key)
        content, source, status = await _invoke_parsed(llm, prompt_input, "Perplexity")
        summary = await _summarize_content(llm, content, "Perplexity")
        logger.info("Perplexity 응답 완료")
        msg = content if not source else f"{content} (src: {source})"
        return GraphState(
            perplexity_answer=content,
            perplexity_status=status,
            perplexity_summary=summary,
            raw_responses={"Perplexity": content},
            raw_sources={"Perplexity": source},
            self_summaries={"Perplexity": summary},
            messages=[format_response_message("Perplexity", msg)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Perplexity 호출 실패: %s", exc)
        return GraphState(
            perplexity_status=status,
            messages=[format_response_message("Perplexity 오류", exc)],
        )


async def call_mistral(state: GraphState) -> GraphState:
    """Mistral AI 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Mistral") or question
    if ChatMistralAI is None:
        error = RuntimeError("langchain-mistralai 패키지가 설치되어 있지 않습니다.")
        logger.warning("Mistral AI 사용 불가: %s", error)
        status = build_status_from_error(error)
        return GraphState(
            mistral_status=status,
            messages=[format_response_message("Mistral 오류", error)],
        )
    try:
        settings = get_settings()
        llm = ChatMistralAI(model=settings.model_mistral, temperature=0)
        content, source, status = await _invoke_parsed(llm, prompt, "Mistral")
        summary = await _summarize_content(llm, content, "Mistral")
        logger.info("Mistral 응답 완료")
        return GraphState(
            mistral_answer=content,
            mistral_status=status,
            mistral_summary=summary,
            raw_responses={"Mistral": content},
            raw_sources={"Mistral": source},
            self_summaries={"Mistral": summary},
            messages=[format_response_message("Mistral", content)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Mistral 호출 실패: %s", exc)
        return GraphState(
            mistral_status=status,
            messages=[format_response_message("Mistral 오류", exc)],
        )


async def call_groq(state: GraphState) -> GraphState:
    """Groq 기반 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Groq") or question
    if ChatGroq is None:
        error = RuntimeError("langchain-groq 패키지가 설치되어 있지 않습니다.")
        logger.warning("Groq 사용 불가: %s", error)
        status = build_status_from_error(error)
        return GraphState(
            groq_status=status,
            messages=[format_response_message("Groq 오류", error)],
        )
    try:
        settings = get_settings()
        llm = ChatGroq(model=settings.model_groq, temperature=0)
        content, source, status = await _invoke_parsed(llm, prompt, "Groq")
        summary = await _summarize_content(llm, content, "Groq")
        logger.info("Groq 응답 완료")
        return GraphState(
            groq_answer=content,
            groq_status=status,
            groq_summary=summary,
            raw_responses={"Groq": content},
            raw_sources={"Groq": source},
            self_summaries={"Groq": summary},
            messages=[format_response_message("Groq", content)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Groq 호출 실패: %s", exc)
        return GraphState(
            groq_status=status,
            messages=[format_response_message("Groq 오류", exc)],
        )


async def call_cohere(state: GraphState) -> GraphState:
    """Cohere Command 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Cohere") or question
    if ChatCohere is None:
        error = RuntimeError("langchain-cohere 패키지가 설치되어 있지 않습니다.")
        logger.warning("Cohere 사용 불가: %s", error)
        status = build_status_from_error(error)
        return GraphState(
            cohere_status=status,
            messages=[format_response_message("Cohere 오류", error)],
        )
    try:
        settings = get_settings()
        llm = ChatCohere(model=settings.model_cohere, temperature=0)
        content, source, status = await _invoke_parsed(llm, prompt, "Cohere")
        summary = await _summarize_content(llm, content, "Cohere")
        logger.info("Cohere 응답 완료")
        return GraphState(
            cohere_answer=content,
            cohere_status=status,
            cohere_summary=summary,
            raw_responses={"Cohere": content},
            raw_sources={"Cohere": source},
            self_summaries={"Cohere": summary},
            messages=[format_response_message("Cohere", content)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Cohere 호출 실패: %s", exc)
        return GraphState(
            cohere_status=status,
            messages=[format_response_message("Cohere 오류", exc)],
        )


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

    question = state.get("question")
    if not question:
        raise ValueError("질문이 비어 있습니다.")
    active_models = state.get("active_models") or _default_active_models()
    logger.info("LLM fan-out 실행: %s", ", ".join(active_models))
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
    "build_status_from_response",
    "build_status_from_error",
    "format_response_message",
]
