"""LangGraph 워크플로우를 구성하고 스트리밍 이벤트를 노출한다."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.logger import get_logger

from .llm_registry import ChatOpenAI, create_uuid
from .nodes import (
    DEFAULT_MAX_TURNS,
    NODE_CONFIG,
    _ainvoke,
    _preview,
    call_anthropic,
    call_cohere,
    call_gemini,
    call_groq,
    call_mistral,
    call_openai,
    call_perplexity,
    call_upstage,
    dispatch_llm_calls,
    GraphState,
    init_question,
)

logger = get_logger(__name__)


def build_workflow():
    """StateGraph를 구성하고 LangGraph 앱으로 컴파일한다."""

    logger.debug("LangGraph 워크플로우 컴파일 시작")
    workflow = StateGraph(GraphState)
    workflow.add_node("init_question", init_question)
    workflow.add_node("call_openai", call_openai)
    workflow.add_node("call_gemini", call_gemini)
    workflow.add_node("call_anthropic", call_anthropic)
    workflow.add_node("call_upstage", call_upstage)
    workflow.add_node("call_perplexity", call_perplexity)
    workflow.add_node("call_mistral", call_mistral)
    workflow.add_node("call_groq", call_groq)
    workflow.add_node("call_cohere", call_cohere)

    workflow.add_conditional_edges("init_question", dispatch_llm_calls)

    workflow.add_edge("call_openai", END)
    workflow.add_edge("call_gemini", END)
    workflow.add_edge("call_anthropic", END)
    workflow.add_edge("call_upstage", END)
    workflow.add_edge("call_perplexity", END)
    workflow.add_edge("call_mistral", END)
    workflow.add_edge("call_groq", END)
    workflow.add_edge("call_cohere", END)

    workflow.set_entry_point("init_question")
    compiled = workflow.compile()
    logger.info("LangGraph 워크플로우 컴파일 완료")
    return compiled


_app = None


def get_app():
    """싱글턴 형태로 컴파일된 LangGraph 앱을 반환한다."""

    global _app
    if _app is None:
        _app = build_workflow()
    return _app


async def _summarize_history(history: list[dict[str, str]] | None, limit: int = 400) -> str | None:
    """과거 대화 이력을 2문장 이내로 요약한다."""

    if not history:
        return None

    text_lines = [f"{item.get('role')}: {item.get('content')}" for item in history if item.get("content")]
    history_text = "\n".join(text_lines)
    prompt = (
        "다음 대화 이력을 2문장 이하, 400자 이내로 요약하세요. 핵심 논점만 남기고 세부사항은 생략합니다.\n\n"
        f"{history_text}"
    )
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    try:
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return str(content)[:limit]
    except Exception as exc:  # pragma: no cover - 요약 실패 시 안전 폴백
        logger.warning("대화 요약 실패, 원본을 절단해 사용합니다: %s", exc)
        compact = " ".join(history_text.split())
        return compact[:limit]


def _build_current_inputs(question: str, active_models: list[str]) -> dict[str, str]:
    """현 턴에서 사용할 모델별 입력 프롬프트를 생성한다."""

    prompts: dict[str, str] = {}
    for node_name in active_models:
        label = NODE_CONFIG[node_name]["label"]
        sections = [question]
        sections.append(
            "사용자 질문에 최신 답변을 제시하세요. 필요하면 이전 맥락을 반영하세요."
        )
        sections.append("응답은 5문장 이하, 600자 이내로 간결하게 작성하세요.")
        prompts[label] = "\n\n".join([part for part in sections if part])
    return prompts


def _normalize_messages(messages: list | None) -> list[dict[str, str]]:
    """Streamlit 표시를 위해 메시지를 표준화한다."""

    normalized: list[dict[str, str]] = []
    for message in messages or []:
        if isinstance(message, (list, tuple)) and len(message) == 2:
            role, content = message
            normalized.append({"role": str(role), "content": str(content)})
        else:
            normalized.append({"role": "system", "content": str(message)})
    return normalized


def _extend_unique_messages(
    target: list[dict[str, str]], new_messages: list[dict[str, str]] | None, seen: set[tuple[str, str]]
) -> None:
    """중복 없이 메시지를 추가한다."""

    for message in new_messages or []:
        role = str(message.get("role"))
        content = str(message.get("content"))
        key = (role, content)
        if key in seen:
            continue
        seen.add(key)
        target.append({"role": role, "content": content})


async def stream_graph(
    question: str, *, turn: int = 1, max_turns: int | None = None, history: list[dict[str, str]] | None = None
) -> AsyncIterator[dict[str, Any]]:
    """질문을 받아 LangGraph 워크플로우에서 발생하는 이벤트를 스트리밍한다."""

    if not question or not question.strip():
        raise ValueError("질문을 입력해주세요.")

    resolved_max_turns = max_turns or DEFAULT_MAX_TURNS
    if turn > resolved_max_turns:
        warning = f"최대 턴({resolved_max_turns})을 초과했습니다. 새 질문으로 시작해주세요."
        logger.warning("턴 초과 - 실행 중단: turn=%s, max=%s", turn, resolved_max_turns)
        yield {"type": "error", "message": warning, "node": None, "model": None, "turn": turn}
        return

    logger.info("LangGraph 스트림 실행: %s", _preview(question))
    base_question = question.strip()
    app = get_app()
    start_time = time.perf_counter()
    active_models = list(NODE_CONFIG.keys())
    current_inputs = _build_current_inputs(base_question, active_models)
    conversation_history = list(history or [])
    conversation_history.append({"role": "user", "content": base_question})
    state_inputs: GraphState = {
        "question": base_question,
        "max_turns": resolved_max_turns,
        "turn": turn,
        "conversation_history": conversation_history,
        "current_inputs": current_inputs,
        "active_models": active_models,
    }

    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": str(create_uuid())})
    try:
        async for event in app.astream(state_inputs, config=config):
            turn_index = state_inputs.get("turn") or 1
            for node_name, state in event.items():
                if node_name == "__end__":
                    continue
                if node_name not in NODE_CONFIG:
                    continue
                meta = NODE_CONFIG[node_name]
                logger.debug("이벤트 수신: %s (turn=%s)", meta["label"], turn_index)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                yield {
                    "model": meta["label"],
                    "node": node_name,
                    "answer": state.get(meta["answer_key"]),
                    "status": state.get(meta["status_key"]) or {},
                    "messages": _normalize_messages(state.get("messages")),
                    "type": "partial",
                    "turn": turn_index,
                    "elapsed_ms": elapsed_ms,
                }
    except Exception as exc:
        logger.error("LangGraph 스트림 오류: %s", exc)
        yield {
            "type": "error",
            "message": str(exc),
            "node": None,
            "model": None,
            "turn": turn,
        }


__all__ = ["stream_graph", "DEFAULT_MAX_TURNS", "build_workflow"]
