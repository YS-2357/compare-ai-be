"""LangGraph 워크플로우를 구성하고 스트리밍 이벤트를 노출한다."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.logger import get_logger

from .llm_registry import create_uuid
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
settings_cache = get_settings()


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


async def stream_graph(
    question: str, *, turn: int = 1, max_turns: int | None = None, history: list[dict[str, str]] | None = None
) -> AsyncIterator[dict[str, Any]]:
    """질문을 받아 LangGraph 워크플로우에서 발생하는 이벤트를 스트리밍한다."""

    if not question or not question.strip():
        raise ValueError("질문을 입력해주세요.")

    resolved_max_turns = max_turns or settings_cache.max_turns_default
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
    # 히스토리 → LangGraph 상태용 메시지로 변환
    user_messages = []
    model_messages: dict[str, list] = {}
    for item in history or []:
        role = item.get("role", "user")
        content = item.get("content", "")
        model_label = item.get("model")
        if not content:
            continue
        if role == "assistant" and model_label:
            msgs = model_messages.get(model_label, [])
            msgs.append((role, content))
            model_messages[model_label] = msgs
        else:
            user_messages.append((role, content))
    user_messages.append(("user", base_question))
    state_inputs: GraphState = {
        "max_turns": resolved_max_turns,
        "turn": turn,
        "active_models": active_models,
        "user_messages": user_messages,
        "model_messages": model_messages,
        "model_summaries": {},
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
                model_label = meta["label"]
                model_msgs = (state.get("model_messages") or {}).get(model_label, [])
                # 마지막 assistant 발화를 answer로 사용
                answer = None
                for msg in reversed(model_msgs):
                    if isinstance(msg, (list, tuple)) and len(msg) == 2 and msg[0] == "assistant":
                        answer = msg[1]
                        break
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        answer = msg.get("content")
                        break
                if answer is None:
                    answer = (state.get("raw_responses") or {}).get(model_label)
                api_status = (state.get("api_status") or {}).get(model_label) or {}
                yield {
                    "model": model_label,
                    "node": node_name,
                    "answer": answer,
                    "status": api_status,
                    "source": (state.get("raw_sources") or {}).get(model_label),
                    "messages": _normalize_messages(model_msgs),
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
