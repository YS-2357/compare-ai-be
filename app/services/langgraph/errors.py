"""LangGraph/LLM 에러 및 상태 헬퍼."""

from __future__ import annotations

import json
from typing import Any, cast

from app.logger import get_logger

logger = get_logger(__name__)


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


__all__ = ["_simplify_error_message", "build_status_from_response", "build_status_from_error"]
