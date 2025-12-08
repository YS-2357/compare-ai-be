# app/services/langgraph/

- LangGraph 워크플로우 패키지.
- `workflow.py`: 그래프 생성/컴파일, 이벤트 스트림, `model_overrides`와 `bypass_turn_limit` 처리.
- `nodes.py`: call_* 노드(LCEL 체인 + Pydantic 파서) — MODEL_* 기본값을 바탕으로 공급자별 오버라이드 반영.
- `helpers.py`: 프롬프트 빌더/구조화 파서(한국어 장문 응답).
- `llm_registry.py`: LLM 클라이언트 + LangSmith UUID 도우미.
- `summaries.py`: 답변 요약 유틸.
- `__init__.py`: `stream_graph`, `DEFAULT_MAX_TURNS` 등을 외부에 노출.

이벤트 페이로드
- `partial`: `model`, `answer`, `status`, `source?`, `messages`, `turn`, `elapsed_ms`.
- `summary`: `answers`, `api_status`, `sources`, `durations_ms`, `messages`, `order`, `primary_model`, `usage_limit/remaining`, `model_overrides` 등.
