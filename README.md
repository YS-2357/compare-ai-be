# Compare-AI (FastAPI + Streamlit 단일 레포)

FastAPI 백엔드와 Streamlit UI가 한 레포(`compare-ai`)에 함께 있으며, 단일 커맨드로 로컬 실행합니다.  
> **최종 업데이트: 2025-12-08** — FastAPI `/docs` 예시/설명 보강, 모델 오버라이드/관리자 우회 문서화, Render 단일 레포 배포

## 📋 프로젝트 개요

5개 이상 주요 LLM API(OpenAI, Google Gemini, Anthropic Claude, Upstage Solar, Perplexity 등)를 병렬 호출해 질문별 응답을 비교합니다. 로컬 실행 시 FastAPI와 Streamlit을 동시에 띄우며, Render 배포 시 동일 레포를 사용해 BE/FE 서비스를 각각 구성합니다(명령만 다름).

## 🏗️ 아키텍처

- **백엔드**: FastAPI (Python 3.11+), LangGraph 기반 스트리밍
- **프런트(UI)**: Streamlit (같은 레포 `app/ui/streamlit_app.py`)
- **워크플로우**: LangGraph (병렬 실행)
- **추적/로깅**: LangSmith
- **레이트리밋**: Upstash Redis(필수), `/usage` 조회/차단
- **배포**: Render (동일 레포에서 FastAPI/Streamlit 서비스를 각각 실행)

## 🚀 빠른 시작 (단일 커맨드)

### 1. 환경 설정

```bash
# 가상환경 생성 (이미 있으면 생략)
python -m venv .venv

# 가상환경 활성화
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env` 파일에 필요한 API 키 설정(예시):

```env
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
ANTHROPIC_API_KEY=your-anthropic-key
UPSTAGE_API_KEY=your-upstage-key
PPLX_API_KEY=your-perplexity-key
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=yout-project-name

# Supabase Auth (JWT 검증)
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
# JWKS는 .well-known/jwks.json 경로를 권장
SUPABASE_JWKS_URL=https://xxxx.supabase.co/auth/v1/.well-known/jwks.json
SUPABASE_JWT_AUD=authenticated

# Upstash Redis (일일 사용량 제한, 기본 3회, 필수)
UPSTASH_REDIS_URL=...
UPSTASH_REDIS_TOKEN=...
DAILY_USAGE_LIMIT=3

# 관리자 계정 (인증/레이트리밋 무시)
ADMIN_BYPASS_TOKEN=choose-a-strong-token
```

### 3. 실행 (FastAPI + Streamlit 동시)

```bash
python main.py
```

서버가 시작되면:
- **FastAPI**: http://127.0.0.1:8000 (`APP_MODE=api`로 설정하면 FastAPI만 단독 실행)
- **Streamlit**: http://127.0.0.1:8501

## 📁 프로젝트 구조

```
compare-ai/
├── app/
│   ├── config.py                  # Pydantic Settings (환경변수 관리)
│   ├── main.py                    # FastAPI 앱 팩토리
│   ├── api/                       # 라우터/스키마/의존성
│   │   ├── routes.py              # /health, /api/ask
│   │   ├── auth_routes.py         # /auth/register, /auth/login
│   │   ├── deps.py                # Depends: get_current_user 등
│   │   └── schemas/               # ask.py, auth.py
│   ├── auth/                      # Supabase 검증/클라이언트
│   ├── rate_limit/                # Upstash 클라이언트/Depends
│   ├── services/langgraph/        # LangGraph 워크플로우 분할
│   └── ui/                        # Streamlit UI
├── scripts/run_app.py             # FastAPI+Streamlit 실행 스크립트 (main.py에서 호출)
├── main.py                        # APP_MODE에 따라 api만 또는 둘 다 실행
├── notebooks/
│   └── api_langgraph_test.ipynb
├── docs/
│   ├── changelog/
│   └── development/
└── .env
```

## 🔧 주요 기능

### 1. 멀티 LLM 병렬 호출
- OpenAI GPT-5-nano
- Google Gemini 2.5 Flash Lite
- Anthropic Claude Haiku 4.5
- Upstage Solar Mini
- Perplexity Sonar

### 2. LangGraph 워크플로우
- 질문 초기화 → 5개 LLM 병렬 호출 → 응답 수집 및 요약
- 각 LLM의 성공/실패 상태 추적
- 에러 발생 시에도 다른 모델의 응답은 정상 수집

### 3. LangSmith 추적
- 모든 LLM 호출이 LangSmith에 자동 기록
- 프로젝트: `API-LangGraph-Test`
- 토큰 사용량, 응답 시간, 에러 로그 추적

### 4. Frontend (동일 레포 Streamlit)
- Streamlit UI 사이드바에서 모델 선택 → `models` 필드로 API에 전달되어 LangGraph에 반영
- Supabase Auth JWT를 `Authorization: Bearer <token>`으로 FastAPI에 전달
- 관리자 이메일(`ADMIN_EMAIL`) 로그인 시 `/usage` 응답 `remaining = null`로 우회 적용

## 🔗 API 엔드포인트

### Health Check
```bash
GET /health
```

### 사용량 조회
```bash
GET /usage
```
JWT 필요. `ADMIN_EMAIL`로 지정된 계정이면 `remaining`이 `null`로 내려가며 제한 없이 사용.

### 질문 처리
```bash
POST /api/ask
Content-Type: application/json

{
  "question": "당신의 질문을 입력하세요",
  "models": {
    "openai": "gpt-4o-nano",
    "gemini": "gemini-2.0-flash"
  }
}
```

**응답 형식**

- 스트리밍 방식(Newline Delimited JSON)
  - `type: "partial"` 이벤트가 모델별 완료 순서대로 도착합니다.
  - 마지막에는 `type: "summary"` 이벤트가 전체 결과(`question`, `answers`, `api_status`, `messages`)를 포함해 전달됩니다.

예시 스트림:
```
{"type":"partial","model":"OpenAI","answer":"...","status":{"status":200,"detail":"stop"}}
{"type":"partial","model":"Gemini","answer":"...","status":{"status":200,"detail":"stop"}}
...
{"type":"summary","result":{"question":"AI란 무엇인가?","answers":{...},"api_status":{...},"messages":[...]}}
```

**사용량 헤더(있을 경우)**
- `X-Usage-Limit`: 일일 한도 (`DAILY_USAGE_LIMIT`, 기본 3)
- `X-Usage-Remaining`: 이번 호출 기준 남은 횟수 (Upstash 장애 시 503/429 반환, 폴백 없음)

## 📝 참고

- 변경 이력과 최신 이슈/해결 현황: `docs/changelog/`, `docs/development/`
- 실시간 동작/스키마: FastAPI `/docs`(Swagger)와 코드 주석을 우선 확인

## 📄 라이선스

MIT License

## 👥 기여

버그 리포트 및 기능 제안은 이슈로 등록해주세요.
