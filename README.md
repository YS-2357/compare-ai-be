# Compare-AI (Backend)

FastAPI 기반 멀티 LLM 비교 API (프런트는 별도 레포 `compare-ai-fe`)  
> **최종 업데이트: 2025-12-03** — 스트리밍 순차 표시(모델별), 프롬프트 영어화/응답은 한국어, Upstash 필수 + `/usage` 조회, 기본 모델 gpt-4o-mini

## 📋 프로젝트 개요

5개의 주요 LLM API(OpenAI, Google Gemini, Anthropic Claude, Upstage Solar, Perplexity)를 병렬로 호출하여 동일한 질문에 대한 각 모델의 응답을 비교할 수 있는 웹 애플리케이션입니다.

## 🏗️ 아키텍처

- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: 별도 레포 `compare-ai-fe`(예: Next.js + Supabase Auth)
- **워크플로우**: LangGraph (병렬 실행)
- **추적/로깅**: LangSmith
- **레이트리밋**: Upstash Redis(필수), 일일 호출 제한 조회/차단
- **배포**: Render/Vercel 등 서버리스·컨테이너, HTTPS 기본 (배포 편의를 위해 의존성/파이썬 버전 고정)

## 🚀 빠른 시작 (백엔드)

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

### 3. 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

서버가 시작되면:
- **FastAPI**: http://127.0.0.1:8000 (클라우드에서는 제공 도메인 사용)

## 📁 프로젝트 구조

```
api-test/
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
│   └── ui/                        # Streamlit 로컬 UI (개발용)
├── scripts/run_app.py             # FastAPI+Streamlit 실행 스크립트
├── main.py                        # scripts/run_app.py 래퍼(또는 APP_MODE=api)
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

### 4. Frontend (별도 레포 `compare-ai-fe`)
- Supabase Auth로 로그인/회원가입 후 JWT 획득
- JWT를 `Authorization: Bearer <token>` 헤더에 담아 이 백엔드 `/api/ask` 호출
- `.env`에 `ADMIN_EMAIL=youngsunx20@gmail.com`처럼 지정한 이메일로 로그인하면 `/usage` 응답의 `remaining`이 `null`이 되며 일일 제한 없이 사용할 수 있다.
- Streamlit UI(로컬 실행 기준)에서는 사이드바에서 OpenAI/Gemini/Claude 등 각 LLM의 모델을 선택할 수 있으며, 선택값은 API 요청 시 `models` 필드로 전달되어 LangGraph 실행에 반영된다.

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

## 📝 변경 이력

상세한 날짜별 변경 이력은 [`docs/changelog/`](docs/changelog/) 디렉토리를 참조하세요.

## 🛠️ 개발 가이드

### 노트북 기준 개발
- `notebooks/api_langgraph_test.ipynb`가 기준 구현
- 노트북에서 검증된 코드만 프로덕션 코드로 이식
- LangSmith 로깅 설정은 노트북 기준 유지

### 코드 수정 시 주의사항
1. 노트북 파일은 수정하지 않음 (기준 유지)
2. 모델명은 노트북과 동일하게 유지
3. LangSmith 프로젝트명: `API-LangGraph-Test`
4. UUID v7 사용 (LangSmith 권장)

## ⚠️ 알려진 이슈

### 1. 응답 시간
- 5개 LLM을 병렬로 호출하므로 1~2분 소요
- 프런트엔드에서 스트리밍 응답을 받을 때 타임아웃을 충분히 길게 설정하세요.

### 2. 패키지 호환성
- `numpy` 버전 충돌 가능 → 가상환경 사용 필수
- `langchain-upstage`의 의존성 버전 주의

### 3. 사용량 표시 버그 (UI)
- Streamlit UI에서 재로그인 직후 남은 횟수가 항상 3으로 보일 수 있습니다(실제 서버 제한은 Upstash/캐시에 따라 정상 적용). `/api/ask` 호출 후 내려오는 헤더/summary 값으로 즉시 동기화하는 패치를 예정 중입니다.

## 📄 라이선스

MIT License

## 👥 기여

버그 리포트 및 기능 제안은 이슈로 등록해주세요.
