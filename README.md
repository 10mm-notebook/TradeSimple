# TradeSimple - 수입업무 간편화 AI 도우미

> HS코드 분류 및 관세 계산 AI 에이전트 시스템  
> K Intelligence 해커톤 2025 출품작

---

## 📋 목차

- [개요](#개요)
- [핵심 특징](#핵심-특징)
- [에이전트 아키텍처](#에이전트-아키텍처)
- [기술 스택](#기술-스택)
- [설치 및 실행](#설치-및-실행)
- [API 문서](#api-문서)
- [프로젝트 구조](#프로젝트-구조)

---

## 개요

**TradeSimple**은 LangGraph 기반 AI 에이전트로 복잡한 수입 업무를 자동화합니다.

### 해결하는 문제

| 기존 방식 | TradeSimple |
|-----------|-------------|
| HS 코드를 직접 알아야 검색 가능 | 물품 설명만으로 자동 분류 |
| 여러 사이트에서 환율/관세율 조회 | 실시간 자동 조회 |
| 수동 계산 및 보고서 작성 | 원클릭 보고서 생성 (PDF/Word/Excel) |
| AI 분류 오류 시 수정 불가 | **Human-in-the-Loop로 사용자가 직접 검증·선택** |
| 평균 30분+ 소요 | 1분 내 완료 |

### 입력 예시

```
미국에서 스마트워치 100개를 개당 300달러에 수입하려고 합니다.
```

### 출력 결과

- HS 코드: `8517.62-9090` (무선통신기기) ← **사용자가 3개 후보 중 선택**
- 적용 관세율: 0% (한-미 FTA)
- 총 예상 비용: 약 44,550,000원
- PDF/Word/Excel 보고서 자동 생성

---

## 핵심 특징

### 1. LangGraph 기반 멀티 에이전트 시스템

**Supervisor 패턴**으로 3개의 전문 에이전트를 중앙에서 오케스트레이션합니다.

```
                    ┌─────────────────┐
                    │   Supervisor    │
                    │   (LangGraph)   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │  HS Code      │ │     Tax       │ │    Report     │
    │  Finder       │ │  Calculator   │ │    Writer     │
    │  (ReAct)      │ │               │ │  (Parallel)   │
    └───────────────┘ └───────────────┘ └───────────────┘
```

### 2. ReAct 패턴 에이전트

**LLM이 스스로 도구를 선택**하고 실행하는 자율적 에이전트입니다.

```python
# HS Code Finder - ReAct 루프 예시
Thought: "노트북"을 검색해야 한다
Action: hs_code_search("노트북")
Observation: [8471.30 휴대용 자동자료처리기계, 8471.41 ...]
Thought: 8471.30이 가장 적합해 보인다. 관세율을 확인하자
Action: tariff_search_by_hs_code("8471.30-0000")
Observation: 기본세율 0%, 품명: 휴대용 자동자료처리기계
Thought: 결과를 반환한다
Final Answer: HS 코드 8471.30-0000, 관세율 0%
```

- **검색어 확장**: LLM이 재질/용도/형태 정보를 보강한 검색 문장으로 RAG 정확도 개선
- **검색근거 인용**: 관세청 DB의 **원문 문장 그대로 인용**하여 근거 제시

### 3. Human-in-the-Loop

**HS 코드 분류는 수입 비용에 직접적인 영향을 미치는 핵심 정보입니다.**

AI가 완전 자동으로 처리하는 대신, **프로세스 중간에 사람의 판단을 반영**합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Human-in-the-Loop 프로세스                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1️⃣ 사용자 입력                                                │
│      "중국에서 노트북 50대를 개당 800위안에 수입"                  │
│                         │                                       │
│                         ▼                                       │
│   2️⃣ HS Code Finder 에이전트가 후보 3개 검색                    │
│      ┌────────────┬────────────┬────────────┐                  │
│      │ 8471.30    │ 8471.41    │ 8471.49    │                  │
│      │ 휴대용 PC  │ 디스플레이 │ 기타 자동   │                  │
│      │ [선택]     │ [선택]     │ 처리기계   │                  │
│      │            │            │ [선택]     │                  │
│      └────────────┴────────────┴────────────┘                  │
│                         │                                       │
│                         ▼                                       │
│   3️⃣ 사용자가 가장 적합한 코드 선택                             │
│                         │                                       │
│                         ▼                                       │
│   4️⃣ Tax Calculator + Report Writer 에이전트 실행              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| 완전 자동화 | Human-in-the-Loop |
|------------|-------------------|
| AI가 단일 결과 제시 | AI가 후보 3개 제시, 사용자가 선택 |
| 오류 발생 시 전체 결과가 틀림 | 사용자 검증으로 오류 방지 |
| 블랙박스 (왜 이 코드인지 모름) | 각 후보별 분류 근거 설명 |

### 4. 비동기 병렬 처리

독립적인 작업을 `asyncio.gather`로 동시 실행하여 응답 시간을 단축합니다.

```python
# 병렬 실행 예시
hs_result, exchange_result = await asyncio.gather(
    fetch_hs_code_candidates(),  # HS 코드 검색
    fetch_exchange_rate()         # 환율 조회
)
```

- **HS 코드 검색 + 환율 조회** → 병렬 (1단계)
- **PDF + Word + Excel 보고서 생성** → 병렬 (2단계)

### 5. API 중심 아키텍처

UI와 비즈니스 로직을 완전히 분리하여 확장성을 확보했습니다.

```
[Streamlit UI] ──HTTP──> [FastAPI API] ──> [LangGraph Agents]
    :8501                    :8000
```

---

## 에이전트 아키텍처

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TradeSimple                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐          ┌─────────────────────────────────────┐  │
│  │   Streamlit     │  HTTP    │          FastAPI API Server         │  │
│  │    Web UI       │─────────►│              :8000                  │  │
│  │   :8501         │          │  /api/v1/analyze, /api/v1/calculate │  │
│  └─────────────────┘          └─────────────────┬───────────────────┘  │
│                                                 │                       │
│                                                 ▼                       │
│                               ┌─────────────────────────────────────┐  │
│                               │        LangGraph Supervisor         │  │
│                               │    (StateGraph + Conditional Edges) │  │
│                               └─────────────────┬───────────────────┘  │
│                                                 │                       │
│              ┌──────────────────────────────────┼──────────────────┐   │
│              ▼                                  ▼                  ▼   │
│     ┌─────────────────┐              ┌─────────────────┐  ┌───────────┐│
│     │  HS Code Finder │              │  Tax Calculator │  │  Report   ││
│     │    (ReAct)      │              │                 │  │  Writer   ││
│     │                 │              │                 │  │ (Parallel)││
│     │ LLM이 도구를    │              │ 환율 조회 +     │  │           ││
│     │ 자율적으로 선택 │              │ 비용 계산       │  │ PDF/Word/ ││
│     │                 │              │                 │  │ Excel     ││
│     └────────┬────────┘              └────────┬────────┘  └─────┬─────┘│
│              │                                │                  │      │
│              ▼                                ▼                  ▼      │
│     ┌─────────────────┐              ┌─────────────────┐  ┌───────────┐│
│     │ • hs_code_      │              │ • exchange_     │  │ • pdf_    ││
│     │   search        │              │   rate_loader   │  │   report  ││
│     │   (FAISS RAG)   │              │   (API)         │  │ • word_   ││
│     │ • tariff_       │              │ • cost_         │  │   report  ││
│     │   search        │              │   calculator    │  │ • excel_  ││
│     │   (관세DB)      │              │                 │  │   report  ││
│     └─────────────────┘              └─────────────────┘  └───────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 에이전트별 상세

#### 1. HS Code Finder (ReAct 에이전트)

| 항목 | 내용 |
|------|------|
| 패턴 | ReAct (Reasoning + Acting) |
| LLM | GPT-4o-mini |
| 도구 | `hs_code_search`, `tariff_search_by_hs_code` |
| 역할 | 물품 설명 → HS 코드 후보 3개 검색 |

```python
# langgraph.prebuilt.create_react_agent 사용
self.agent = create_react_agent(
    model=self.llm,
    tools=[hs_code_search, tariff_search_by_hs_code],
)
```

#### 2. Tax Calculator

| 항목 | 내용 |
|------|------|
| 도구 | `exchange_rate_loader`, `final_cost_calculator` |
| 역할 | 환율 조회 + 관세/부가세 계산 |

```
총 비용 = 물품가격(원화) + 관세 + 부가세(10%)
```

#### 3. Report Writer (병렬 처리)

| 항목 | 내용 |
|------|------|
| 도구 | `pdf_report_exporter`, `word_report_exporter`, `excel_report_exporter` |
| 역할 | 3가지 형식 보고서 동시 생성 |

### 워크플로우 (LangGraph StateGraph)

```python
# graph.py
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("input_validator", input_validator_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("parallel_fetch", parallel_fetch_node)  # HS 코드 + 환율 병렬
workflow.add_node("tax_calculator", tax_calculator_node)
workflow.add_node("report_writer", report_writer_node)

# 조건부 엣지
workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "parallel_fetch": "parallel_fetch",
        "tax_calculator": "tax_calculator",
        "report_writer": "report_writer",
        "end_node": END,
    }
)
```

---

## 기술 스택

### AI / LLM

| 기술 | 용도 |
|------|------|
| **LangGraph** | 에이전트 오케스트레이션, StateGraph |
| **LangChain** | LLM 통합, 도구 정의 |
| **OpenAI GPT-4o** | 정보 추출, 분류 판단 |
| **GPT-4o-mini** | ReAct 에이전트 (비용 최적화) |

### Vector Store / RAG

| 기술 | 용도 |
|------|------|
| **FAISS** | HS 코드 벡터 검색 |
| **HuggingFace Embeddings** | 한국어 임베딩 (ko-sroberta) |

### API / Infrastructure

| 기술 | 용도 |
|------|------|
| **FastAPI** | REST API 서버 |
| **Streamlit** | 웹 UI |
| **Docker Compose** | 멀티 컨테이너 배포 |

---

## 설치 및 실행

### 사전 요구사항

- Python 3.10+
- OpenAI API 키

### 로컬 실행

```bash
# 1. 클론 및 의존성 설치
git clone https://github.com/your-repo/tradesimple.git
cd TradeSimple
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 3. API 서버 실행 (터미널 1)
python -m api.server
# → http://localhost:8000

# 4. Streamlit UI 실행 (터미널 2)
streamlit run app/main.py
# → http://localhost:8501
```

### Docker 실행

```bash
docker-compose up -d --build

# Web UI: http://localhost:8501
# API: http://localhost:8000/docs
```

---

## API 문서

### HITL 2단계 API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/analyze` | 1단계: 입력 분석 + HS 코드 후보 3개 반환 |
| POST | `/api/v1/calculate` | 2단계: 선택된 HS 코드로 비용 계산 + 보고서 생성 |

**중요:** `calculate`는 **analyze 단계에서 저장된 세션 입력값**을 사용합니다.  
따라서 `calculate` 요청에는 `session_id`와 `selected_hs_code`만 보내는 것이 정상 흐름입니다.
  
**세션 저장 방식:** 현재는 **메모리 기반**이므로 서버 재시작 시 세션이 초기화됩니다.

#### 1단계: 분석 요청

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"message": "미국에서 스마트워치 100개를 개당 300달러에 수입"}'
```

**응답:**
```json
{
  "session_id": "abc123",
  "phase": "hs_code_selection",
  "hs_code_candidates": [
    {"hs_code": "8517.62-9090", "품명": "무선통신기기", "적합도": "..."},
    {"hs_code": "9102.12-0000", "품명": "전자식 손목시계", "적합도": "..."},
    {"hs_code": "8471.30-0000", "품명": "휴대용 자동자료처리기계", "적합도": "..."}
  ],
  "exchange_rate": 1380.5
}
```

#### 2단계: 계산 요청

```bash
curl -X POST http://localhost:8000/api/v1/calculate \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "selected_hs_code": "8517.62-9090"}'
```

**입력 보정이 필요한 경우**  
1) 같은 `session_id`로 `analyze`를 다시 호출해 입력값을 갱신하거나  
2) `calculate`에 필요한 필드를 선택적으로 전달해 덮어쓸 수 있습니다.

---

## 프로젝트 구조

```
TradeSimple/
├── api/                          # REST API 서버
│   ├── server.py                 # FastAPI 엔드포인트
│   └── schemas.py                # Pydantic 스키마
│
├── app/                          # 코어 애플리케이션
│   ├── main.py                   # Streamlit Web UI
│   ├── graph.py                  # LangGraph StateGraph 정의
│   ├── state.py                  # AgentState TypedDict
│   ├── tools.py                  # 도구 정의 (6개)
│   └── agents/                   # ReAct 에이전트
│       ├── hs_code_finder.py     # HS 코드 검색 (ReAct)
│       ├── tax_calculator.py     # 비용 계산
│       └── report_writer.py      # 보고서 생성
│
├── data/                         # 데이터
│   ├── hsk_guide.pdf             # HS 코드 가이드 (RAG)
│   └── tariff_by_hs.csv          # 관세율 DB
│
├── vector_store/                 # FAISS 인덱스
├── docker-compose.yml
├── Dockerfile                    # API 서버
├── Dockerfile.web                # Web UI (경량)
├── requirements-web.txt          # Web UI 전용 의존성
└── requirements.txt
```

---

## 디자인 패턴

| 패턴 | 적용 |
|------|------|
| **Supervisor Pattern** | LangGraph로 3개 에이전트 중앙 제어 |
| **ReAct Pattern** | HS Code Finder가 도구를 자율적으로 선택 |
| **Human-in-the-Loop** | HS 코드 선택 단계에서 사용자 개입 |
| **Tool-Augmented Generation** | 6개 전문 도구로 LLM 기능 확장 |
| **Async Parallel Processing** | asyncio.gather로 병렬 처리 |
| **API-First Architecture** | UI/로직 분리, 확장 가능한 구조 |

---

## 환경 변수

| 변수명 | 필수 | 설명 |
|--------|------|------|
| `OPENAI_API_KEY` | ✅ | OpenAI API 키 |
| `API_BASE_URL` | ❌ | API 서버 URL (기본: http://localhost:8000) |
| `LANGCHAIN_TRACING_V2` | ❌ | LangSmith 트레이싱 |

---

## 라이선스

MIT License
