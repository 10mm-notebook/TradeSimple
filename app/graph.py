# app/graph.py
"""
LangGraph 기반 Supervisor 그래프 정의
- 중앙 집중형 오케스트레이션 패턴
- 비동기 병렬 처리: HS 코드 검색 + 환율 조회 동시 실행
- 진짜 ReAct 에이전트 연동
"""
import re
import asyncio
from typing import Literal, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from app.state import AgentState, get_initial_state, REQUIRED_FIELDS, FIELD_NAMES_KR
from app.agents import HSCodeFinderAgent, TaxCalculatorAgent, ReportWriterAgent
from app.tools import exchange_rate_loader


# LLM 초기화
def get_llm():
    """OpenAI LLM 인스턴스 반환"""
    return ChatOpenAI(model="gpt-4o", temperature=0)


# ===== 노드 함수 정의 =====

async def input_validator_node(state: AgentState) -> Dict[str, Any]:
    """
    입력 검증 노드
    - 사용자 메시지에서 물품명, 수량, 단가, 통화 추출
    - 필수 정보 누락 시 missing_info에 기록
    """
    print("[Node] input_validator 실행")
    
    messages = state.get("messages", [])
    if not messages:
        return {
            "missing_info": REQUIRED_FIELDS.copy(),
            "current_phase": "request_info",
            "error": "입력 메시지가 없습니다."
        }
    
    # 마지막 사용자 메시지 가져오기
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        return {
            "missing_info": REQUIRED_FIELDS.copy(),
            "current_phase": "request_info",
        }
    
    # LLM을 사용하여 정보 추출
    llm = get_llm()
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 수입 비용 계산을 위한 정보 추출 전문가입니다.
사용자 메시지에서 다음 정보를 추출하세요:

1. item_name: 수입하려는 물품명 (예: 냉동 참치, 스마트워치, 노트북)
2. quantity: 수량 (숫자만)
3. unit_price: 단가 (숫자만)
4. currency: 통화 (USD, EUR, JPY 등. 달러는 USD, 엔은 JPY, 유로는 EUR)

다음 형식으로 정확히 응답하세요:
ITEM_NAME: [물품명 또는 NONE]
QUANTITY: [숫자 또는 NONE]
UNIT_PRICE: [숫자 또는 NONE]
CURRENCY: [통화코드 또는 NONE]"""),
        ("human", "{user_message}")
    ])
    
    response = await llm.ainvoke(extraction_prompt.format_messages(user_message=user_message))
    extraction_text = response.content
    
    # 추출 결과 파싱
    extracted = {}
    
    # ITEM_NAME 추출
    match = re.search(r'ITEM_NAME:\s*(.+?)(?:\n|$)', extraction_text)
    if match and match.group(1).strip().upper() != 'NONE':
        extracted["item_name"] = match.group(1).strip()
    
    # QUANTITY 추출
    match = re.search(r'QUANTITY:\s*(\d+)', extraction_text)
    if match:
        extracted["quantity"] = int(match.group(1))
    
    # UNIT_PRICE 추출
    match = re.search(r'UNIT_PRICE:\s*([\d.]+)', extraction_text)
    if match:
        extracted["unit_price"] = float(match.group(1))
    
    # CURRENCY 추출
    match = re.search(r'CURRENCY:\s*([A-Z]{3})', extraction_text)
    if match:
        extracted["currency"] = match.group(1)
    else:
        # 기본값 USD
        if extracted.get("unit_price"):
            extracted["currency"] = "USD"
    
    # 기존 상태와 병합
    item_name = extracted.get("item_name") or state.get("item_name")
    quantity = extracted.get("quantity") or state.get("quantity")
    unit_price = extracted.get("unit_price") or state.get("unit_price")
    currency = extracted.get("currency") or state.get("currency")
    
    # 누락된 정보 확인
    missing = []
    if not item_name:
        missing.append("item_name")
    if not quantity:
        missing.append("quantity")
    if not unit_price:
        missing.append("unit_price")
    if not currency:
        missing.append("currency")
    
    update = {
        "item_name": item_name,
        "quantity": quantity,
        "unit_price": unit_price,
        "currency": currency,
        "missing_info": missing if missing else None,
        "current_phase": "request_info" if missing else "parallel_fetch",
    }
    
    print(f"[Node] input_validator 완료: 추출됨={extracted}, 누락={missing}")
    return update


async def request_info_node(state: AgentState) -> Dict[str, Any]:
    """
    정보 요청 노드
    - 누락된 정보에 대해 사용자에게 재입력 요청
    """
    print("[Node] request_info 실행")
    
    missing = state.get("missing_info", [])
    if not missing:
        return {"current_phase": "supervisor"}
    
    missing_names = [FIELD_NAMES_KR.get(f, f) for f in missing]
    request_message = f"다음 정보가 필요합니다: {', '.join(missing_names)}\n\n"
    request_message += "예시: '미국에서 스마트워치 100개를 개당 300달러에 수입하려고 합니다.'"
    
    return {
        "messages": [AIMessage(content=request_message)],
        "current_phase": "waiting_input",
    }


async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor 노드
    - 전체 워크플로우 조율
    - 현재 단계 확인 및 다음 작업 결정
    """
    print("[Node] supervisor 실행")
    
    current_phase = state.get("current_phase", "input_validation")
    
    # 단계별 상태 확인
    if state.get("missing_info"):
        return {"current_phase": "request_info"}
    
    # HS 코드와 환율이 모두 없으면 병렬 조회
    if not state.get("hs_code") and not state.get("exchange_rate"):
        return {"current_phase": "parallel_fetch"}
    
    # HS 코드만 없으면
    if not state.get("hs_code"):
        return {"current_phase": "hs_code_finder"}
    
    # 비용 계산이 안 됐으면
    if state.get("total_cost") is None:
        return {"current_phase": "tax_calculator"}
    
    # 보고서가 없으면
    if not state.get("report_paths"):
        return {"current_phase": "report_writer"}
    
    # 모든 작업 완료
    return {"current_phase": "complete"}


async def parallel_fetch_node(state: AgentState) -> Dict[str, Any]:
    """
    🔥 병렬 조회 노드 (핵심!)
    - HS 코드 검색과 환율 조회를 동시에 실행
    - asyncio.gather를 사용한 진짜 병렬 처리
    """
    print("[Node] parallel_fetch 실행 - HS 코드 검색 + 환율 조회 병렬 시작")
    
    item_name = state.get("item_name")
    currency = state.get("currency", "USD")
    
    if not item_name:
        return {"error": "물품명이 없습니다.", "current_phase": "request_info"}
    
    # 상태 메시지
    status_msg = AIMessage(content=f"**병렬 처리 시작:** '{item_name}'의 HS 코드 검색과 {currency} 환율 조회를 동시에 실행합니다...")
    
    # 🔥 병렬 실행: HS 코드 검색 + 환율 조회
    async def fetch_hs_code():
        """HS 코드 검색 (ReAct 에이전트)"""
        agent = HSCodeFinderAgent()
        return await agent.run(item_name)
    
    async def fetch_exchange_rate():
        """환율 조회"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: exchange_rate_loader.invoke({"target_currency": currency})
        )
    
    # asyncio.gather로 동시 실행!
    print("[Node] parallel_fetch - asyncio.gather 시작")
    hs_result, exchange_result = await asyncio.gather(
        fetch_hs_code(),
        fetch_exchange_rate()
    )
    print("[Node] parallel_fetch - asyncio.gather 완료")
    
    return {
        "messages": [status_msg],
        "hs_code": hs_result["hs_code"],
        "hs_code_rationale": hs_result["rationale"],
        "tariff_rate": hs_result["tariff_rate"],
        "exchange_rate": exchange_result["rate"],
        "current_phase": "tax_calculator",
    }


async def hs_code_finder_node(state: AgentState) -> Dict[str, Any]:
    """
    HS Code Finder 노드 (단독 실행용)
    - 환율이 이미 있는 경우에만 사용
    """
    print("[Node] hs_code_finder 실행")
    
    item_name = state.get("item_name")
    if not item_name:
        return {"error": "물품명이 없습니다.", "current_phase": "request_info"}
    
    status_msg = AIMessage(content=f"**HS Code & Tax Finder (ReAct):** '{item_name}'의 HS 코드를 검색합니다...")
    
    agent = HSCodeFinderAgent()
    result = await agent.run(item_name)
    
    return {
        "messages": [status_msg],
        "hs_code": result["hs_code"],
        "hs_code_rationale": result["rationale"],
        "tariff_rate": result["tariff_rate"],
        "current_phase": "tax_calculator",
    }


async def tax_calculator_node(state: AgentState) -> Dict[str, Any]:
    """
    Tax Calculator 노드 (ReAct 패턴)
    - 환율이 이미 조회된 상태에서 비용 계산
    """
    print("[Node] tax_calculator 실행")
    
    unit_price = state.get("unit_price")
    quantity = state.get("quantity")
    currency = state.get("currency")
    tariff_rate = state.get("tariff_rate", 0.0)
    exchange_rate = state.get("exchange_rate")
    
    if not all([unit_price, quantity, currency]):
        return {"error": "비용 계산에 필요한 정보가 부족합니다.", "current_phase": "request_info"}
    
    status_msg = AIMessage(content=f"**Tax Calculator (ReAct):** 비용을 계산합니다...")
    
    agent = TaxCalculatorAgent()
    result = await agent.run(
        unit_price=unit_price,
        quantity=quantity,
        currency=currency,
        tariff_rate=tariff_rate
    )
    
    # 병렬 조회에서 이미 환율을 가져왔다면 그 값 유지
    final_exchange_rate = exchange_rate or result["exchange_rate"]
    
    return {
        "messages": [status_msg],
        "exchange_rate": final_exchange_rate,
        "tax_amount": result["tax_amount"],
        "total_cost": result["total_cost"],
        "current_phase": "report_writer",
    }


async def report_writer_node(state: AgentState) -> Dict[str, Any]:
    """
    Report Writer 노드 (병렬 보고서 생성)
    - PDF, Word, Excel을 asyncio.gather로 동시 생성
    """
    print("[Node] report_writer 실행")
    
    status_msg = AIMessage(content="**Report Writer:** 최종 보고서를 생성합니다 (PDF/Word/Excel 병렬)...")
    
    agent = ReportWriterAgent()
    
    exchange_source = "exchangerate-api.com"
    
    # 부가세 계산
    total_krw = state.get("unit_price", 0) * state.get("quantity", 0) * state.get("exchange_rate", 1)
    tax_amount = state.get("tax_amount", 0)
    vat_amount = (total_krw + tax_amount) * 0.10
    
    result = await agent.run(
        item_name=state.get("item_name", ""),
        quantity=state.get("quantity", 0),
        unit_price=state.get("unit_price", 0),
        currency=state.get("currency", "USD"),
        hs_code=state.get("hs_code", ""),
        hs_code_rationale=state.get("hs_code_rationale", ""),
        tariff_rate=state.get("tariff_rate", 0),
        exchange_rate=state.get("exchange_rate", 0),
        exchange_source=exchange_source,
        tax_amount=tax_amount,
        vat_amount=vat_amount,
        total_cost=state.get("total_cost", 0),
        report_format="all"  # PDF/Word/Excel 병렬 생성
    )
    
    final_msg = AIMessage(
        content=result["report_content"],
        additional_kwargs={"report_paths": result["report_paths"]}
    )
    
    return {
        "messages": [status_msg, final_msg],
        "report_content": result["report_content"],
        "report_paths": result["report_paths"],
        "current_phase": "complete",
    }


# ===== 라우팅 함수 =====

def route_after_input_validation(state: AgentState) -> Literal["request_info", "supervisor"]:
    """입력 검증 후 라우팅"""
    if state.get("missing_info"):
        return "request_info"
    return "supervisor"


def route_supervisor(state: AgentState) -> Literal["parallel_fetch", "hs_code_finder", "tax_calculator", "report_writer", "end_node"]:
    """Supervisor 라우팅"""
    phase = state.get("current_phase", "")
    
    if phase == "parallel_fetch":
        return "parallel_fetch"
    elif phase == "hs_code_finder":
        return "hs_code_finder"
    elif phase == "tax_calculator":
        return "tax_calculator"
    elif phase == "report_writer":
        return "report_writer"
    else:
        return "end_node"


# ===== 그래프 생성 =====

def create_graph():
    """LangGraph 그래프 생성"""
    
    # StateGraph 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("input_validator", input_validator_node)
    workflow.add_node("request_info", request_info_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("parallel_fetch", parallel_fetch_node)  # 🔥 병렬 조회 노드
    workflow.add_node("hs_code_finder", hs_code_finder_node)
    workflow.add_node("tax_calculator", tax_calculator_node)
    workflow.add_node("report_writer", report_writer_node)
    
    # 엣지 추가
    workflow.add_edge(START, "input_validator")
    
    workflow.add_conditional_edges(
        "input_validator",
        route_after_input_validation,
        {
            "request_info": "request_info",
            "supervisor": "supervisor",
        }
    )
    
    workflow.add_edge("request_info", END)
    
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "parallel_fetch": "parallel_fetch",
            "hs_code_finder": "hs_code_finder",
            "tax_calculator": "tax_calculator",
            "report_writer": "report_writer",
            "end_node": END,
        }
    )
    
    # 각 노드에서 supervisor로 복귀
    workflow.add_edge("parallel_fetch", "supervisor")
    workflow.add_edge("hs_code_finder", "supervisor")
    workflow.add_edge("tax_calculator", "supervisor")
    workflow.add_edge("report_writer", "supervisor")
    
    # 컴파일
    app = workflow.compile()
    
    return app


# 그래프 인스턴스 (lazy initialization)
_graph = None

def get_graph():
    """그래프 인스턴스 반환 (싱글톤)"""
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph


async def run_agent(user_input: str, current_state: Optional[Dict] = None) -> Dict[str, Any]:
    """
    에이전트 실행 함수
    
    Args:
        user_input: 사용자 입력 메시지
        current_state: 현재 상태 (대화 지속 시)
        
    Returns:
        업데이트된 상태 딕셔너리
    """
    graph = get_graph()
    
    # 초기 상태 설정
    if current_state is None:
        state = get_initial_state()
    else:
        state = current_state.copy()
    
    # 사용자 메시지 추가
    state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
    
    # 그래프 실행
    final_state = await graph.ainvoke(state)
    
    return final_state
