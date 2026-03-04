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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from app.state import AgentState, get_initial_state, REQUIRED_FIELDS, FIELD_NAMES_KR
from app.agents import HSCodeFinderAgent, TaxCalculatorAgent, ReportWriterAgent
from app.models import get_llm
from app.tools import exchange_rate_loader


# ===== 노드 함수 정의 =====

async def input_validator_node(state: AgentState) -> Dict[str, Any]:
    """
    입력 검증 노드
    - 사용자 메시지에서 물품명, 수량, 단가, 통화 추출
    - 필수 정보 누락 시 missing_info에 기록
    """
    print("[Node] input_validator 실행")
    
    # 🔥 이미 HS 코드가 선택되어 있으면 (HITL 후) 바로 tax_calculator로
    if state.get("hs_code") and state.get("current_phase") == "tax_calculator":
        print("[Node] input_validator - HS 코드 이미 선택됨, tax_calculator로 바로 진행")
        return {"current_phase": "tax_calculator"}
    
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
2. quantity: 수량 (숫자만, 단위 무시)
3. quantity_unit: 수량의 단위 (개, kg, g, lb, 톤, 박스 등)
4. unit_price: 단가 (숫자만)
5. price_unit: 단가 기준 (1개당, 100g당, 1kg당, 박스당 등)
6. total_foreign_price: **총 외화 금액을 직접 계산** (중요!)
7. currency: 통화코드 (3자리 ISO 코드)
8. raw_material: 원재료 또는 주원료 (예: 면 100%, 알루미늄 합금, 우유, 카카오) — 없으면 NONE
9. processing_method: 가공방법 (예: 냉동, 훈제, 로스팅, 발효, 압착, 분무건조) — 없으면 NONE
10. product_form: 제품형태 (예: 분말, 원단, 완성품, 원과, 알맹이, 캡슐) — 없으면 NONE
11. main_material: 주요 소재/성분 (예: 리튬이온, 카카오 35% 이상, 다운 충전재, 천연 가죽) — 없으면 NONE

**단위 계산 예시 (중요!):**
- "100그램당 5위안, 300kg 수입" → 300kg = 300,000g, 총 가격 = (300,000 / 100) × 5 = 15,000
- "1kg당 10달러, 500kg 수입" → 총 가격 = 500 × 10 = 5,000
- "개당 300달러, 100개 수입" → 총 가격 = 100 × 300 = 30,000

**통화 변환 규칙:**
- "달러", "$", "불" → USD
- "엔", "¥" → JPY
- "유로", "€" → EUR
- "위안", "元" → CNY
- "원", "₩" → KRW
- "파운드", "£" → GBP
- 호주/캐나다/홍콩/싱가포르/대만 달러 → AUD/CAD/HKD/SGD/TWD
- "바트" → THB, "동" → VND, "루피" → INR, "루블" → RUB
- 통화 미명시 → USD

다음 형식으로 정확히 응답하세요:
ITEM_NAME: [물품명]
QUANTITY: [총 수량 숫자]
QUANTITY_UNIT: [수량 단위: 개/kg/g/lb/톤/박스 등]
UNIT_PRICE: [단가 숫자]
PRICE_UNIT: [단가 기준: 1개당/100g당/1kg당 등]
TOTAL_FOREIGN_PRICE: [총 외화 금액 계산 결과]
CURRENCY: [통화코드]
RAW_MATERIAL: [원재료 또는 NONE]
PROCESSING_METHOD: [가공방법 또는 NONE]
PRODUCT_FORM: [제품형태 또는 NONE]
MAIN_MATERIAL: [주요 소재/성분 또는 NONE]"""),
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
    match = re.search(r'QUANTITY:\s*([\d,]+)', extraction_text)
    if match:
        extracted["quantity"] = int(match.group(1).replace(',', ''))
    
    # QUANTITY_UNIT 추출
    match = re.search(r'QUANTITY_UNIT:\s*(.+?)(?:\n|$)', extraction_text)
    if match and match.group(1).strip().upper() != 'NONE':
        extracted["quantity_unit"] = match.group(1).strip()
    
    # UNIT_PRICE 추출
    match = re.search(r'UNIT_PRICE:\s*([\d.,]+)', extraction_text)
    if match:
        extracted["unit_price"] = float(match.group(1).replace(',', ''))
    
    # PRICE_UNIT 추출
    match = re.search(r'PRICE_UNIT:\s*(.+?)(?:\n|$)', extraction_text)
    if match and match.group(1).strip().upper() != 'NONE':
        extracted["price_unit"] = match.group(1).strip()
    
    # TOTAL_FOREIGN_PRICE 추출 (LLM이 계산한 총 외화 금액)
    match = re.search(r'TOTAL_FOREIGN_PRICE:\s*([\d.,]+)', extraction_text)
    if match:
        extracted["total_foreign_price"] = float(match.group(1).replace(',', ''))
    
    # CURRENCY 추출 (NON, NONE 제외)
    match = re.search(r'CURRENCY:\s*([A-Z]{3})', extraction_text)
    if match:
        curr = match.group(1).upper()
        if curr in ("NON", "NONE", "NAN"):
            # 잘못된 값이면 기본값 USD
            extracted["currency"] = "USD"
        else:
            extracted["currency"] = curr
    else:
        # 기본값 USD
        extracted["currency"] = "USD"
    
    # 사용자 입력에서 직접 통화 패턴 추출 (LLM이 못 잡은 경우 보완)
    # 순서 중요: 더 구체적인 패턴을 먼저 검사
    if user_message:
        msg = user_message
        msg_lower = msg.lower()
        
        # 복합어 먼저 체크 (호주 달러, 캐나다 달러 등)
        if "호주" in msg_lower and "달러" in msg_lower:
            extracted["currency"] = "AUD"
        elif "캐나다" in msg_lower and "달러" in msg_lower:
            extracted["currency"] = "CAD"
        elif "홍콩" in msg_lower and "달러" in msg_lower:
            extracted["currency"] = "HKD"
        elif "싱가포르" in msg_lower and "달러" in msg_lower:
            extracted["currency"] = "SGD"
        elif "대만" in msg_lower and "달러" in msg_lower:
            extracted["currency"] = "TWD"
        # 단일 키워드
        elif "원" in msg or "₩" in msg:
            extracted["currency"] = "KRW"
        elif "달러" in msg_lower or "$" in msg or "불" in msg_lower:
            extracted["currency"] = "USD"
        elif "엔" in msg_lower or "¥" in msg:
            extracted["currency"] = "JPY"
        elif "유로" in msg_lower or "€" in msg:
            extracted["currency"] = "EUR"
        elif "위안" in msg_lower or "元" in msg:
            extracted["currency"] = "CNY"
        elif "파운드" in msg_lower or "£" in msg:
            extracted["currency"] = "GBP"
        elif "프랑" in msg_lower:
            extracted["currency"] = "CHF"
        elif "바트" in msg_lower:
            extracted["currency"] = "THB"
        elif "동" in msg and ("베트남" in msg_lower or "vnd" in msg_lower):
            extracted["currency"] = "VND"
        elif "루피" in msg_lower:
            extracted["currency"] = "INR"
        elif "루블" in msg_lower:
            extracted["currency"] = "RUB"
        elif "링깃" in msg_lower:
            extracted["currency"] = "MYR"
        elif "페소" in msg_lower:
            extracted["currency"] = "PHP"
    
    # 상세 정보 추출 (선택 필드 — NONE이면 None으로)
    for field_key, label in [
        ("raw_material", "RAW_MATERIAL"),
        ("processing_method", "PROCESSING_METHOD"),
        ("product_form", "PRODUCT_FORM"),
        ("main_material", "MAIN_MATERIAL"),
    ]:
        m = re.search(rf'{label}:\s*(.+?)(?:\n|$)', extraction_text)
        if m:
            val = m.group(1).strip()
            if val.upper() != "NONE" and val:
                extracted[field_key] = val

    # 기존 상태와 병합
    item_name = extracted.get("item_name") or state.get("item_name")
    quantity = extracted.get("quantity") or state.get("quantity")
    quantity_unit = extracted.get("quantity_unit") or state.get("quantity_unit") or "개"
    unit_price = extracted.get("unit_price") or state.get("unit_price")
    price_unit = extracted.get("price_unit") or state.get("price_unit") or "1개당"
    total_foreign_price = extracted.get("total_foreign_price") or state.get("total_foreign_price")
    currency = extracted.get("currency") or state.get("currency")
    
    # total_foreign_price가 없으면 기본 계산 (quantity * unit_price)
    if not total_foreign_price and quantity and unit_price:
        total_foreign_price = quantity * unit_price
    
    # 누락된 정보 확인 (total_foreign_price가 있으면 개별 필드 없어도 OK)
    missing = []
    if not item_name:
        missing.append("item_name")
    if not total_foreign_price:
        if not quantity:
            missing.append("quantity")
        if not unit_price:
            missing.append("unit_price")
    if not currency:
        missing.append("currency")
    
    update = {
        "item_name": item_name,
        "quantity": quantity,
        "quantity_unit": quantity_unit,
        "unit_price": unit_price,
        "price_unit": price_unit,
        "total_foreign_price": total_foreign_price,
        "currency": currency,
        # 상세 정보 (기존 state 값 우선, 새 추출값으로 보완)
        "raw_material": extracted.get("raw_material") or state.get("raw_material"),
        "processing_method": extracted.get("processing_method") or state.get("processing_method"),
        "product_form": extracted.get("product_form") or state.get("product_form"),
        "main_material": extracted.get("main_material") or state.get("main_material"),
        "missing_info": missing if missing else None,
        "current_phase": "request_info" if missing else "parallel_fetch",
    }

    print(f"[Node] input_validator 완료: 추출됨={extracted}, 총외화={total_foreign_price}, 누락={missing}")
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
    current_phase = state.get("current_phase", "input_validation")
    total_cost = state.get("total_cost")
    hs_code = state.get("hs_code")
    report_paths = state.get("report_paths")
    
    print(f"[Node] supervisor 실행 - phase={current_phase}, hs_code={hs_code}, total_cost={total_cost}, report_paths={bool(report_paths)}")
    
    # 🔥 Human-in-the-Loop: HS 코드 선택 대기 상태면 종료 (사용자 선택 필요)
    if current_phase == "hs_code_selection":
        print("[Node] supervisor - HS 코드 선택 대기 (Human-in-the-Loop)")
        return {"current_phase": "hs_code_selection"}  # 상태 유지하며 종료 → UI에서 선택
    
    # 🔥 이미 비용 계산이 완료되었으면 (current_phase가 report_writer면) 보고서로
    if current_phase == "report_writer" and total_cost is not None:
        print("[Node] supervisor - 비용 계산 완료, report_writer로")
        return {"current_phase": "report_writer"}
    
    # 단계별 상태 확인
    if state.get("missing_info"):
        return {"current_phase": "request_info"}
    
    # HS 코드가 없으면 병렬 조회 (HS 코드 후보 + 환율 동시 실행)
    if not hs_code:
        return {"current_phase": "parallel_fetch"}

    # 비용 계산이 안 됐으면 (total_cost가 None이거나 명시적으로 0이 아닌 None)
    if total_cost is None:
        return {"current_phase": "tax_calculator"}
    
    # 보고서가 없으면
    if not report_paths:
        return {"current_phase": "report_writer"}
    
    # 모든 작업 완료
    print("[Node] supervisor - 완료")
    return {"current_phase": "complete"}


async def parallel_fetch_node(state: AgentState) -> Dict[str, Any]:
    """
    🔥 병렬 조회 노드 (Human-in-the-Loop 적용)
    - HS 코드 후보 3개 검색과 환율 조회를 동시에 실행
    - 사용자가 HS 코드를 선택할 수 있도록 후보 반환
    """
    print("[Node] parallel_fetch 실행 - HS 코드 후보 검색 + 환율 조회 병렬 시작")
    
    item_name = state.get("item_name")
    currency = state.get("currency", "USD")
    
    if not item_name:
        return {"error": "물품명이 없습니다.", "current_phase": "request_info"}
    
    status_msg = AIMessage(content=f"**병렬 처리 시작:** '{item_name}'의 HS 코드 후보를 검색하고 {currency} 환율을 조회합니다...")
    
    # 🔥 병렬 실행: HS 코드 후보 검색 + 환율 조회
    async def fetch_hs_code_candidates():
        """HS 코드 후보 3개 검색 (Human-in-the-Loop)"""
        agent = HSCodeFinderAgent()
        return await agent.run_with_candidates(
            item_name,
            raw_material=state.get("raw_material"),
            processing_method=state.get("processing_method"),
            product_form=state.get("product_form"),
            main_material=state.get("main_material"),
        )
    
    async def fetch_exchange_rate():
        """환율 조회"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: exchange_rate_loader.invoke({"target_currency": currency})
        )
    
    print("[Node] parallel_fetch - asyncio.gather 시작")
    hs_result, exchange_result = await asyncio.gather(
        fetch_hs_code_candidates(),
        fetch_exchange_rate()
    )
    print("[Node] parallel_fetch - asyncio.gather 완료")
    
    candidates = hs_result.get("candidates", [])
    
    # 후보가 있으면 사용자 선택 대기, 없으면 바로 진행
    if candidates and len(candidates) > 0:
        # 선택 안내 메시지
        selection_msg = AIMessage(
            content="**HS 코드 후보를 찾았습니다.** 아래에서 가장 적합한 HS 코드를 선택해주세요:",
            additional_kwargs={"hs_code_candidates": candidates}
        )
        return {
            "messages": [status_msg, selection_msg],
            "hs_code_candidates": candidates,
            "exchange_rate": exchange_result["rate"],
            "current_phase": "hs_code_selection",  # 사용자 선택 대기
        }
    else:
        # 후보 없으면 기본 검색 결과 사용
        agent = HSCodeFinderAgent()
        default_result = await agent.run(
            item_name,
            raw_material=state.get("raw_material"),
            processing_method=state.get("processing_method"),
            product_form=state.get("product_form"),
            main_material=state.get("main_material"),
        )
        return {
            "messages": [status_msg],
            "hs_code": default_result["hs_code"],
            "hs_code_rationale": default_result["rationale"],
            "tariff_rate": default_result["tariff_rate"],
            "exchange_rate": exchange_result["rate"],
            "current_phase": "tax_calculator",
        }


async def tax_calculator_node(state: AgentState) -> Dict[str, Any]:
    """
    Tax Calculator 노드
    - 환율이 이미 조회된 상태에서 비용 계산 (재조회 없음)
    - 무게/개수 등 다양한 단위 지원
    """
    print("[Node] tax_calculator 실행")
    
    unit_price = state.get("unit_price") or 0
    quantity = state.get("quantity") or 0
    quantity_unit = state.get("quantity_unit") or "개"
    price_unit = state.get("price_unit") or "1개당"
    total_foreign_price = state.get("total_foreign_price")
    currency = state.get("currency")
    tariff_rate = state.get("tariff_rate", 0.0)
    exchange_rate = state.get("exchange_rate")
    
    # total_foreign_price가 있거나, unit_price와 quantity가 있어야 함
    if not total_foreign_price and not (unit_price and quantity):
        return {"error": "비용 계산에 필요한 정보가 부족합니다.", "current_phase": "request_info"}
    if not currency:
        return {"error": "통화 정보가 없습니다.", "current_phase": "request_info"}
    
    status_msg = AIMessage(content=f"**Tax Calculator (ReAct):** 비용을 계산합니다...")
    
    agent = TaxCalculatorAgent()
    result = await agent.run(
        unit_price=unit_price,
        quantity=quantity,
        currency=currency,
        tariff_rate=tariff_rate,
        total_foreign_price=total_foreign_price,
        quantity_unit=quantity_unit,
        price_unit=price_unit,
        exchange_rate=exchange_rate,  # parallel_fetch에서 조회된 값이 있으면 재사용
    )

    final_exchange_rate = result["exchange_rate"]
    final_total_cost = result["total_cost"]
    final_tax_amount = result["tax_amount"]
    
    print(f"[Node] tax_calculator 완료 - total_cost={final_total_cost}, tax={final_tax_amount}, exchange_rate={final_exchange_rate}")
    
    return {
        "messages": [status_msg],
        "exchange_rate": final_exchange_rate,
        "tax_amount": final_tax_amount,
        "total_cost": final_total_cost,
        "current_phase": "report_writer",
    }


async def report_writer_node(state: AgentState) -> Dict[str, Any]:
    """
    Report Writer 노드 (병렬 보고서 생성)
    - PDF, Word, Excel을 asyncio.gather로 동시 생성
    - 파일명에 품목명과 HS코드 포함
    """
    print("[Node] report_writer 실행")
    
    status_msg = AIMessage(content="**Report Writer:** 최종 보고서를 생성합니다 (PDF/Word/Excel 병렬)...")
    
    agent = ReportWriterAgent()
    
    exchange_source = "exchangerate-api.com"
    
    # 총 외화 금액 및 원화 계산
    total_foreign_price = state.get("total_foreign_price") or (state.get("unit_price", 0) * state.get("quantity", 0))
    total_krw = total_foreign_price * state.get("exchange_rate", 1)
    tax_amount = state.get("tax_amount", 0)
    vat_amount = (total_krw + tax_amount) * 0.10
    
    report_id = state.get("report_id", 0)
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
        report_format="all",  # PDF/Word/Excel 병렬 생성
        report_id=report_id,
        quantity_unit=state.get("quantity_unit", "개"),
        price_unit=state.get("price_unit", "1개당"),
        total_foreign_price=total_foreign_price,
        raw_material=state.get("raw_material"),
        processing_method=state.get("processing_method"),
        product_form=state.get("product_form"),
        main_material=state.get("main_material"),
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


def route_supervisor(state: AgentState) -> Literal["parallel_fetch", "tax_calculator", "report_writer", "end_node"]:
    """Supervisor 라우팅"""
    phase = state.get("current_phase", "")

    print(f"[Route] supervisor → {phase}")

    if phase == "parallel_fetch":
        return "parallel_fetch"
    elif phase == "tax_calculator":
        return "tax_calculator"
    elif phase == "report_writer":
        return "report_writer"
    else:
        # hs_code_selection / complete / 기타 → 그래프 종료
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
    workflow.add_node("parallel_fetch", parallel_fetch_node)
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
            "tax_calculator": "tax_calculator",
            "report_writer": "report_writer",
            "end_node": END,
        }
    )

    # 각 노드에서 supervisor로 복귀
    workflow.add_edge("parallel_fetch", "supervisor")
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


def _status_message(node_name: str, state: Dict[str, Any]) -> str:
    """노드별 스트리밍용 상태 메시지."""
    item_name = state.get("item_name") or "물품"
    currency = state.get("currency", "USD")
    now = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    if node_name == "input_validator":
        return "📥 입력 정보를 분석하고 있습니다..."
    if node_name == "request_info":
        return "⏳ 추가 정보가 필요합니다. 안내 메시지를 작성 중입니다."
    if node_name == "parallel_fetch":
        return f"🔍 **'{item_name}'**의 HS 코드를 검색하고, **{currency}** 환율을 조회하고 있습니다..."
    if node_name == "tax_calculator":
        er = state.get("exchange_rate")
        if er is not None:
            return f"💱 {now} 현재 환율은 **{er:,.2f} KRW/{currency}** 입니다. 비용을 계산하고 있습니다..."
        return "💰 비용을 계산하고 있습니다..."
    if node_name == "report_writer":
        return "📝 PDF·Word·Excel 보고서를 생성하고 있습니다..."
    return "처리 중..."


async def run_agent(
    user_input: str,
    current_state: Optional[Dict] = None,
    report_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    에이전트 실행 함수
    
    Args:
        user_input: 사용자 입력 메시지
        current_state: 현재 상태 (대화 지속 시)
        report_id: 메시지별 보고서 파일 구분용 (report_0.pdf 등)
        
    Returns:
        업데이트된 상태 딕셔너리
    """
    graph = get_graph()
    
    if current_state is None:
        state = get_initial_state()
    else:
        state = current_state.copy()
    
    state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
    if report_id is not None:
        state["report_id"] = report_id
    
    final_state = await graph.ainvoke(state)
    return final_state


async def run_agent_stream(
    user_input: str,
    current_state: Optional[Dict] = None,
    report_id: Optional[int] = None,
    thread_id: Optional[str] = None,
):
    """
    에이전트 실행 + 실시간 진행 상황 스트리밍 (async generator).
    yield: {"message": str, "state": dict}

    Args:
        thread_id: LangSmith 추적용 스레드 ID (세션 ID와 동일하게 사용하면
                   analyze → calculate 전 과정이 하나의 스레드로 묶임)
    """
    graph = get_graph()

    if current_state is None:
        state = get_initial_state()
    else:
        state = current_state.copy()

    state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
    if report_id is not None:
        state["report_id"] = report_id

    lang_config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    current = dict(state)
    async for event in graph.astream(state, lang_config, stream_mode="updates"):
        for node_name, update in event.items():
            current = {**current, **update}
            msg = _status_message(node_name, current)
            yield {"message": msg, "state": current}

    yield {"message": "✅ 처리 완료", "state": current}


async def continue_after_hs_selection(
    selected_hs_code: str,
    selected_tariff_rate: float,
    selected_rationale: str,
    current_state: Dict[str, Any],
    report_id: Optional[int] = None,
    thread_id: Optional[str] = None,
):
    """
    사용자가 HS 코드를 선택한 후 계산을 계속 진행 (async generator).

    Args:
        selected_hs_code: 사용자가 선택한 HS 코드
        selected_tariff_rate: 선택한 HS 코드의 관세율 (0이면 조회 필요)
        selected_rationale: 선택 근거
        current_state: 현재 상태
        report_id: 보고서 파일 번호
        thread_id: LangSmith 추적용 스레드 ID (analyze 단계와 동일한 세션 ID 사용)
    """
    from app.tools import tariff_search_by_hs_code
    
    graph = get_graph()
    
    state = current_state.copy()
    state["hs_code"] = selected_hs_code
    state["hs_code_rationale"] = selected_rationale
    state["hs_code_candidates"] = None  # 선택 완료
    state["current_phase"] = "tax_calculator"  # 다음 단계로
    
    if report_id is not None:
        state["report_id"] = report_id
    
    # 🔥 관세율이 0이면 조회 (HITL 단계에서는 관세율 미조회)
    tariff_rate = selected_tariff_rate
    if tariff_rate == 0.0:
        try:
            tariff_result = tariff_search_by_hs_code.invoke({"hs_code": selected_hs_code})
            # 결과에서 관세율 추출
            import re
            tariff_match = re.search(r'최종[^:]*세율[:\s]*([0-9.]+)', tariff_result)
            if tariff_match:
                tariff_rate = float(tariff_match.group(1))
            else:
                # 기본세율 추출 시도
                basic_match = re.search(r'기본세율[:\s]*([0-9.]+)', tariff_result)
                if basic_match:
                    tariff_rate = float(basic_match.group(1))
        except Exception as e:
            print(f"[continue_after_hs_selection] 관세율 조회 실패: {e}")
            tariff_rate = 0.0
    
    state["tariff_rate"] = tariff_rate
    
    # 선택 메시지 추가
    selection_msg = AIMessage(
        content=f"**선택된 HS 코드:** {selected_hs_code} (관세율 {tariff_rate}%)\n\n계산을 진행합니다..."
    )
    state["messages"] = state.get("messages", []) + [selection_msg]
    
    lang_config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    current = dict(state)
    async for event in graph.astream(state, lang_config, stream_mode="updates"):
        for node_name, update in event.items():
            current = {**current, **update}
            msg = _status_message(node_name, current)
            yield {"message": msg, "state": current}

    yield {"message": "✅ 처리 완료", "state": current}
