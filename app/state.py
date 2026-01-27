# app/state.py
"""
AgentState 정의 - LangGraph 상태 관리를 위한 TypedDict
모든 필드는 초기값 None으로 설정됨
"""
from typing import TypedDict, Optional, List, Dict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    수입 비용 계산 에이전트의 상태를 관리하는 TypedDict
    
    사용자 입력 필드:
        - item_name: 구매하고자 하는 물품명
        - quantity: 물품 개수
        - unit_price: 물품 단가
        - currency: 통화 (USD, EUR, JPY 등)
        - report_format: 원하는 보고서 양식 (all로 고정)
    
    처리 결과 필드:
        - hs_code: HS 코드 (10자리)
        - hs_code_rationale: HS 코드 분류 근거
        - tariff_rate: 관세율 (%)
        - exchange_rate: 실시간 환율 (KRW 기준)
        - tax_amount: 예상 관세 금액
        - total_cost: 종합 비용 (최종 예상 지출액)
        - report_paths: 생성된 보고서 파일 경로들
        - report_content: 보고서 내용
    
    흐름 제어 필드:
        - messages: 대화 기록
        - missing_info: 부족한 정보 목록
        - current_phase: 현재 처리 단계
        - error: 에러 메시지
    """
    # 사용자 입력 (초기값 None)
    item_name: Optional[str]
    quantity: Optional[int]
    unit_price: Optional[float]
    currency: Optional[str]
    report_format: Optional[str]
    
    # 처리 결과 (초기값 None)
    hs_code: Optional[str]
    hs_code_rationale: Optional[str]
    tariff_rate: Optional[float]
    exchange_rate: Optional[float]
    tax_amount: Optional[float]
    total_cost: Optional[float]
    report_paths: Optional[Dict[str, str]]
    report_content: Optional[str]
    
    # 흐름 제어
    messages: Annotated[List[BaseMessage], add_messages]
    missing_info: Optional[List[str]]
    current_phase: Optional[str]
    error: Optional[str]


def get_initial_state() -> dict:
    """초기 상태를 반환하는 헬퍼 함수"""
    return {
        # 사용자 입력
        "item_name": None,
        "quantity": None,
        "unit_price": None,
        "currency": None,
        "report_format": "all",  # PDF, Word, Excel 모두 생성
        
        # 처리 결과
        "hs_code": None,
        "hs_code_rationale": None,
        "tariff_rate": None,
        "exchange_rate": None,
        "tax_amount": None,
        "total_cost": None,
        "report_paths": None,
        "report_content": None,
        
        # 흐름 제어
        "messages": [],
        "missing_info": None,
        "current_phase": "input_validation",
        "error": None,
    }


# 필수 입력 필드 목록
REQUIRED_FIELDS = ["item_name", "quantity", "unit_price", "currency"]

# 필드 한글명 매핑
FIELD_NAMES_KR = {
    "item_name": "물품명",
    "quantity": "개수",
    "unit_price": "단가",
    "currency": "통화",
}
