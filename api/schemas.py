# api/schemas.py
"""
API 요청/응답 스키마 정의 - HITL(Human-in-the-Loop) 지원
"""
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


# ===== 1단계: 입력 분석 + HS 코드 후보 검색 =====

class AnalyzeRequest(BaseModel):
    """1단계: 입력 분석 요청"""
    message: str = Field(
        ...,
        description="사용자 입력 메시지 (물품명, 수량, 단가, 통화 포함)",
        examples=["미국에서 스마트워치 100개를 개당 300달러에 수입하려고 합니다."]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="세션 ID (기존 세션 재사용 시)"
    )


class HSCodeCandidate(BaseModel):
    """HS 코드 후보"""
    hs_code: str = Field(..., description="HS 코드 (10자리)")
    품명: Optional[str] = Field(None, description="품명")
    적합도: Optional[str] = Field(None, description="분류 적합도 설명")
    rag_context: Optional[str] = Field(None, description="관세청 DB 검색 근거")


class AnalyzeResponse(BaseModel):
    """
    1단계: 입력 분석 응답

    phase 값에 따라 세 가지 시나리오로 분기됩니다:
    - hs_code_selection : HS 코드 후보 3개 반환, 사용자가 선택 후 /calculate 호출
    - complete          : HS 코드가 입력에 이미 포함 → 전체 파이프라인 완료, 보고서 즉시 반환
    - need_more_info    : 필수 정보 누락, missing_info 참고 후 재요청
    """
    success: bool = Field(..., description="처리 성공 여부")
    session_id: str = Field(..., description="세션 ID")
    phase: str = Field(..., description="현재 단계 (hs_code_selection / complete / need_more_info / error)")

    # 추출된 정보
    item_name: Optional[str] = Field(None, description="물품명")
    quantity: Optional[int] = Field(None, description="수량")
    quantity_unit: Optional[str] = Field(None, description="수량 단위")
    unit_price: Optional[float] = Field(None, description="단가")
    price_unit: Optional[str] = Field(None, description="단가 기준")
    total_foreign_price: Optional[float] = Field(None, description="총 외화 금액")
    currency: Optional[str] = Field(None, description="통화")

    # HITL: HS 코드 후보 3개 (phase=hs_code_selection 시)
    hs_code_candidates: Optional[List[HSCodeCandidate]] = Field(
        None, description="HS 코드 후보 목록 (최대 3개, phase=hs_code_selection 시 포함)"
    )

    # 환율
    exchange_rate: Optional[float] = Field(None, description="적용 환율")

    # 추가 정보 필요 시
    missing_info: Optional[List[str]] = Field(None, description="누락된 정보 목록")

    # phase=complete 시: 전체 계산 결과 + 보고서
    hs_code: Optional[str] = Field(None, description="HS 코드 (phase=complete 시 포함)")
    hs_code_rationale: Optional[str] = Field(None, description="HS 코드 분류 근거")
    tariff_rate: Optional[float] = Field(None, description="관세율 (%)")
    total_krw: Optional[float] = Field(None, description="총 원화 환산 금액")
    tax_amount: Optional[float] = Field(None, description="예상 관세액")
    vat_amount: Optional[float] = Field(None, description="예상 부가세")
    total_cost: Optional[float] = Field(None, description="총 예상 비용 (관세 + 부가세 포함)")
    report_content: Optional[str] = Field(None, description="보고서 내용")
    report_paths: Optional[Dict[str, str]] = Field(None, description="보고서 파일 경로 (pdf/word/excel)")

    # 메시지
    message: Optional[str] = Field(None, description="안내 메시지")
    error: Optional[str] = Field(None, description="오류 메시지")


# ===== 2단계: HS 코드 선택 후 비용 계산 =====

class CalculateRequest(BaseModel):
    """2단계: 비용 계산 요청 (HS 코드 선택 후)"""
    session_id: str = Field(..., description="세션 ID (1단계에서 받은 값)")
    selected_hs_code: str = Field(..., description="사용자가 선택한 HS 코드")
    # 필요 시 입력값을 덮어쓰기 (세션 미스매치/추가 정보 보정용)
    item_name: Optional[str] = Field(None, description="물품명 (옵션)")
    quantity: Optional[int] = Field(None, description="수량 (옵션)")
    quantity_unit: Optional[str] = Field(None, description="수량 단위 (옵션)")
    unit_price: Optional[float] = Field(None, description="단가 (옵션)")
    price_unit: Optional[str] = Field(None, description="단가 기준 (옵션)")
    total_foreign_price: Optional[float] = Field(None, description="총 외화 금액 (옵션)")
    currency: Optional[str] = Field(None, description="통화 (옵션)")


class CalculateResponse(BaseModel):
    """2단계: 비용 계산 응답"""
    success: bool = Field(..., description="처리 성공 여부")
    session_id: str = Field(..., description="세션 ID")
    
    # 입력 정보
    item_name: Optional[str] = Field(None, description="물품명")
    quantity: Optional[int] = Field(None, description="수량")
    quantity_unit: Optional[str] = Field(None, description="수량 단위")
    unit_price: Optional[float] = Field(None, description="단가")
    price_unit: Optional[str] = Field(None, description="단가 기준")
    total_foreign_price: Optional[float] = Field(None, description="총 외화 금액")
    currency: Optional[str] = Field(None, description="통화")
    
    # 분석 결과
    hs_code: Optional[str] = Field(None, description="선택된 HS 코드")
    hs_code_rationale: Optional[str] = Field(None, description="HS 코드 분류 근거")
    tariff_rate: Optional[float] = Field(None, description="관세율 (%)")
    exchange_rate: Optional[float] = Field(None, description="적용 환율")
    
    # 비용 계산 결과
    total_krw: Optional[float] = Field(None, description="총 물품가격 (원화)")
    tax_amount: Optional[float] = Field(None, description="예상 관세액")
    vat_amount: Optional[float] = Field(None, description="예상 부가세")
    total_cost: Optional[float] = Field(None, description="총 예상 비용")
    
    # 보고서
    report_content: Optional[str] = Field(None, description="보고서 내용")
    report_paths: Optional[Dict[str, str]] = Field(None, description="보고서 파일 경로 (pdf, word, excel)")
    
    # 메시지
    message: Optional[str] = Field(None, description="안내 메시지")
    error: Optional[str] = Field(None, description="오류 메시지")


# ===== 기타 =====

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = "healthy"
    service: str = "tradesimple-api"
    version: str = "2.0.0"  # HITL 지원 버전


class SessionResponse(BaseModel):
    """세션 상태 조회 응답"""
    session_id: str
    phase: str
    item_name: Optional[str] = None
    hs_code: Optional[str] = None
    total_cost: Optional[float] = None
    created_at: str
    updated_at: str


# ===== 레거시 호환 (기존 API) =====

class ImportCostRequest(BaseModel):
    """[레거시] 수입 비용 계산 요청 (end-to-end, HITL 없음)"""
    message: str = Field(
        ...,
        description="사용자 입력 메시지",
        examples=["미국에서 스마트워치 100개를 개당 300달러에 수입하려고 합니다."]
    )
    session_id: Optional[str] = Field(default=None, description="세션 ID")


class ImportCostResponse(BaseModel):
    """[레거시] 수입 비용 계산 응답 (end-to-end, HITL 없음)"""
    success: bool
    session_id: str
    current_phase: Optional[str] = None
    missing_info: Optional[List[str]] = None
    item_name: Optional[str] = None
    quantity: Optional[int] = None
    unit_price: Optional[float] = None
    currency: Optional[str] = None
    hs_code: Optional[str] = None
    hs_code_rationale: Optional[str] = None
    tariff_rate: Optional[float] = None
    exchange_rate: Optional[float] = None
    tax_amount: Optional[float] = None
    total_cost: Optional[float] = None
    report_content: Optional[str] = None
    report_paths: Optional[Dict[str, str]] = None
    assistant_message: Optional[str] = None
    error: Optional[str] = None
