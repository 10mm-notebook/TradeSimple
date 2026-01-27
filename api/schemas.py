# api/schemas.py
"""
API 요청/응답 스키마 정의
"""
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


class ImportCostRequest(BaseModel):
    """수입 비용 계산 요청 스키마"""
    message: str = Field(
        ...,
        description="사용자 입력 메시지 (물품명, 수량, 단가, 통화 포함)",
        examples=["미국에서 스마트워치 100개를 개당 300달러에 수입하려고 합니다."]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="세션 ID (대화 지속 시 사용)"
    )


class ImportCostResponse(BaseModel):
    """수입 비용 계산 응답 스키마"""
    success: bool = Field(..., description="처리 성공 여부")
    session_id: str = Field(..., description="세션 ID")
    
    # 상태 정보
    current_phase: Optional[str] = Field(None, description="현재 처리 단계")
    missing_info: Optional[List[str]] = Field(None, description="누락된 정보 목록")
    
    # 추출된 정보
    item_name: Optional[str] = Field(None, description="물품명")
    quantity: Optional[int] = Field(None, description="수량")
    unit_price: Optional[float] = Field(None, description="단가")
    currency: Optional[str] = Field(None, description="통화")
    
    # 분석 결과
    hs_code: Optional[str] = Field(None, description="HS 코드")
    hs_code_rationale: Optional[str] = Field(None, description="HS 코드 분류 근거")
    tariff_rate: Optional[float] = Field(None, description="관세율 (%)")
    exchange_rate: Optional[float] = Field(None, description="적용 환율")
    tax_amount: Optional[float] = Field(None, description="예상 관세액")
    total_cost: Optional[float] = Field(None, description="총 예상 비용")
    
    # 보고서
    report_content: Optional[str] = Field(None, description="보고서 내용")
    report_paths: Optional[Dict[str, str]] = Field(None, description="생성된 보고서 파일 경로")
    
    # 메시지
    assistant_message: Optional[str] = Field(None, description="AI 응답 메시지")
    error: Optional[str] = Field(None, description="오류 메시지")


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = "healthy"
    service: str = "tradesimple-api"
    version: str = "1.0.0"


class SessionState(BaseModel):
    """세션 상태 저장용"""
    state: Dict[str, Any]
    created_at: str
    updated_at: str
