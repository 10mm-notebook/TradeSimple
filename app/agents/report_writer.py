# app/agents/report_writer.py
"""
Report Writer Agent
- 진짜 ReAct 패턴: LLM이 보고서 내용 구성 및 형식 결정
- asyncio.gather로 PDF/Word/Excel 병렬 생성
"""
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from app.tools import pdf_report_exporter, word_report_exporter, excel_report_exporter


# 보고서 작성용 시스템 프롬프트
REPORT_WRITER_SYSTEM_PROMPT = """당신은 수입 비용 분석 보고서 작성 전문가입니다.
제공된 데이터를 바탕으로 명확하고 전문적인 보고서를 작성해야 합니다.

## 보고서 구성
1. 요약: 핵심 정보를 한눈에 볼 수 있도록
2. 분석 대상: 물품명, 수량, 단가 등 기본 정보
3. HS 코드 분류 결과: HS 코드와 분류 근거
4. 환율 정보: 적용 환율과 출처
5. 비용 계산 결과: 단계별 금액과 총 비용
6. 참고사항: 주의사항 및 면책 조항

## 작성 스타일
- 전문적이고 공식적인 톤 유지
- 숫자는 천 단위 구분 기호 사용
- 중요 정보는 강조 표시
"""


class ReportWriterAgent:
    """
    Report Writer 에이전트 (ReAct 패턴 + 병렬 처리)
    
    LLM을 활용하여 보고서 내용을 구성하고,
    asyncio.gather로 PDF/Word/Excel을 병렬 생성합니다.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.tools = {
            "pdf_report_exporter": pdf_report_exporter,
            "word_report_exporter": word_report_exporter,
            "excel_report_exporter": excel_report_exporter,
        }
    
    async def run(
        self,
        item_name: str,
        quantity: int,
        unit_price: float,
        currency: str,
        hs_code: str,
        hs_code_rationale: str,
        tariff_rate: float,
        exchange_rate: float,
        exchange_source: str,
        tax_amount: float,
        vat_amount: float,
        total_cost: float,
        report_format: str = "all"
    ) -> Dict[str, Any]:
        """
        LLM으로 보고서 내용 생성 후 병렬로 파일 생성
        
        Args:
            item_name: 물품명
            ... (기타 파라미터)
            report_format: 보고서 형식 (all/pdf/word/excel)
            
        Returns:
            {
                "report_content": str,
                "report_paths": dict,
                "export_results": list
            }
        """
        print(f"[ReportWriterAgent] 실행 시작: {item_name}")
        
        # Step 1: LLM을 사용하여 보고서 내용 생성
        total_foreign = unit_price * quantity
        total_krw = total_foreign * exchange_rate
        
        report_data = {
            "date": datetime.now().strftime("%Y년 %m월 %d일"),
            "item_name": item_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "currency": currency,
            "total_foreign": total_foreign,
            "hs_code": hs_code,
            "hs_code_rationale": hs_code_rationale,
            "tariff_rate": tariff_rate,
            "exchange_rate": exchange_rate,
            "exchange_source": exchange_source,
            "total_krw": total_krw,
            "tax_amount": tax_amount,
            "vat_amount": vat_amount,
            "total_cost": total_cost,
        }
        
        # LLM으로 보고서 내용 생성
        report_content = await self._generate_report_content(report_data)
        
        print(f"[ReportWriterAgent] 보고서 내용 생성 완료")
        
        # Step 2: 병렬로 모든 형식 파일 생성 (asyncio.gather)
        report_paths = {}
        export_results = []
        
        # Excel용 데이터
        excel_data = {
            "물품명": item_name,
            "수량": quantity,
            "단가": unit_price,
            "통화": currency,
            "총물품가격(외화)": total_foreign,
            "HS코드": hs_code,
            "관세율(%)": tariff_rate,
            "환율": exchange_rate,
            "총물품가격(원화)": total_krw,
            "예상관세(원)": tax_amount,
            "예상부가세(원)": vat_amount,
            "총예상비용(원)": total_cost,
        }
        
        if report_format == "all":
            # 🔥 병렬로 모든 형식 생성 (핵심!)
            print(f"[ReportWriterAgent] PDF/Word/Excel 병렬 생성 시작...")
            
            tasks = [
                self._export_pdf_async(report_content),
                self._export_word_async(report_content),
                self._export_excel_async(excel_data),
            ]
            
            # asyncio.gather로 동시 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    export_results.append(result)
                    if result.get("success"):
                        report_paths[result["format"]] = result["path"]
                else:
                    export_results.append({"success": False, "error": str(result)})
            
            print(f"[ReportWriterAgent] 병렬 생성 완료: {list(report_paths.keys())}")
        else:
            # 단일 형식 생성
            if report_format == "pdf":
                result = await self._export_pdf_async(report_content)
            elif report_format == "word":
                result = await self._export_word_async(report_content)
            elif report_format == "excel":
                result = await self._export_excel_async(excel_data)
            else:
                result = {"success": False, "error": f"Unknown format: {report_format}"}
            
            export_results.append(result)
            if result.get("success"):
                report_paths[result["format"]] = result["path"]
        
        return {
            "report_content": report_content,
            "report_paths": report_paths,
            "export_results": export_results,
        }
    
    async def _generate_report_content(self, data: Dict[str, Any]) -> str:
        """LLM을 사용하여 보고서 내용 생성"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", REPORT_WRITER_SYSTEM_PROMPT),
            ("human", """다음 데이터를 바탕으로 수입 비용 분석 보고서를 작성해주세요:

## 기본 정보
- 작성일: {date}
- 물품명: {item_name}
- 수량: {quantity:,}개
- 단가: {unit_price:,.2f} {currency}
- 총 물품가격(외화): {total_foreign:,.2f} {currency}

## HS 코드 분류
- HS 코드: {hs_code}
- 분류 근거: {hs_code_rationale}
- 적용 관세율: {tariff_rate}%

## 환율 정보
- 적용 환율: {exchange_rate:,.2f} KRW/{currency}
- 환율 출처: {exchange_source}

## 비용 계산
- 총 물품가격(원화): {total_krw:,.0f}원
- 예상 관세: {tax_amount:,.0f}원
- 예상 부가세: {vat_amount:,.0f}원
- 총 예상 비용: {total_cost:,.0f}원

마크다운 형식으로 보기 좋게 작성해주세요.""")
        ])
        
        messages = prompt.format_messages(**data)
        response = await self.llm.ainvoke(messages)
        
        return response.content
    
    async def _export_pdf_async(self, content: str) -> Dict[str, Any]:
        """PDF 내보내기 (비동기 래핑)"""
        try:
            # 동기 함수를 비동기로 실행
            loop = asyncio.get_event_loop()
            filename = "report.pdf"
            result = await loop.run_in_executor(
                None,
                lambda: self.tools["pdf_report_exporter"].invoke({
                    "report_content": content,
                    "filename": filename
                })
            )
            return {
                "format": "pdf",
                "path": filename,
                "success": True,
                "message": result
            }
        except Exception as e:
            return {
                "format": "pdf",
                "success": False,
                "error": str(e)
            }
    
    async def _export_word_async(self, content: str) -> Dict[str, Any]:
        """Word 내보내기 (비동기 래핑)"""
        try:
            loop = asyncio.get_event_loop()
            filename = "report.docx"
            result = await loop.run_in_executor(
                None,
                lambda: self.tools["word_report_exporter"].invoke({
                    "report_content": content,
                    "filename": filename
                })
            )
            return {
                "format": "word",
                "path": filename,
                "success": True,
                "message": result
            }
        except Exception as e:
            return {
                "format": "word",
                "success": False,
                "error": str(e)
            }
    
    async def _export_excel_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Excel 내보내기 (비동기 래핑)"""
        try:
            loop = asyncio.get_event_loop()
            filename = "report.xlsx"
            result = await loop.run_in_executor(
                None,
                lambda: self.tools["excel_report_exporter"].invoke({
                    "data": data,
                    "filename": filename
                })
            )
            return {
                "format": "excel",
                "path": filename,
                "success": True,
                "message": result
            }
        except Exception as e:
            return {
                "format": "excel",
                "success": False,
                "error": str(e)
            }
