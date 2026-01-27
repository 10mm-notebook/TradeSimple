# app/agents/tax_calculator.py
"""
Tax Calculator Agent
- 진짜 ReAct 패턴: LLM이 환율 조회 및 비용 계산 수행
- LLM을 활용한 계산 검증 및 설명 생성
"""
import asyncio
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from app.tools import exchange_rate_loader, final_cost_calculator


# ReAct 에이전트용 시스템 프롬프트
TAX_CALCULATOR_SYSTEM_PROMPT = """당신은 수입 비용 계산 전문가입니다. 환율을 조회하고 관세, 부가세를 포함한 총 수입 비용을 정확하게 계산해야 합니다.

## 사용 가능한 도구
1. **exchange_rate_loader**: 특정 통화와 KRW 사이의 실시간 환율을 조회합니다.
2. **final_cost_calculator**: 단가, 수량, 환율, 관세율을 입력받아 총 비용을 계산합니다.

## 계산 공식
1. 총 물품가격(원화) = 단가 × 수량 × 환율
2. 관세 = 총 물품가격(원화) × 관세율
3. 부가세 = (총 물품가격 + 관세) × 10%
4. 총 비용 = 총 물품가격 + 관세 + 부가세

## 작업 순서
1. 먼저 exchange_rate_loader 도구로 해당 통화의 환율을 조회하세요.
2. final_cost_calculator 도구로 총 비용을 계산하세요.
3. 계산 결과를 검증하고 설명을 추가하세요.

## 중요 사항
- 환율은 실시간 데이터를 사용합니다.
- 계산 결과를 반드시 검증하세요.
- 각 단계별 금액을 명확히 보여주세요.

## 최종 응답 형식
작업이 완료되면 반드시 다음 형식으로 응답하세요:

[계산 결과]
- 적용 환율: [환율] KRW/[통화]
- 총 물품가격(원화): [금액]원
- 예상 관세: [금액]원
- 예상 부가세: [금액]원
- 총 예상 비용: [금액]원
- 검증: [계산이 올바른지 확인 결과]
"""


class TaxCalculatorAgent:
    """
    Tax Calculator 에이전트 (진짜 ReAct 패턴)
    
    LangGraph의 create_react_agent를 사용하여
    LLM이 스스로 환율 조회 및 비용 계산을 수행하고 결과를 검증합니다.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [exchange_rate_loader, final_cost_calculator]
        
        # ReAct 에이전트 생성
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            state_modifier=TAX_CALCULATOR_SYSTEM_PROMPT,
        )
    
    async def run(
        self,
        unit_price: float,
        quantity: int,
        currency: str,
        tariff_rate: float
    ) -> Dict[str, Any]:
        """
        ReAct 패턴으로 환율 조회 및 비용 계산 실행
        
        Args:
            unit_price: 물품 단가 (외화)
            quantity: 수량
            currency: 통화 코드 (USD, EUR 등)
            tariff_rate: 관세율 (%)
            
        Returns:
            {
                "exchange_rate": float,
                "exchange_source": str,
                "tax_amount": float,
                "vat_amount": float,
                "total_cost": float,
                "breakdown": str,
                "agent_messages": List[str]
            }
        """
        print(f"[TaxCalculatorAgent] ReAct 실행 시작: {quantity}개 × {unit_price} {currency}, 관세율 {tariff_rate}%")
        
        # ReAct 에이전트 실행
        input_message = HumanMessage(content=f"""다음 수입 물품의 총 비용을 계산해주세요:

- 단가: {unit_price} {currency}
- 수량: {quantity}개
- 적용 관세율: {tariff_rate}%

먼저 {currency} 환율을 조회한 후, 총 비용을 계산해주세요.""")
        
        result = await self.agent.ainvoke({
            "messages": [input_message]
        })
        
        # 에이전트 메시지에서 결과 추출
        messages = result.get("messages", [])
        agent_messages = []
        final_response = ""
        
        # 도구 호출 결과에서 실제 값 추출
        exchange_rate = None
        total_cost = None
        tax_amount = None
        vat_amount = None
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                agent_messages.append(f"[AI] {msg.content[:200]}..." if len(msg.content) > 200 else f"[AI] {msg.content}")
                final_response = msg.content
                
                # tool_calls 결과 확인
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get('name') == 'exchange_rate_loader':
                            pass  # 결과는 ToolMessage에서 확인
            
            # ToolMessage에서 실제 값 추출
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                if '"rate":' in msg.content:
                    import json
                    try:
                        data = json.loads(msg.content)
                        if 'rate' in data:
                            exchange_rate = data['rate']
                    except:
                        pass
                if '"total_cost":' in msg.content:
                    import json
                    try:
                        data = json.loads(msg.content)
                        if 'total_cost' in data:
                            total_cost = data['total_cost']
                            tax_amount = data.get('tariff_amount', 0)
                            vat_amount = data.get('vat_amount', 0)
                    except:
                        pass
        
        # 결과가 없으면 직접 계산
        if exchange_rate is None:
            exchange_result = exchange_rate_loader.invoke({"target_currency": currency})
            exchange_rate = exchange_result["rate"]
        
        if total_cost is None:
            cost_result = final_cost_calculator.invoke({
                "item_price": unit_price,
                "quantity": quantity,
                "exchange_rate": exchange_rate,
                "tariff_rate": tariff_rate
            })
            total_cost = cost_result["total_cost"]
            tax_amount = cost_result["tariff_amount"]
            vat_amount = cost_result["vat_amount"]
            breakdown = cost_result["breakdown"]
        else:
            total_krw = unit_price * quantity * exchange_rate
            breakdown = f"""--- 최종 비용 계산 결과 ---
1. 총 물품 가격 (원화): {total_krw:,.0f} 원
   (단가 {unit_price:,.2f} × 수량 {quantity} × 환율 {exchange_rate:,.2f})
2. 예상 관세 ({tariff_rate}%): {tax_amount:,.0f} 원
3. 예상 부가세 (10%): {vat_amount:,.0f} 원
--------------------------------
   총 예상 수입 비용: {total_cost:,.0f} 원
--------------------------------"""
        
        result = {
            "exchange_rate": exchange_rate,
            "exchange_source": "exchangerate-api.com",
            "currency": currency,
            "tax_amount": tax_amount,
            "vat_amount": vat_amount,
            "total_cost": total_cost,
            "breakdown": breakdown,
            "agent_messages": agent_messages,
        }
        
        print(f"[TaxCalculatorAgent] ReAct 완료: 총 {result['total_cost']:,.0f}원")
        return result
