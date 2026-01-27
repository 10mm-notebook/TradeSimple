# app/agents/hs_code_finder.py
"""
HS Code & Tax Finder Agent
- 진짜 ReAct 패턴: LLM이 스스로 도구를 선택하고 실행
- langgraph.prebuilt.create_react_agent 활용
"""
import re
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from app.tools import hs_code_search, tariff_search_by_hs_code


# ReAct 에이전트용 시스템 프롬프트
HS_CODE_FINDER_SYSTEM_PROMPT = """당신은 HS 코드 분류 전문가입니다. 사용자가 제공한 물품 설명을 바탕으로 정확한 HS 코드를 찾고 관세율을 조회해야 합니다.

## 사용 가능한 도구
1. **hs_code_search**: 물품 설명으로 HS 코드 정보를 검색합니다.
2. **tariff_search_by_hs_code**: HS 코드로 관세율을 조회합니다.

## 작업 순서
1. 먼저 hs_code_search 도구로 물품의 HS 코드를 검색하세요.
2. 검색 결과에서 가장 적합한 HS 코드를 선택하세요.
3. tariff_search_by_hs_code 도구로 해당 HS 코드의 관세율을 조회하세요.
4. 최종 결과를 정리하여 반환하세요.

## 중요 사항
- HS 코드는 10자리 형식입니다 (예: 0303.43-0000, 8517.62-9090)
- 검색 결과가 없거나 불명확하면, 물품 특성에 기반하여 가장 유사한 분류를 추정하세요.
- 반드시 관세율까지 조회한 후 응답하세요.

## 최종 응답 형식
작업이 완료되면 반드시 다음 형식으로 응답하세요:

[결과]
- HS 코드: [10자리 HS 코드]
- 관세율: [숫자]%
- 분류 근거: [왜 이 HS 코드가 적합한지 설명]
"""


class HSCodeFinderAgent:
    """
    HS Code & Tax Finder 에이전트 (진짜 ReAct 패턴)
    
    LangGraph의 create_react_agent를 사용하여
    LLM이 스스로 Thought → Action → Observation 루프를 수행합니다.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [hs_code_search, tariff_search_by_hs_code]
        
        # ReAct 에이전트 생성
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            state_modifier=HS_CODE_FINDER_SYSTEM_PROMPT,
        )
    
    async def run(self, item_name: str) -> Dict[str, Any]:
        """
        ReAct 패턴으로 HS 코드 검색 및 관세율 조회 실행
        
        Args:
            item_name: 검색할 물품명
            
        Returns:
            {
                "hs_code": str,
                "tariff_rate": float,
                "rationale": str,
                "agent_messages": List[str]
            }
        """
        print(f"[HSCodeFinderAgent] ReAct 실행 시작: {item_name}")
        
        # ReAct 에이전트 실행
        input_message = HumanMessage(content=f"다음 물품의 HS 코드를 찾고 관세율을 조회해주세요: {item_name}")
        
        result = await self.agent.ainvoke({
            "messages": [input_message]
        })
        
        # 에이전트 메시지에서 결과 추출
        messages = result.get("messages", [])
        agent_messages = []
        final_response = ""
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                agent_messages.append(f"[AI] {msg.content[:200]}..." if len(msg.content) > 200 else f"[AI] {msg.content}")
                final_response = msg.content
        
        # 결과 파싱
        hs_code = self._extract_hs_code(final_response)
        tariff_rate = self._extract_tariff_rate(final_response)
        rationale = self._extract_rationale(final_response)
        
        result = {
            "hs_code": hs_code or "미확인",
            "tariff_rate": tariff_rate,
            "rationale": rationale or f"'{item_name}'에 대한 HS 코드 분류 결과",
            "agent_messages": agent_messages,
        }
        
        print(f"[HSCodeFinderAgent] ReAct 완료: HS Code={result['hs_code']}, Tariff={result['tariff_rate']}%")
        return result
    
    def _extract_hs_code(self, text: str) -> Optional[str]:
        """텍스트에서 HS 코드 추출"""
        patterns = [
            r'HS\s*코드[:\s]*([0-9]{4}\.[0-9]{2}-[0-9]{4})',
            r'HS\s*코드[:\s]*([0-9]{4}\.[0-9]{2}\.[0-9]{4})',
            r'\b([0-9]{4}\.[0-9]{2}-[0-9]{4})\b',
            r'\b([0-9]{4}\.[0-9]{2}\.[0-9]{4})\b',
            r"'([0-9]{4}\.[0-9]{2}-[0-9]{4})'",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_tariff_rate(self, text: str) -> float:
        """관세율 추출"""
        patterns = [
            r'관세율[:\s]*([0-9.]+)\s*%',
            r'최종[^:]*세율[:\s]*([0-9.]+)\s*%',
            r'([0-9.]+)\s*%\s*(?:관세|세율)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        
        # 무세 체크
        if '무세' in text or '0%' in text:
            return 0.0
        
        return 0.0
    
    def _extract_rationale(self, text: str) -> Optional[str]:
        """분류 근거 추출"""
        patterns = [
            r'분류\s*근거[:\s]*(.+?)(?:\n|$)',
            r'근거[:\s]*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
        
        return None
