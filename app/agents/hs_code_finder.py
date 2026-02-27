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
from app.tools import hs_code_search, tariff_search_by_hs_code, reset_hs_code_search_limit


# ReAct 에이전트용 시스템 프롬프트 (후보 3개 반환용 - 관세율 조회 없이 HS 코드만)
HS_CODE_FINDER_CANDIDATES_PROMPT = """당신은 HS 코드 분류 전문가입니다. 사용자가 제공한 물품 설명을 바탕으로 **가장 적합한 HS 코드 후보 3개**를 찾아야 합니다.

## 사용 가능한 도구
1. **hs_code_search**: 물품 설명으로 HS 코드 정보를 검색합니다.

## 중요: 관세율 조회 금지!
- **tariff_search_by_hs_code 도구를 호출하지 마세요.** 관세율은 사용자가 HS 코드를 선택한 후에 별도로 조회됩니다.
- hs_code_search 도구만 사용하여 HS 코드 후보를 찾으세요.

## 검색 키워드 규칙
- 핵심 단어 2개 정도로 검색하세요 (예: "냉동 참치", "스마트워치")
- 여러 키워드로 검색해서 다양한 후보를 찾으세요.

## 작업 순서
1. hs_code_search로 물품의 HS 코드를 검색합니다.
2. 검색 결과에서 **가장 적합한 후보 3개**를 선별합니다.
3. 각 후보의 관세율을 tariff_search_by_hs_code로 조회합니다.
4. 3개 후보를 정확도 순으로 정렬하여 반환합니다.

## 최종 응답 형식 (필수!)
반드시 아래 형식으로 **3개 후보**를 응답하세요:

[후보1] (가장 유력)
- HS 코드: XXXX.XX-XXXX
- 관세율: X%
- 품명: [해당 HS 코드의 품명]
- 적합도: [왜 이 코드가 적합한지 한 줄 설명]

[후보2]
- HS 코드: XXXX.XX-XXXX
- 관세율: X%
- 품명: [해당 HS 코드의 품명]
- 적합도: [왜 이 코드가 적합한지 한 줄 설명]

[후보3]
- HS 코드: XXXX.XX-XXXX
- 관세율: X%
- 품명: [해당 HS 코드의 품명]
- 적합도: [왜 이 코드가 적합한지 한 줄 설명]
"""

# ReAct 에이전트용 시스템 프롬프트
HS_CODE_FINDER_SYSTEM_PROMPT = """당신은 HS 코드 분류 전문가입니다. 사용자가 제공한 물품 설명을 바탕으로 정확한 HS 코드를 찾고 관세율을 조회해야 합니다.

## 사용 가능한 도구
1. **hs_code_search**: 물품 설명으로 HS 코드 정보를 검색합니다.
2. **tariff_search_by_hs_code**: HS 코드로 관세율을 조회합니다.

## 검색 키워드 규칙 (중요)
- 사용자가 긴 문장으로 설명하더라도, 실제 도구에 전달하는 검색어는 항상 **아주 짧은 키워드 2개 정도**여야 합니다.
- 우선순위:
  1) 한국어 상품명 기준으로 핵심 단어 2개 (예: "미국산 냉동 참치 수입" → "냉동 참치")
  2) 필요 시 간단한 영문 2단어 키워드를 보조적으로 추가 (예: "냉동 참치" + "frozen tuna")
- 문장 전체를 그대로 검색어로 쓰지 말고, 기능/용도/재질/형태를 가장 잘 나타내는 2개 정도의 단어만 골라서 사용하세요.

## 작업 순서
1. 먼저 hs_code_search 도구로 물품의 HS 코드를 검색하세요.
2. 검색 결과에서 가장 적합한 HS 코드를 선택하세요.
3. tariff_search_by_hs_code 도구로 해당 HS 코드의 관세율을 조회하세요.
4. 최종 결과를 정리하여 반환하세요.

## 중요 사항
- HS 코드는 10자리 형식입니다 (예: 0303.43-0000, 8517.62-9090)
- 검색 결과가 없거나 불명확하면, 물품 특성에 기반하여 가장 유사한 분류를 추정하세요.
- 반드시 관세율까지 조회한 후 응답하세요.

## 도구 호출 제한 (중요)
- 동일한 검색어로 **hs_code_search 도구를 3회 초과 호출하지 마세요.**
- 같은 검색어로 3회 호출 후에도 확신이 없으면, 그 안에서 가장 유력한 후보를 선택하고 다음 단계(tariff_search_by_hs_code)로 진행해야 합니다.
- 불필요하게 같은 도구를 반복 호출하지 말고, 가능한 적은 호출로 최선의 HS 코드를 결정하세요.

## 최종 응답 형식 (필수 준수)
작업이 완료되면 반드시 아래 형식으로만 응답하세요. HS 코드는 반드시 **한 줄에** 다음 중 하나의 형식으로 기재합니다.

- **HS 코드: XXXX.XX-XXXX**  (예: HS 코드: 0303.43-0000, HS 코드: 8517.62-9090)
- 또는 숫자만 10자리: **XXXXXXXXXX** (예: 0303430000, 8517629090)

형식 규칙:
- 10자리: 앞 4자리.중간 2자리-뒤 4자리 (점과 하이픈 사용) 또는 연속 10자리 숫자.
- 다른 형식(괄호, 따옴표만 있는 경우 등)으로 쓰지 말고, 위 두 가지 중 하나로 명확히 적어주세요.

[결과]
- HS 코드: XXXX.XX-XXXX (또는 10자리 숫자)
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
        # 비용과 레이트 리밋을 고려해 gpt-4o-mini 사용
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools = [hs_code_search, tariff_search_by_hs_code]
        
        # ReAct 에이전트 생성
        # 설치된 langgraph 버전에서는 state_modifier 인자를 지원하지 않으므로,
        # 시스템 프롬프트는 run()에서 SystemMessage로 주입한다.
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
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

        # 도구 호출 카운터 리셋 (에이전트 1회 실행당 3회 제한 적용)
        reset_hs_code_search_limit()
        
        # ReAct 에이전트 실행
        # 시스템 프롬프트를 SystemMessage로 명시적으로 추가
        input_message = HumanMessage(
            content=f"다음 물품의 HS 코드를 찾고 관세율을 조회해주세요: {item_name}"
        )
        result = await self.agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=HS_CODE_FINDER_SYSTEM_PROMPT),
                    input_message,
                ]
            }
        )
        
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
    
    async def run_with_candidates(self, item_name: str) -> Dict[str, Any]:
        """
        HS 코드 후보 3개를 반환 (Human-in-the-Loop용)
        
        Args:
            item_name: 검색할 물품명
            
        Returns:
            {
                "candidates": [
                    {"hs_code": str, "tariff_rate": float, "품명": str, "적합도": str},
                    ...
                ],
                "agent_messages": List[str]
            }
        """
        print(f"[HSCodeFinderAgent] 후보 검색 시작: {item_name}")

        reset_hs_code_search_limit()
        
        input_message = HumanMessage(
            content=f"다음 물품의 HS 코드 후보 3개를 찾아주세요: {item_name}"
        )
        result = await self.agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=HS_CODE_FINDER_CANDIDATES_PROMPT),
                    input_message,
                ]
            }
        )
        
        messages = result.get("messages", [])
        agent_messages = []
        final_response = ""
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                agent_messages.append(f"[AI] {msg.content[:200]}..." if len(msg.content) > 200 else f"[AI] {msg.content}")
                final_response = msg.content
        
        # 후보 3개 파싱
        candidates = self._extract_candidates(final_response)
        
        # 후보가 3개 미만이면 기본 결과로 보충
        if len(candidates) < 3:
            default_result = await self.run(item_name)
            if default_result["hs_code"] != "미확인":
                found_codes = [c["hs_code"] for c in candidates]
                if default_result["hs_code"] not in found_codes:
                    candidates.append({
                        "hs_code": default_result["hs_code"],
                        "tariff_rate": default_result["tariff_rate"],
                        "품명": item_name,
                        "적합도": default_result["rationale"] or "AI 추천",
                    })
        
        # 최소 1개는 있어야 함
        if not candidates:
            candidates = [{
                "hs_code": "미확인",
                "tariff_rate": 0.0,
                "품명": item_name,
                "적합도": "검색 결과 없음 - 직접 입력 필요",
            }]
        
        print(f"[HSCodeFinderAgent] 후보 {len(candidates)}개 반환")
        return {
            "candidates": candidates[:3],  # 최대 3개
            "agent_messages": agent_messages,
        }
    
    def _extract_candidates(self, text: str) -> List[Dict[str, Any]]:
        """텍스트에서 후보 3개 파싱."""
        candidates = []
        
        # [후보1], [후보2], [후보3] 섹션 파싱
        sections = re.split(r'\[후보[123]\]', text)
        
        for section in sections[1:]:  # 첫 번째는 헤더 이전
            hs_code = self._extract_hs_code(section)
            tariff_rate = self._extract_tariff_rate(section)
            
            # 품명 추출
            품명_match = re.search(r'품명[:\s]*(.+?)(?:\n|$)', section)
            품명 = 품명_match.group(1).strip() if 품명_match else ""
            
            # 적합도 추출
            적합도_match = re.search(r'적합도[:\s]*(.+?)(?:\n|$)', section)
            적합도 = 적합도_match.group(1).strip() if 적합도_match else ""
            
            if hs_code:
                candidates.append({
                    "hs_code": hs_code,
                    "tariff_rate": tariff_rate,
                    "품명": 품명[:50] if 품명 else "",
                    "적합도": 적합도[:100] if 적합도 else "",
                })
        
        return candidates
    
    def _normalize_hs_code(self, raw: str) -> str:
        """추출한 HS 코드를 표준 형식 XXXX.XX-XXXX로 정규화."""
        digits = "".join(c for c in raw if c.isdigit())
        if len(digits) >= 10:
            return f"{digits[:4]}.{digits[4:6]}-{digits[6:10]}"
        if len(digits) == 6:
            return f"{digits[:4]}.{digits[4:6]}-0000"
        if len(digits) == 9:
            return f"{digits[:4]}.{digits[4:6]}-{digits[6:9]}0"
        return raw.strip()

    def _extract_hs_code(self, text: str) -> Optional[str]:
        """텍스트에서 HS 코드를 다양한 형식으로 추출 후 표준 형식으로 반환."""
        # 1) 이미 표준 형식: XXXX.XX-XXXX (HS 코드: 포함 여부 무관)
        patterns_standard = [
            r'HS\s*코드[:\s]*([0-9]{4}\.[0-9]{2}-[0-9]{4})',
            r'HS\s*코드[:\s]*([0-9]{4}\.[0-9]{2}\.[0-9]{4})',
            r'\b([0-9]{4}\.[0-9]{2}-[0-9]{4})\b',
            r'\b([0-9]{4}\.[0-9]{2}\.[0-9]{4})\b',
            r'[\(\'"\s]([0-9]{4}\.[0-9]{2}-[0-9]{4})[\)\'"\s]',
        ]
        for pattern in patterns_standard:
            match = re.search(pattern, text)
            if match:
                return self._normalize_hs_code(match.group(1))

        # 2) 연속 10자리 숫자 (구분자 없음)
        ten_digit = re.search(r'\b([0-9]{10})\b', text)
        if ten_digit:
            return self._normalize_hs_code(ten_digit.group(1))

        # 3) 9자리 숫자 (일부 DB 형식)
        nine_digit = re.search(r'\b([0-9]{9})\b', text)
        if nine_digit:
            return self._normalize_hs_code(nine_digit.group(1))

        # 4) "HS 코드:" 뒤에 오는 숫자만 있는 경우 (공백/줄바꿈 포함)
        after_label = re.search(r'HS\s*코드[:\s]+([0-9.\-]+)', text)
        if after_label:
            return self._normalize_hs_code(after_label.group(1))

        # 5) 6자리 (장·호): XXXX.XX 또는 XXXXXX
        six_digit = re.search(r'\b([0-9]{4}\.[0-9]{2})\b', text)
        if six_digit:
            return self._normalize_hs_code(six_digit.group(1))
        six_plain = re.search(r'\b([0-9]{6})\b', text)
        if six_plain:
            return self._normalize_hs_code(six_plain.group(1))

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
