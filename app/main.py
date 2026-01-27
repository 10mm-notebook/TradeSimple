# app/main.py
"""
TradeSimple - 수입업무 간편화 AI 도우미
Streamlit 기반 웹 인터페이스
"""
import streamlit as st
import asyncio
import os
from langchain_core.messages import HumanMessage, AIMessage
from app.graph import run_agent, get_initial_state
from app.state import FIELD_NAMES_KR

# 페이지 설정
st.set_page_config(
    page_title="TradeSimple - 수입 비용 계산 AI",
    page_icon="🚢",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown('<div class="main-header">🚢 TradeSimple</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">수입업무 간편화 AI 도우미 - HS코드 분류 및 관세 계산</div>', unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.header("📋 사용 가이드")
    st.markdown("""
    **입력 예시:**
    - "미국에서 스마트워치 100개를 개당 300달러에 수입하려고 합니다."
    - "냉동 참치 500kg를 일본에서 kg당 50달러에 수입합니다."
    - "중국에서 노트북 50대를 개당 800달러에 수입하려고 해요."
    
    **필요한 정보:**
    - 물품명 (수입하려는 상품)
    - 수량 (개수 또는 중량)
    - 단가 (개당/kg당 가격)
    - 통화 (달러, 엔, 유로 등)
    """)
    
    st.divider()
    
    st.header("📊 현재 상태")
    if "agent_state" in st.session_state:
        state = st.session_state.agent_state
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("물품명", state.get("item_name") or "-")
            st.metric("수량", f"{state.get('quantity'):,}개" if state.get("quantity") else "-")
        with col2:
            st.metric("단가", f"{state.get('unit_price'):,.2f} {state.get('currency', '')}" if state.get("unit_price") else "-")
            st.metric("HS 코드", state.get("hs_code") or "-")
        
        if state.get("total_cost"):
            st.success(f"💰 총 예상 비용: {state['total_cost']:,.0f} 원")
    else:
        st.info("아직 입력된 정보가 없습니다.")
    
    st.divider()
    
    if st.button("🔄 새로운 분석 시작", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_state = get_initial_state()
        st.rerun()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = get_initial_state()

# 이전 대화 기록 표시
for i, message in enumerate(st.session_state.messages):
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)
        
        # 보고서 다운로드 버튼
        if role == "assistant":
            additional_kwargs = getattr(message, 'additional_kwargs', {})
            report_paths = additional_kwargs.get("report_paths", {})
            
            if report_paths:
                st.divider()
                st.subheader("📥 보고서 다운로드")
                
                cols = st.columns(3)
                
                # PDF 다운로드
                if "pdf" in report_paths and os.path.exists(report_paths["pdf"]):
                    with cols[0]:
                        with open(report_paths["pdf"], "rb") as f:
                            st.download_button(
                                label="📄 PDF 다운로드",
                                data=f.read(),
                                file_name="import_cost_report.pdf",
                                mime="application/pdf",
                                key=f"pdf_{i}",
                                use_container_width=True
                            )
                
                # Word 다운로드
                if "word" in report_paths and os.path.exists(report_paths["word"]):
                    with cols[1]:
                        with open(report_paths["word"], "rb") as f:
                            st.download_button(
                                label="📝 Word 다운로드",
                                data=f.read(),
                                file_name="import_cost_report.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"word_{i}",
                                use_container_width=True
                            )
                
                # Excel 다운로드
                if "excel" in report_paths and os.path.exists(report_paths["excel"]):
                    with cols[2]:
                        with open(report_paths["excel"], "rb") as f:
                            st.download_button(
                                label="📊 Excel 다운로드",
                                data=f.read(),
                                file_name="import_cost_report.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"excel_{i}",
                                use_container_width=True
                            )

# 사용자 입력 처리
if prompt := st.chat_input("수입하려는 물품의 정보를 입력해주세요 (물품명, 수량, 단가, 통화)"):
    # 사용자 메시지 표시
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.status("🤖 AI 에이전트가 작업을 시작합니다...", expanded=True) as status:
            try:
                # 비동기 실행
                async def process():
                    return await run_agent(
                        user_input=prompt,
                        current_state=st.session_state.agent_state
                    )
                
                # 단계별 상태 표시
                status.update(label="📥 입력 정보를 분석합니다...")
                
                # 에이전트 실행
                result_state = asyncio.run(process())
                
                # 상태 업데이트
                st.session_state.agent_state = result_state
                
                # 진행 상황 표시
                if result_state.get("hs_code"):
                    status.update(label=f"🔍 HS 코드 확인: {result_state['hs_code']}")
                
                if result_state.get("exchange_rate"):
                    status.update(label=f"💱 환율 조회 완료: {result_state['exchange_rate']:,.2f} KRW/{result_state.get('currency', 'USD')}")
                
                if result_state.get("total_cost"):
                    status.update(label=f"💰 비용 계산 완료: {result_state['total_cost']:,.0f}원")
                
                if result_state.get("report_paths"):
                    status.update(label="📝 보고서 생성 완료!", state="complete")
                elif result_state.get("missing_info"):
                    status.update(label="⏳ 추가 정보가 필요합니다", state="complete")
                else:
                    status.update(label="✅ 처리 완료", state="complete")
                
                # 응답 메시지 추출 및 표시
                messages = result_state.get("messages", [])
                response_content = ""
                report_paths = {}
                
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        response_content = msg.content
                        report_paths = getattr(msg, 'additional_kwargs', {}).get("report_paths", {})
                
                if response_content:
                    st.markdown(response_content)
                    
                    # 보고서 다운로드 버튼
                    if report_paths:
                        st.divider()
                        st.subheader("📥 보고서 다운로드")
                        
                        cols = st.columns(3)
                        
                        if "pdf" in report_paths and os.path.exists(report_paths["pdf"]):
                            with cols[0]:
                                with open(report_paths["pdf"], "rb") as f:
                                    st.download_button(
                                        label="📄 PDF 다운로드",
                                        data=f.read(),
                                        file_name="import_cost_report.pdf",
                                        mime="application/pdf",
                                        key="pdf_new",
                                        use_container_width=True
                                    )
                        
                        if "word" in report_paths and os.path.exists(report_paths["word"]):
                            with cols[1]:
                                with open(report_paths["word"], "rb") as f:
                                    st.download_button(
                                        label="📝 Word 다운로드",
                                        data=f.read(),
                                        file_name="import_cost_report.docx",
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        key="word_new",
                                        use_container_width=True
                                    )
                        
                        if "excel" in report_paths and os.path.exists(report_paths["excel"]):
                            with cols[2]:
                                with open(report_paths["excel"], "rb") as f:
                                    st.download_button(
                                        label="📊 Excel 다운로드",
                                        data=f.read(),
                                        file_name="import_cost_report.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="excel_new",
                                        use_container_width=True
                                    )
                    
                    # 세션에 메시지 저장
                    final_message = AIMessage(
                        content=response_content,
                        additional_kwargs={"report_paths": report_paths}
                    )
                    st.session_state.messages.append(final_message)
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# 푸터
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    TradeSimple - 수입업무 간편화 AI 도우미 | K Intelligence 해커톤 2025
</div>
""", unsafe_allow_html=True)
