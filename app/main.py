# app/main.py
"""
TradeSimple - 수입업무 간편화 AI 도우미
Streamlit 기반 웹 인터페이스

🔥 API 중심 아키텍처: 모든 로직은 FastAPI 서버에서 처리
   Streamlit은 순수 UI 레이어로만 동작
"""
import streamlit as st
import httpx
import os

# 환경 변수에서 API 서버 URL 가져오기 (Docker 또는 로컬)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# 페이지 설정
st.set_page_config(
    page_title="TradeSimple - 수입 비용 계산 AI",
    page_icon="🚢",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .status-box { padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .info-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    /* 사이드바 채팅 로그 */
    [data-testid="stSidebar"] .chat-log-item { font-size: 0.8rem; line-height: 1.35; margin: 0.25rem 0; color: #444; }
    [data-testid="stSidebar"] .chat-log-user { font-weight: 600; color: #1f77b4; }
    [data-testid="stSidebar"] .chat-log-asst { font-weight: 500; color: #2ca02c; }
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
    
    # API 서버 상태 표시
    st.caption(f"🔌 API: `{API_BASE_URL}`")
    
    # 새로운 분석 시작 버튼
    if st.button("➕ 새로운 분석 시작", use_container_width=True):
        # 현재 세션에 내용이 있으면 히스토리에 저장
        if st.session_state.get("chat_history"):
            if "session_history" not in st.session_state:
                st.session_state.session_history = []
            # 세션 요약
            first_item = st.session_state.get("current_item_name", f"세션 {len(st.session_state.session_history) + 1}")
            st.session_state.session_history.append({
                "name": first_item,
                "chat_history": st.session_state.chat_history.copy(),
            })
        # 새 세션 시작
        st.session_state.chat_history = []
        st.session_state.session_id = None
        st.session_state.pending_candidates = None
        st.session_state.current_item_name = None
        st.rerun()
    
    st.divider()
    
    st.header("📊 채팅 로그")
    
    # 이전 세션 히스토리
    if st.session_state.get("session_history"):
        st.caption("**이전 세션**")
        for idx, session in enumerate(st.session_state.session_history):
            if st.button(f"📁 {session['name']}", key=f"session_{idx}", use_container_width=True):
                st.session_state.chat_history = session["chat_history"]
                st.session_state.session_id = None
                st.session_state.pending_candidates = None
                st.rerun()
        st.divider()
    
    # 현재 세션 채팅 로그
    st.caption("**현재 세션**")
    if st.session_state.get("chat_history"):
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                txt = msg["content"][:25] + "..." if len(msg["content"]) > 25 else msg["content"]
                st.markdown(f'<div class="chat-log-item"><span class="chat-log-user">You</span> {txt}</div>', unsafe_allow_html=True)
            else:
                item = msg.get("item_name", "-")
                hs = msg.get("hs_code", "-")
                st.markdown(f'<div class="chat-log-item"><span class="chat-log-asst">AI</span> {item} · {hs}</div>', unsafe_allow_html=True)
    else:
        st.caption("대화가 없습니다.")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "pending_candidates" not in st.session_state:
    st.session_state.pending_candidates = None
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "current_item_name" not in st.session_state:
    st.session_state.current_item_name = None


# ===== API 호출 함수 =====

def call_analyze_api(message: str, session_id: str = None) -> dict:
    """1단계: 입력 분석 API 호출"""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{API_BASE_URL}/api/v1/analyze",
            json={"message": message, "session_id": session_id}
        )
        return response.json()


def call_calculate_api(session_id: str, selected_hs_code: str) -> dict:
    """2단계: 비용 계산 API 호출"""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{API_BASE_URL}/api/v1/calculate",
            json={"session_id": session_id, "selected_hs_code": selected_hs_code}
        )
        return response.json()


def download_report(filename: str) -> bytes:
    """보고서 다운로드"""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{API_BASE_URL}/api/v1/reports/{filename}")
        return response.content


# ===== HS 코드 선택 처리 (버튼 클릭 후) =====

if st.session_state.get("selected_hs_code"):
    selected_hs = st.session_state.pop("selected_hs_code")
    session_id = st.session_state.session_id
    
    with st.chat_message("assistant"):
        st.markdown(f"**✅ 선택된 HS 코드:** `{selected_hs}`")
        st.markdown("관세율을 조회하고 비용을 계산합니다...")
        
        with st.spinner("계산 중..."):
            try:
                result = call_calculate_api(session_id, selected_hs)
                
                if result.get("success"):
                    # 결과 표시
                    st.markdown(result.get("report_content", ""))
                    
                    # 보고서 다운로드
                    report_paths = result.get("report_paths", {})
                    if report_paths:
                        st.divider()
                        st.subheader("📥 보고서 다운로드")
                        cols = st.columns(3)
                        
                        for idx, (fmt, path) in enumerate(report_paths.items()):
                            if path:
                                filename = os.path.basename(path)
                                with cols[idx]:
                                    try:
                                        data = download_report(filename)
                                        label = {"pdf": "📄 PDF", "word": "📝 Word", "excel": "📊 Excel"}.get(fmt, fmt)
                                        mime = {
                                            "pdf": "application/pdf",
                                            "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        }.get(fmt, "application/octet-stream")
                                        st.download_button(label, data, filename, mime, key=f"dl_{fmt}", use_container_width=True)
                                    except Exception as e:
                                        st.error(f"{fmt} 다운로드 실패")
                    
                    # 채팅 히스토리에 추가
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get("report_content", ""),
                        "item_name": result.get("item_name"),
                        "hs_code": result.get("hs_code"),
                        "total_cost": result.get("total_cost"),
                        "report_paths": report_paths,
                    })
                else:
                    st.error(f"오류: {result.get('error', '알 수 없는 오류')}")
                
            except Exception as e:
                st.error(f"API 호출 실패: {str(e)}")
    
    st.session_state.pending_candidates = None
    st.stop()


# ===== HS 코드 선택 UI (Human-in-the-Loop) =====

if st.session_state.pending_candidates:
    pending = st.session_state.pending_candidates
    candidates = pending.get("candidates", [])
    
    if not candidates:
        st.warning("HS 코드 후보를 찾지 못했습니다. 다시 시도해주세요.")
        st.session_state.pending_candidates = None
        st.stop()
    
    st.markdown("---")
    st.subheader("🔍 HS 코드 선택")
    st.markdown("**AI가 찾은 HS 코드 후보입니다. 가장 적합한 코드를 선택해주세요:**")
    st.caption("선택 후 관세율이 자동으로 조회됩니다.")
    
    cols = st.columns(3)
    
    for idx, cand in enumerate(candidates[:3]):
        with cols[idx]:
            hs_code = cand.get("hs_code", "미확인")
            품명 = cand.get("품명", "") or ""
            적합도 = cand.get("적합도", "") or ""
            rag_context = cand.get("rag_context", "") or ""
            
            품명_display = (품명[:30] + "...") if len(품명) > 30 else 품명
            적합도_display = (적합도[:80] + "...") if len(적합도) > 80 else 적합도
            
            st.markdown(f"""
<div style="
    border: 2px solid #1f77b4; 
    border-radius: 12px; 
    padding: 16px; 
    text-align: center; 
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
">
    <div>
        <div style="font-size: 0.85em; color: #666; margin-bottom: 4px;">후보 {idx + 1}</div>
        <div style="font-size: 1.3em; font-weight: bold; color: #1f77b4; margin-bottom: 10px; font-family: monospace;">
            📦 {hs_code}
        </div>
        <div style="font-size: 0.95em; color: #333; margin-bottom: 8px; font-weight: 500;">
            {품명_display}
        </div>
    </div>
    <div style="
        font-size: 0.9em; 
        color: #555; 
        line-height: 1.5; 
        text-align: left;
        background: #fff;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ddd;
    ">
        <div style="font-weight: 600; color: #1f77b4; margin-bottom: 4px;">💡 분류 근거</div>
        {적합도_display if 적합도_display else "AI 분석 결과"}
    </div>
</div>
            """, unsafe_allow_html=True)
            
            if rag_context:
                with st.expander("📄 관세청 DB 검색 결과", expanded=False):
                    st.markdown(
                        f"<div style='font-size: 0.9em; line-height: 1.6; color: #444;'>{rag_context}</div>",
                        unsafe_allow_html=True
                    )
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            
            if st.button(f"✅ 이 코드 선택", key=f"hs_select_{idx}", use_container_width=True, type="primary"):
                st.session_state.selected_hs_code = hs_code
                st.rerun()
    
    for idx in range(len(candidates), 3):
        with cols[idx]:
            st.markdown("<div style='min-height: 220px;'></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.stop()


# ===== 이전 대화 기록 표시 =====

for msg in st.session_state.chat_history:
    role = msg["role"]
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg["content"])
        else:
            st.markdown(msg.get("content", ""))
            
            report_paths = msg.get("report_paths", {})
            if report_paths:
                st.divider()
                st.subheader("📥 보고서 다운로드")
                cols = st.columns(3)
                for idx, (fmt, path) in enumerate(report_paths.items()):
                    if path:
                        filename = os.path.basename(path)
                        with cols[idx]:
                            try:
                                data = download_report(filename)
                                label = {"pdf": "📄 PDF", "word": "📝 Word", "excel": "📊 Excel"}.get(fmt, fmt)
                                mime = {
                                    "pdf": "application/pdf",
                                    "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                }.get(fmt, "application/octet-stream")
                                st.download_button(label, data, filename, mime, key=f"hist_{fmt}_{id(msg)}", use_container_width=True)
                            except:
                                pass


# ===== 사용자 입력 처리 =====

if prompt := st.chat_input("수입하려는 물품의 정보를 입력해주세요 (물품명, 수량, 단가, 통화)"):
    # 사용자 메시지 표시
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # API 호출
    with st.chat_message("assistant"):
        with st.spinner("📥 입력 정보를 분석하고 있습니다..."):
            try:
                result = call_analyze_api(prompt, st.session_state.session_id)
                
                if result.get("success"):
                    st.session_state.session_id = result.get("session_id")
                    st.session_state.current_item_name = result.get("item_name")
                    
                    phase = result.get("phase")
                    
                    if phase == "hs_code_selection":
                        # HS 코드 후보 표시
                        candidates = result.get("hs_code_candidates", [])
                        exchange_rate = result.get("exchange_rate")
                        item_name = result.get("item_name")
                        currency = result.get("currency")
                        
                        st.markdown(f"**'{item_name}'**의 HS 코드 후보를 찾았습니다.")
                        if exchange_rate:
                            st.markdown(f"💱 현재 환율: **{exchange_rate:,.2f} KRW/{currency}**")
                        
                        st.session_state.pending_candidates = {
                            "candidates": [
                                {
                                    "hs_code": c.get("hs_code"),
                                    "품명": c.get("품명"),
                                    "적합도": c.get("적합도"),
                                    "rag_context": c.get("rag_context"),
                                }
                                for c in candidates
                            ]
                        }
                        st.rerun()
                    
                    elif phase == "need_more_info":
                        # 추가 정보 필요
                        st.warning(result.get("message", "추가 정보가 필요합니다."))
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result.get("message", ""),
                        })
                    
                    else:
                        st.info(result.get("message", "처리 중..."))
                
                else:
                    st.error(f"오류: {result.get('error', result.get('message', '알 수 없는 오류'))}")
                    
            except httpx.ConnectError:
                st.error(f"❌ API 서버에 연결할 수 없습니다.\n\n서버 주소: `{API_BASE_URL}`\n\n**해결 방법:**\n1. API 서버가 실행 중인지 확인하세요.\n2. `python -m api.server`로 API 서버를 시작하세요.")
            except Exception as e:
                st.error(f"API 호출 실패: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# 푸터
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    TradeSimple - 수입업무 간편화 AI 도우미 | API-First Architecture
</div>
""", unsafe_allow_html=True)
