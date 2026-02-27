# app/main.py
"""
TradeSimple - 수입업무 간편화 AI 도우미
Streamlit 웹 인터페이스

개선 사항:
- SSE 스트리밍으로 실시간 진행 상황 표시
- 사이드바 ChatGPT 스타일 채팅 로그
- 새 물품 입력 시 세션 자동 초기화
"""
import json
import os
import httpx
import streamlit as st
from datetime import datetime

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeSimple",
    page_icon="🚢",
    layout="wide",
)

# ── 커스텀 CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 사이드바 폰트 크기 */
[data-testid="stSidebar"] { font-size: 0.78rem !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-size: 0.82rem !important; margin: 0 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li { font-size: 0.78rem !important; line-height: 1.4; }
[data-testid="stSidebar"] .stButton button {
    font-size: 0.76rem !important;
    padding: 0.25rem 0.6rem !important;
    border-radius: 6px;
}
/* 세션 히스토리 아이템 */
.session-card {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    padding: 5px 8px;
    border-radius: 6px;
    margin: 2px 0;
    cursor: pointer;
    background: transparent;
    transition: background 0.15s;
}
.session-card:hover { background: #f0f2f6; }
.session-icon { font-size: 0.8rem; margin-top: 1px; flex-shrink: 0; }
.session-meta { display: flex; flex-direction: column; min-width: 0; }
.session-name {
    font-size: 0.76rem;
    font-weight: 600;
    color: #1f2937;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 160px;
}
.session-time { font-size: 0.68rem; color: #9ca3af; margin-top: 1px; }
/* 스트리밍 상태 메시지 */
.stream-status {
    font-size: 0.85rem;
    color: #374151;
    padding: 8px 12px;
    background: #f9fafb;
    border-left: 3px solid #3b82f6;
    border-radius: 4px;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:2rem;font-weight:700;margin-bottom:0.2rem;">🚢 TradeSimple</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:0.95rem;color:#6b7280;margin-bottom:1.2rem;">'
    'HS코드 분류 · 관세 계산 · 보고서 자동 생성 AI</div>',
    unsafe_allow_html=True,
)

# ── 세션 상태 초기화 ──────────────────────────────────────────────────────────
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


# ── API 호출 함수 ─────────────────────────────────────────────────────────────

def call_analyze_api_stream(message: str, session_id: str = None):
    """SSE 스트리밍으로 analyze 호출. 진행 dict를 순차적으로 yield."""
    try:
        with httpx.Client(timeout=180.0) as client:
            with client.stream(
                "POST",
                f"{API_BASE_URL}/api/v1/analyze/stream",
                json={"message": message, "session_id": session_id},
            ) as response:
                for line in response.iter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        try:
                            yield json.loads(line[6:])
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        yield {"done": True, "success": False, "error": str(e), "session_id": session_id or ""}


def call_calculate_api_stream(session_id: str, selected_hs_code: str):
    """SSE 스트리밍으로 calculate 호출."""
    try:
        with httpx.Client(timeout=180.0) as client:
            with client.stream(
                "POST",
                f"{API_BASE_URL}/api/v1/calculate/stream",
                json={"session_id": session_id, "selected_hs_code": selected_hs_code},
            ) as response:
                for line in response.iter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        try:
                            yield json.loads(line[6:])
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        yield {"done": True, "success": False, "error": str(e), "session_id": session_id}


def download_report(filename: str) -> bytes:
    """보고서 파일 다운로드."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{API_BASE_URL}/api/v1/reports/{filename}")
        return response.content


def _render_report_downloads(report_paths: dict, key_suffix: str = ""):
    """보고서 다운로드 버튼 3개 렌더링."""
    if not report_paths:
        return
    st.divider()
    st.subheader("📥 보고서 다운로드")
    cols = st.columns(3)
    label_map = {"pdf": "📄 PDF", "word": "📝 Word", "excel": "📊 Excel"}
    mime_map = {
        "pdf": "application/pdf",
        "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    for idx, (fmt, path) in enumerate(report_paths.items()):
        if not path:
            continue
        filename = os.path.basename(path)
        with cols[idx]:
            try:
                data = download_report(filename)
                st.download_button(
                    label_map.get(fmt, fmt),
                    data,
                    filename,
                    mime_map.get(fmt, "application/octet-stream"),
                    key=f"dl_{fmt}_{key_suffix}",
                    use_container_width=True,
                )
            except Exception:
                st.error(f"{fmt} 다운로드 실패")


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**🚢 TradeSimple**")
    st.caption(f"API: `{API_BASE_URL}`")

    # 새 분석 버튼
    if st.button("＋ 새로운 분석", use_container_width=True):
        if st.session_state.chat_history:
            item = st.session_state.current_item_name or f"세션 {len(st.session_state.session_history) + 1}"
            st.session_state.session_history.append({
                "name": item,
                "chat_history": st.session_state.chat_history.copy(),
                "timestamp": datetime.now().strftime("%H:%M"),
                "date": datetime.now().strftime("%Y-%m-%d"),
            })
        st.session_state.chat_history = []
        st.session_state.session_id = None
        st.session_state.pending_candidates = None
        st.session_state.current_item_name = None
        st.rerun()

    st.divider()

    # ── 이전 세션 히스토리 (ChatGPT 스타일) ─────────────────────────────────
    if st.session_state.session_history:
        st.caption("이전 검색")
        # 최신순 정렬
        for idx, session in enumerate(reversed(st.session_state.session_history)):
            real_idx = len(st.session_state.session_history) - 1 - idx
            name_short = session["name"][:22] + "…" if len(session["name"]) > 22 else session["name"]
            ts = session.get("timestamp", "")

            col_icon, col_info = st.columns([1, 6])
            with col_icon:
                st.markdown("🔍")
            with col_info:
                if st.button(
                    f"{name_short}",
                    key=f"hist_{real_idx}",
                    use_container_width=True,
                    help=f"클릭하여 이전 대화 불러오기 • {ts}",
                ):
                    st.session_state.chat_history = session["chat_history"].copy()
                    st.session_state.session_id = None
                    st.session_state.pending_candidates = None
                    st.rerun()
        st.divider()

    # ── 현재 세션 미니 로그 ──────────────────────────────────────────────────
    st.caption("현재 세션")
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history[-6:]:   # 최근 6개만
            if msg["role"] == "user":
                txt = msg["content"][:28] + "…" if len(msg["content"]) > 28 else msg["content"]
                st.markdown(
                    f'<div style="font-size:0.74rem;color:#1d4ed8;padding:1px 0;">▶ {txt}</div>',
                    unsafe_allow_html=True,
                )
            else:
                item = msg.get("item_name") or ""
                hs = msg.get("hs_code") or ""
                if item:
                    label = f"{item}" + (f" · {hs}" if hs else "")
                    st.markdown(
                        f'<div style="font-size:0.74rem;color:#059669;padding:1px 0;">✓ {label}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.caption("대화가 없습니다.")

    st.divider()

    # ── 사용 가이드 (접힘) ────────────────────────────────────────────────────
    with st.expander("📋 입력 예시 보기"):
        st.markdown("""
- 미국에서 스마트워치 100개를 개당 300달러에 수입
- 냉동 참치 500kg, 일본, kg당 50달러
- 중국에서 노트북 50대 개당 800달러
        """)


# ── HS 코드 선택 처리 (버튼 클릭 직후) ──────────────────────────────────────

if st.session_state.get("selected_hs_code"):
    selected_hs = st.session_state.pop("selected_hs_code")
    session_id = st.session_state.session_id

    with st.chat_message("assistant"):
        st.markdown(f"**✅ 선택된 HS 코드:** `{selected_hs}`")
        status_ph = st.empty()

        final_data = None
        for event in call_calculate_api_stream(session_id, selected_hs):
            if not event.get("done"):
                status_ph.markdown(
                    f'<div class="stream-status">{event["message"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                final_data = event

        status_ph.empty()

        if final_data and final_data.get("success"):
            st.markdown(final_data.get("report_content", ""))
            _render_report_downloads(final_data.get("report_paths", {}), key_suffix=f"sel_{selected_hs}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_data.get("report_content", ""),
                "item_name": final_data.get("item_name"),
                "hs_code": final_data.get("hs_code"),
                "total_cost": final_data.get("total_cost"),
                "report_paths": final_data.get("report_paths", {}),
            })
        else:
            err = (final_data or {}).get("error", "알 수 없는 오류")
            st.error(f"오류: {err}")

    st.session_state.pending_candidates = None
    st.stop()


# ── HS 코드 선택 UI (Human-in-the-Loop) ──────────────────────────────────────

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

            품명_display = (품명[:30] + "…") if len(품명) > 30 else 품명
            적합도_display = (적합도[:80] + "…") if len(적합도) > 80 else 적합도

            st.markdown(f"""
<div style="
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    min-height: 220px;
    display: flex; flex-direction: column; justify-content: space-between;
">
    <div>
        <div style="font-size:0.78rem;color:#6b7280;margin-bottom:4px;">후보 {idx + 1}</div>
        <div style="font-size:1.2rem;font-weight:700;color:#1d4ed8;margin-bottom:8px;font-family:monospace;">
            📦 {hs_code}
        </div>
        <div style="font-size:0.88rem;color:#374151;margin-bottom:8px;font-weight:500;">
            {품명_display}
        </div>
    </div>
    <div style="
        font-size:0.82rem;color:#4b5563;line-height:1.5;text-align:left;
        background:#fff;padding:10px;border-radius:8px;border:1px solid #e5e7eb;
    ">
        <div style="font-weight:600;color:#1d4ed8;margin-bottom:4px;">💡 분류 근거</div>
        {적합도_display if 적합도_display else "AI 분석 결과"}
    </div>
</div>
            """, unsafe_allow_html=True)

            if rag_context:
                with st.expander("📄 관세청 DB 근거", expanded=False):
                    st.markdown(
                        f"<div style='font-size:0.82rem;line-height:1.6;color:#374151;'>{rag_context}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            if st.button(f"✅ 이 코드 선택", key=f"hs_select_{idx}", use_container_width=True, type="primary"):
                st.session_state.selected_hs_code = hs_code
                st.rerun()

    for idx in range(len(candidates), 3):
        with cols[idx]:
            st.markdown("<div style='min-height:220px;'></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.stop()


# ── 이전 대화 기록 표시 ───────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    role = msg["role"]
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg["content"])
        else:
            st.markdown(msg.get("content", ""))
            if msg.get("report_paths"):
                _render_report_downloads(msg["report_paths"], key_suffix=f"hist_{id(msg)}")


# ── 사용자 입력 처리 ──────────────────────────────────────────────────────────

if prompt := st.chat_input("수입하려는 물품의 정보를 입력해주세요 (물품명, 수량, 단가, 통화)"):
    # ── 물품 변경 감지: 항상 새 세션으로 시작 ────────────────────────────────
    # 이전 세션 ID를 초기화하여 이전 결과가 섞이는 것을 방지
    st.session_state.session_id = None
    st.session_state.pending_candidates = None

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_ph = st.empty()
        final_data = None

        # ── 스트리밍으로 analyze 호출 ────────────────────────────────────────
        try:
            for event in call_analyze_api_stream(prompt, session_id=None):
                if not event.get("done"):
                    status_ph.markdown(
                        f'<div class="stream-status">{event["message"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    final_data = event

        except httpx.ConnectError:
            st.error(
                f"❌ API 서버에 연결할 수 없습니다.\n\n"
                f"서버: `{API_BASE_URL}`\n\n"
                "**해결 방법:** `python -m api.server` 로 API 서버를 시작하세요."
            )
            st.stop()

        status_ph.empty()

        if not final_data:
            st.error("API로부터 응답이 없습니다.")
            st.stop()

        # ── 응답 처리 ────────────────────────────────────────────────────────
        if final_data.get("success") and final_data.get("phase") == "hs_code_selection":
            item_name = final_data.get("item_name", "물품")
            exchange_rate = final_data.get("exchange_rate")
            currency = final_data.get("currency", "USD")

            st.session_state.session_id = final_data["session_id"]
            st.session_state.current_item_name = item_name

            st.markdown(f"**'{item_name}'** 의 HS 코드 후보를 찾았습니다.")
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
                    for c in (final_data.get("hs_code_candidates") or [])
                ]
            }
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**'{item_name}'** 의 HS 코드 후보를 찾았습니다. HS 코드를 선택해주세요.",
            })
            st.rerun()

        elif final_data.get("phase") == "need_more_info":
            msg = final_data.get("message", "추가 정보가 필요합니다.")
            st.warning(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})

        elif not final_data.get("success"):
            err = final_data.get("error", "알 수 없는 오류가 발생했습니다.")
            st.error(f"오류: {err}")

        else:
            st.info(final_data.get("message", "처리 중..."))


# ── 푸터 ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#9ca3af;font-size:0.8rem;">'
    'TradeSimple · AI 수입 비용 분석 | API-First Architecture'
    '</div>',
    unsafe_allow_html=True,
)
