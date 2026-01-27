# api/server.py
"""
TradeSimple API Server - LangServe 기반
FastAPI + LangServe로 LangGraph 에이전트를 REST API로 제공
"""
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langserve import add_routes
from langchain_core.messages import HumanMessage, AIMessage

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph import create_graph, get_initial_state
from app.state import AgentState
from api.schemas import (
    ImportCostRequest,
    ImportCostResponse,
    HealthResponse,
)

# 세션 저장소 (프로덕션에서는 Redis 등 사용 권장)
sessions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    print("🚀 TradeSimple API 서버 시작...")
    # 그래프 미리 로드
    _ = create_graph()
    print("✅ LangGraph 에이전트 로드 완료")
    yield
    print("👋 TradeSimple API 서버 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="TradeSimple API",
    description="수입업무 간편화 AI 도우미 - HS코드 분류 및 관세 계산 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== LangServe 라우트 추가 =====
# LangGraph를 /agent 엔드포인트로 노출
graph = create_graph()
add_routes(
    app,
    graph,
    path="/agent",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)


# ===== 커스텀 API 엔드포인트 =====

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse()


@app.post("/api/v1/calculate", response_model=ImportCostResponse, tags=["Import Cost"])
async def calculate_import_cost(request: ImportCostRequest):
    """
    수입 비용 계산 API
    
    사용자 메시지를 입력받아 HS 코드 분류, 관세 계산, 보고서 생성을 수행합니다.
    세션 ID를 통해 대화를 이어갈 수 있습니다.
    """
    try:
        # 세션 관리
        session_id = request.session_id or str(uuid.uuid4())
        
        # 기존 세션 상태 가져오기 또는 새로 생성
        if session_id in sessions:
            current_state = sessions[session_id]["state"]
        else:
            current_state = get_initial_state()
        
        # 사용자 메시지 추가
        current_state["messages"] = current_state.get("messages", []) + [
            HumanMessage(content=request.message)
        ]
        
        # 그래프 실행
        graph = create_graph()
        result_state = await graph.ainvoke(current_state)
        
        # 세션 저장
        sessions[session_id] = {
            "state": result_state,
            "updated_at": datetime.now().isoformat(),
        }
        
        # 응답 메시지 추출
        assistant_message = None
        for msg in reversed(result_state.get("messages", [])):
            if isinstance(msg, AIMessage):
                assistant_message = msg.content
                break
        
        return ImportCostResponse(
            success=True,
            session_id=session_id,
            current_phase=result_state.get("current_phase"),
            missing_info=result_state.get("missing_info"),
            item_name=result_state.get("item_name"),
            quantity=result_state.get("quantity"),
            unit_price=result_state.get("unit_price"),
            currency=result_state.get("currency"),
            hs_code=result_state.get("hs_code"),
            hs_code_rationale=result_state.get("hs_code_rationale"),
            tariff_rate=result_state.get("tariff_rate"),
            exchange_rate=result_state.get("exchange_rate"),
            tax_amount=result_state.get("tax_amount"),
            total_cost=result_state.get("total_cost"),
            report_content=result_state.get("report_content"),
            report_paths=result_state.get("report_paths"),
            assistant_message=assistant_message,
            error=result_state.get("error"),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/session/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    """세션 상태 조회"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session["state"]
    
    return {
        "session_id": session_id,
        "updated_at": session["updated_at"],
        "current_phase": state.get("current_phase"),
        "item_name": state.get("item_name"),
        "quantity": state.get("quantity"),
        "hs_code": state.get("hs_code"),
        "total_cost": state.get("total_cost"),
    }


@app.delete("/api/v1/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """세션 삭제"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": "Session deleted", "session_id": session_id}


@app.get("/api/v1/reports/{filename}", tags=["Reports"])
async def download_report(filename: str):
    """생성된 보고서 다운로드"""
    # 보안: 파일명 검증
    allowed_extensions = [".pdf", ".docx", ".xlsx"]
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(os.getcwd(), filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


# ===== 메인 실행 =====
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
