# api/server.py
"""
TradeSimple API Server - HITL(Human-in-the-Loop) 지원
FastAPI 기반 REST API 서버

API 흐름:
1. POST /api/v1/analyze  → 입력 분석 + HS 코드 후보 3개 반환 (HITL)
2. POST /api/v1/calculate → 사용자가 선택한 HS 코드로 비용 계산 + 보고서 생성
"""
import os
import sys
import re
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import json as _json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.state import get_initial_state
from app.graph import get_graph, continue_after_hs_selection, _status_message
from api.schemas import (
    AnalyzeRequest, AnalyzeResponse, HSCodeCandidate,
    CalculateRequest, CalculateResponse,
    HealthResponse, SessionResponse,
)

# 세션 저장소 (프로덕션에서는 Redis 등 사용)
sessions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    print("🚀 TradeSimple API 서버 시작 (HITL 지원)...")
    yield
    print("👋 TradeSimple API 서버 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="TradeSimple API",
    description="""
수입업무 간편화 AI 도우미 - HS코드 분류 및 관세 계산 API

## HITL (Human-in-the-Loop) 흐름

1. **POST /api/v1/analyze** - 입력 분석 및 HS 코드 후보 검색
   - 사용자 메시지에서 물품명, 수량, 단가, 통화 추출
   - HS 코드 후보 3개 + 환율 병렬 조회
   - 응답: HS 코드 후보 목록 (사용자 선택 대기)

2. **POST /api/v1/calculate** - 비용 계산 및 보고서 생성
   - 사용자가 선택한 HS 코드로 관세율 조회
   - 총 비용 계산 (관세 + 부가세)
   - PDF/Word/Excel 보고서 병렬 생성
""",
    version="2.0.0",
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


# ===== API 엔드포인트 =====

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """헬스체크"""
    return HealthResponse()


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["HITL Flow"])
async def analyze_input(request: AnalyzeRequest):
    """
    1단계: 입력 분석 + HS 코드 후보 검색 (HITL)
    
    - 사용자 메시지에서 물품명, 수량, 단가, 통화 추출
    - HS 코드 후보 3개 검색 (병렬)
    - 환율 조회 (병렬)
    - 응답: 후보 목록 → 사용자가 선택
    """
    import asyncio
    
    try:
        # 세션 생성 또는 재사용
        session_id = request.session_id or str(uuid.uuid4())
        
        # 1. LangGraph 실행 (input_validator → supervisor → parallel_fetch)
        print(f"[API] analyze - LangGraph 실행 시작: {request.message[:50]}...")
        graph = get_graph()
        state = get_initial_state()
        state["messages"] = state.get("messages", []) + [HumanMessage(content=request.message)]

        lang_config = {"configurable": {"thread_id": session_id}}
        result_state = await graph.ainvoke(state, lang_config)
        print(f"[API] analyze - current_phase={result_state.get('current_phase')}")

        # 2. 세션 저장 (LangGraph 상태 그대로)
        sessions[session_id] = {
            "state": result_state,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # 3. 응답 구성
        phase = result_state.get("current_phase")
        candidates_raw = result_state.get("hs_code_candidates", []) or []
        candidates = [
            HSCodeCandidate(
                hs_code=c.get("hs_code", ""),
                품명=c.get("품명", ""),
                적합도=c.get("적합도", ""),
                rag_context=c.get("rag_context", ""),
            )
            for c in candidates_raw[:3]
        ]

        # request_info 메시지 추출
        last_msg = None
        for msg in reversed(result_state.get("messages", [])):
            if isinstance(msg, AIMessage):
                last_msg = msg.content
                break

        return AnalyzeResponse(
            success=True if phase != "request_info" else False,
            session_id=session_id,
            phase="hs_code_selection" if phase == "hs_code_selection" else ("need_more_info" if phase == "request_info" else "analyzing"),
            item_name=result_state.get("item_name"),
            quantity=result_state.get("quantity"),
            quantity_unit=result_state.get("quantity_unit"),
            unit_price=result_state.get("unit_price"),
            price_unit=result_state.get("price_unit"),
            total_foreign_price=result_state.get("total_foreign_price"),
            currency=result_state.get("currency"),
            hs_code_candidates=candidates if phase == "hs_code_selection" else None,
            exchange_rate=result_state.get("exchange_rate"),
            missing_info=result_state.get("missing_info"),
            message=last_msg or "HS 코드 후보를 찾았습니다. 가장 적합한 코드를 선택해주세요."
        )
        
    except Exception as e:
        print(f"[API] analyze - 오류: {e}")
        import traceback
        traceback.print_exc()
        return AnalyzeResponse(
            success=False,
            session_id=request.session_id or str(uuid.uuid4()),
            phase="error",
            error=str(e)
        )


@app.post("/api/v1/calculate", response_model=CalculateResponse, tags=["HITL Flow"])
async def calculate_cost(request: CalculateRequest):
    """
    2단계: 비용 계산 + 보고서 생성
    
    - 선택된 HS 코드로 관세율 조회
    - 총 비용 계산 (관세 + 부가세)
    - PDF/Word/Excel 보고서 병렬 생성
    """
    import asyncio
    
    try:
        # 세션 확인
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다. 먼저 /api/v1/analyze를 호출하세요.")
        
        session = sessions[request.session_id]
        state = session["state"]

        # 입력값 덮어쓰기 (옵션)
        overrides = {
            "item_name": request.item_name,
            "quantity": request.quantity,
            "quantity_unit": request.quantity_unit,
            "unit_price": request.unit_price,
            "price_unit": request.price_unit,
            "total_foreign_price": request.total_foreign_price,
            "currency": request.currency,
        }
        for k, v in overrides.items():
            if v is not None:
                state[k] = v

        # total_foreign_price가 없으면 계산
        if state.get("total_foreign_price") is None:
            if state.get("quantity") and state.get("unit_price"):
                state["total_foreign_price"] = state["quantity"] * state["unit_price"]
        
        print(f"[API] calculate - 세션 {request.session_id}, HS 코드 {request.selected_hs_code}")
        
        # LangGraph 계속 실행 (HS 선택 이후 단계)
        async def run_continue():
            last_state = None
            async for chunk in continue_after_hs_selection(
                selected_hs_code=request.selected_hs_code,
                selected_tariff_rate=0.0,
                selected_rationale="사용자 선택",
                current_state=state,
                report_id=int(datetime.now().timestamp()),
                thread_id=request.session_id,
            ):
                last_state = chunk["state"]
            return last_state

        result_state = await run_continue()
        session["state"] = result_state
        session["updated_at"] = datetime.now().isoformat()

        return CalculateResponse(
            success=True,
            session_id=request.session_id,
            item_name=result_state.get("item_name"),
            quantity=result_state.get("quantity"),
            quantity_unit=result_state.get("quantity_unit"),
            unit_price=result_state.get("unit_price"),
            price_unit=result_state.get("price_unit"),
            total_foreign_price=result_state.get("total_foreign_price"),
            currency=result_state.get("currency"),
            hs_code=result_state.get("hs_code"),
            hs_code_rationale=result_state.get("hs_code_rationale"),
            tariff_rate=result_state.get("tariff_rate"),
            exchange_rate=result_state.get("exchange_rate"),
            total_krw=(result_state.get("total_foreign_price") or 0) * (result_state.get("exchange_rate") or 0),
            tax_amount=result_state.get("tax_amount"),
            vat_amount=result_state.get("vat_amount"),
            total_cost=result_state.get("total_cost"),
            report_content=result_state.get("report_content"),
            report_paths=result_state.get("report_paths"),
            message=f"비용 계산이 완료되었습니다. 총 예상 비용: {result_state.get('total_cost', 0):,.0f}원"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] calculate - 오류: {e}")
        import traceback
        traceback.print_exc()
        return CalculateResponse(
            success=False,
            session_id=request.session_id,
            error=str(e)
        )


@app.post("/api/v1/analyze/stream", tags=["HITL Flow"])
async def analyze_input_stream(request: AnalyzeRequest):
    """
    1단계 스트리밍: 분석 진행 상황을 SSE로 실시간 전송

    - 각 LangGraph 노드 실행 시 진행 메시지를 즉시 전송
    - 최종 이벤트(done=True)에 HS 코드 후보 및 세션 정보 포함
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        from app.graph import get_graph
        graph = get_graph()
        final_state = None
        current = get_initial_state()
        current["messages"] = [HumanMessage(content=request.message)]

        try:
            lang_config = {"configurable": {"thread_id": session_id}}
            async for event in graph.astream(current, lang_config, stream_mode="updates"):
                for node_name, update in event.items():
                    current = {**current, **update}
                    msg = _status_message(node_name, current)
                    payload = _json.dumps({"message": msg, "done": False}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            final_state = current

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {_json.dumps({'done': True, 'success': False, 'error': str(e), 'session_id': session_id}, ensure_ascii=False)}\n\n"
            return

        # 세션 저장
        if final_state:
            sessions[session_id] = {
                "state": final_state,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

        # 최종 결과 이벤트
        phase = (final_state or {}).get("current_phase", "")
        candidates_raw = (final_state or {}).get("hs_code_candidates", []) or []
        candidates = [
            {
                "hs_code": c.get("hs_code", ""),
                "품명": c.get("품명", ""),
                "적합도": c.get("적합도", ""),
                "rag_context": c.get("rag_context", ""),
            }
            for c in candidates_raw[:3]
        ]

        last_msg = None
        for msg in reversed((final_state or {}).get("messages", [])):
            if isinstance(msg, AIMessage):
                last_msg = msg.content
                break

        final_payload = {
            "done": True,
            "success": phase != "request_info",
            "session_id": session_id,
            "phase": (
                "hs_code_selection" if phase == "hs_code_selection"
                else "need_more_info" if phase in ("request_info", "waiting_input")
                else "analyzing"
            ),
            "item_name": (final_state or {}).get("item_name"),
            "quantity": (final_state or {}).get("quantity"),
            "quantity_unit": (final_state or {}).get("quantity_unit"),
            "unit_price": (final_state or {}).get("unit_price"),
            "price_unit": (final_state or {}).get("price_unit"),
            "total_foreign_price": (final_state or {}).get("total_foreign_price"),
            "currency": (final_state or {}).get("currency"),
            "exchange_rate": (final_state or {}).get("exchange_rate"),
            "hs_code_candidates": candidates if phase == "hs_code_selection" else None,
            "missing_info": (final_state or {}).get("missing_info"),
            "message": last_msg or (
                "HS 코드 후보를 찾았습니다. 가장 적합한 코드를 선택해주세요."
                if phase == "hs_code_selection"
                else "추가 정보가 필요합니다."
            ),
        }
        yield f"data: {_json.dumps(final_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/v1/calculate/stream", tags=["HITL Flow"])
async def calculate_cost_stream(request: CalculateRequest):
    """
    2단계 스트리밍: 비용 계산 진행 상황을 SSE로 실시간 전송

    - 관세율 조회, 비용 계산, 보고서 생성 단계별 메시지 스트리밍
    - 최종 이벤트(done=True)에 전체 계산 결과 및 보고서 경로 포함
    """
    async def event_generator():
        try:
            if request.session_id not in sessions:
                yield f"data: {_json.dumps({'done': True, 'success': False, 'error': '세션을 찾을 수 없습니다. 먼저 /api/v1/analyze를 호출하세요.', 'session_id': request.session_id})}\n\n"
                return

            session = sessions[request.session_id]
            state = session["state"].copy()

            # 입력값 덮어쓰기
            overrides = {
                "item_name": request.item_name,
                "quantity": request.quantity,
                "quantity_unit": request.quantity_unit,
                "unit_price": request.unit_price,
                "price_unit": request.price_unit,
                "total_foreign_price": request.total_foreign_price,
                "currency": request.currency,
            }
            for k, v in overrides.items():
                if v is not None:
                    state[k] = v

            if state.get("total_foreign_price") is None:
                if state.get("quantity") and state.get("unit_price"):
                    state["total_foreign_price"] = state["quantity"] * state["unit_price"]

            final_state = None
            report_id = int(datetime.now().timestamp())

            async for event_data in continue_after_hs_selection(
                selected_hs_code=request.selected_hs_code,
                selected_tariff_rate=0.0,
                selected_rationale="사용자 선택",
                current_state=state,
                report_id=report_id,
                thread_id=request.session_id,
            ):
                final_state = event_data["state"]
                msg = event_data["message"]
                if not msg.startswith("✅"):
                    payload = _json.dumps({"message": msg, "done": False}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            # 세션 업데이트
            if final_state:
                session["state"] = final_state
                session["updated_at"] = datetime.now().isoformat()

            report_paths = (final_state or {}).get("report_paths") or {}
            total_cost = (final_state or {}).get("total_cost") or 0

            final_payload = {
                "done": True,
                "success": True,
                "session_id": request.session_id,
                "item_name": (final_state or {}).get("item_name"),
                "quantity": (final_state or {}).get("quantity"),
                "quantity_unit": (final_state or {}).get("quantity_unit"),
                "unit_price": (final_state or {}).get("unit_price"),
                "price_unit": (final_state or {}).get("price_unit"),
                "total_foreign_price": (final_state or {}).get("total_foreign_price"),
                "currency": (final_state or {}).get("currency"),
                "hs_code": (final_state or {}).get("hs_code"),
                "hs_code_rationale": (final_state or {}).get("hs_code_rationale"),
                "tariff_rate": (final_state or {}).get("tariff_rate"),
                "exchange_rate": (final_state or {}).get("exchange_rate"),
                "total_krw": (
                    ((final_state or {}).get("total_foreign_price") or 0)
                    * ((final_state or {}).get("exchange_rate") or 0)
                ),
                "tax_amount": (final_state or {}).get("tax_amount"),
                "vat_amount": (final_state or {}).get("vat_amount"),
                "total_cost": total_cost,
                "report_content": (final_state or {}).get("report_content"),
                "report_paths": report_paths,
                "message": f"비용 계산이 완료되었습니다. 총 예상 비용: {total_cost:,.0f}원",
            }
            yield f"data: {_json.dumps(final_payload, ensure_ascii=False)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {_json.dumps({'done': True, 'success': False, 'error': str(e), 'session_id': request.session_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/v1/session/{session_id}", response_model=SessionResponse, tags=["Session"])
async def get_session(session_id: str):
    """세션 상태 조회"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    
    session = sessions[session_id]
    state = session["state"]
    
    # 현재 단계 결정
    if state.get("total_cost"):
        phase = "complete"
    elif state.get("hs_code_candidates"):
        phase = "hs_code_selection"
    else:
        phase = "analyzing"
    
    return SessionResponse(
        session_id=session_id,
        phase=phase,
        item_name=state.get("item_name"),
        hs_code=state.get("hs_code"),
        total_cost=state.get("total_cost"),
        created_at=session["created_at"],
        updated_at=session["updated_at"],
    )


@app.delete("/api/v1/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """세션 삭제"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    
    del sessions[session_id]
    return {"message": "세션이 삭제되었습니다.", "session_id": session_id}


@app.get("/api/v1/reports/{filename}", tags=["Reports"])
async def download_report(filename: str):
    """보고서 다운로드"""
    allowed_extensions = [".pdf", ".docx", ".xlsx"]
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")
    
    file_path = os.path.join(os.getcwd(), filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
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
