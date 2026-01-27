# app/tools.py
"""
수입 비용 계산 에이전트를 위한 도구 정의
비동기 지원 및 명확한 리턴 타입 제공
"""
import os
import asyncio
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from docx import Document as DocxDocument

# 환경 변수 로드
load_dotenv()

# --- 임베딩 모델 로드 ---
def get_embedding_model():
    """한국어 임베딩 모델 로더"""
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}  # GPU 없이도 동작하도록 CPU 사용
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


# --- CSV 파일을 미리 로드하여 메모리에 저장 ---
TARIFF_DF = None
try:
    TARIFF_DF = pd.read_csv("./data/tariff_by_hs.csv", encoding='cp949', dtype={'세번': str, '잠정세율': str})
    print("tariff_by_hs.csv 파일을 성공적으로 로드했습니다.")
except FileNotFoundError:
    print("경고: ./data/tariff_by_hs.csv 파일을 찾을 수 없습니다.")


# --- Retriever 초기화 (Lazy Loading) ---
_retriever = None

def get_retriever():
    """저장된 FAISS 인덱스로부터 Retriever를 생성 (Lazy Loading)"""
    global _retriever
    if _retriever is not None:
        return _retriever
    
    vs_path = "./vector_store/faiss_index"
    if not os.path.exists(vs_path):
        raise FileNotFoundError("Vector store not found. Please run 'run_preprocessing.py' first.")

    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(vs_path, embedding_model, allow_dangerous_deserialization=True)
    _retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return _retriever


# ===== HS Code & Tax Finder 도구 =====

@tool("hs_code_search")
def hs_code_search(query: str) -> str:
    """
    사용자가 입력한 상품 설명(query)을 바탕으로 관련 HS 코드 정보와 품명 규격 가이드를 검색합니다.
    
    Args:
        query: 검색할 상품 설명 (예: "냉동 참치", "스마트워치", "노트북 컴퓨터")
    
    Returns:
        검색된 HS 코드 관련 정보 문자열 (예상 HS 코드, 품명, 관련 규정 포함)
    """
    print(f"[Tool] hs_code_search 실행: query={query}")
    try:
        retriever = get_retriever()
        retrieved_docs = retriever.invoke(query)
        if not retrieved_docs:
            return f"'{query}'에 대한 HS 코드 정보를 찾을 수 없습니다."
        return "\n\n".join([doc.page_content for doc in retrieved_docs])
    except Exception as e:
        return f"HS 코드 검색 중 오류 발생: {str(e)}"


def parse_tariff_rate(rate_str: str) -> float:
    """'8%', '무세', NaN 같은 문자열 세율을 float 숫자로 변환"""
    if pd.isna(rate_str):
        return np.inf
    if isinstance(rate_str, str):
        if '무세' in rate_str:
            return 0.0
        try:
            return float(rate_str.replace('%', '').strip())
        except ValueError:
            return np.inf
    return float(rate_str)


@tool("tariff_search_by_hs_code")
def tariff_search_by_hs_code(hs_code: str) -> str:
    """
    정확한 HS 코드(세번)를 입력받아 관세율 정보를 조회합니다.
    기본세율, 잠정세율, WTO 협정세율을 비교하여 가장 낮은 세율을 최종 적용 세율로 결정합니다.
    
    Args:
        hs_code: HS 코드 (예: "0303.43-0000", "8517.62-9090")
    
    Returns:
        관세율 정보 문자열 (기본세율, 잠정세율, WTO 협정세율, 최종 적용 세율, 과세 단위)
    """
    print(f"[Tool] tariff_search_by_hs_code 실행: hs_code={hs_code}")
    
    if TARIFF_DF is None:
        return "오류: 관세율 정보 파일(tariff_by_hs.csv)이 로드되지 않았습니다."

    hs_code_cleaned = hs_code.replace('.', '').replace('-', '')
    result_df = TARIFF_DF[
        TARIFF_DF['세번'].astype(str).str.replace('[.-]', '', regex=True).str.startswith(hs_code_cleaned[:6])]

    if result_df.empty:
        return f"HS 코드 '{hs_code}'에 해당하는 관세율 정보를 찾을 수 없습니다."

    target_row = result_df.iloc[0]
    item_name = target_row.get('한글품명', '알 수 없음')

    # 세율 정보 추출
    basic_rate_str = target_row.get('기본세율', '정보 없음')
    provisional_rate_str = target_row.get('잠정세율', '정보 없음')
    wto_rate_str = target_row.get('WTO협정세율', '정보 없음')

    # 단위 정보 추출
    weight_unit = target_row.get('중량단위', '정보 없음')
    quantity_unit = target_row.get('수량단위', '정보 없음')

    # 세율 변환 및 비교
    basic_rate = parse_tariff_rate(basic_rate_str)
    provisional_rate = parse_tariff_rate(provisional_rate_str)
    wto_rate = parse_tariff_rate(wto_rate_str)

    final_rate = min(basic_rate, provisional_rate, wto_rate)

    if final_rate == np.inf:
        return (
            f"HS 코드 '{hs_code}' ({item_name})에 대한 종가세(%) 정보를 찾을 수 없습니다. "
            f"종량세일 수 있으니 단위를 확인하세요. 중량단위: {weight_unit}, 수량단위: {quantity_unit}"
        )

    return (
        f"HS 코드 '{target_row['세번']}' ({item_name})의 세율 조회 결과:\n"
        f"- 기본세율: {basic_rate_str}\n"
        f"- 잠정세율: {provisional_rate_str}\n"
        f"- WTO 협정세율: {wto_rate_str}\n"
        f"-> 최종 적용 세율 (가장 낮은 값): {final_rate}%\n"
        f"- 과세 단위: 중량({weight_unit}), 수량({quantity_unit})"
    )


# ===== Tax Calculator 도구 =====

@tool("exchange_rate_loader")
def exchange_rate_loader(target_currency: str = "USD") -> Dict[str, Any]:
    """
    특정 국가의 통화(target_currency)와 대한민국 원(KRW) 사이의 현재 환율을 가져옵니다.
    
    Args:
        target_currency: 조회할 통화 코드 (기본값: USD, 예: EUR, JPY, CNY)
    
    Returns:
        환율 정보 딕셔너리 {"currency": str, "rate": float, "source": str}
    """
    print(f"[Tool] exchange_rate_loader 실행: currency={target_currency}")
    
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    default_rates = {"USD": 1350.0, "EUR": 1450.0, "JPY": 9.0, "CNY": 185.0}
    
    if not api_key:
        rate = default_rates.get(target_currency.upper(), 1350.0)
        return {
            "currency": target_currency.upper(),
            "rate": rate,
            "source": "default (API key not found)"
        }
    
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{target_currency}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            rate = data['conversion_rates'].get('KRW', default_rates.get(target_currency.upper(), 1350.0))
            return {
                "currency": target_currency.upper(),
                "rate": rate,
                "source": "exchangerate-api.com"
            }
    except requests.RequestException as e:
        print(f"[Tool] 환율 API 오류: {e}")
    
    rate = default_rates.get(target_currency.upper(), 1350.0)
    return {
        "currency": target_currency.upper(),
        "rate": rate,
        "source": "default (API error)"
    }


@tool("final_cost_calculator")
def final_cost_calculator(
    item_price: float, 
    quantity: int, 
    exchange_rate: float, 
    tariff_rate: float
) -> Dict[str, Any]:
    """
    상품 단가, 수량, 환율, 관세율을 입력받아 총 예상 수입 원가를 계산합니다.
    부가세(10%)는 관세가 포함된 금액에 부과됩니다.
    
    Args:
        item_price: 상품 단가 (외화)
        quantity: 수량
        exchange_rate: 환율 (KRW 기준)
        tariff_rate: 관세율 (%, 예: 8은 8%를 의미)
    
    Returns:
        계산 결과 딕셔너리 (각 단계별 금액 및 최종 비용)
    """
    print(f"[Tool] final_cost_calculator 실행: price={item_price}, qty={quantity}, rate={exchange_rate}, tariff={tariff_rate}")
    
    # 계산
    total_item_price_foreign = item_price * quantity
    total_item_price_krw = total_item_price_foreign * exchange_rate
    tariff = total_item_price_krw * (tariff_rate / 100)
    price_with_tariff = total_item_price_krw + tariff
    vat = price_with_tariff * 0.10
    total_cost = price_with_tariff + vat
    
    return {
        "total_item_price_foreign": total_item_price_foreign,
        "total_item_price_krw": total_item_price_krw,
        "tariff_rate_percent": tariff_rate,
        "tariff_amount": tariff,
        "vat_amount": vat,
        "total_cost": total_cost,
        "breakdown": (
            f"--- 최종 비용 계산 결과 ---\n"
            f"1. 총 물품 가격 (원화): {total_item_price_krw:,.0f} 원\n"
            f"   (단가 {item_price:,.2f} × 수량 {quantity} × 환율 {exchange_rate:,.2f})\n"
            f"2. 예상 관세 ({tariff_rate}%): {tariff:,.0f} 원\n"
            f"3. 예상 부가세 (10%): {vat:,.0f} 원\n"
            f"--------------------------------\n"
            f"   총 예상 수입 비용: {total_cost:,.0f} 원\n"
            f"--------------------------------"
        )
    }


# ===== Report Writer 도구 =====

@tool("pdf_report_exporter")
def pdf_report_exporter(report_content: str, filename: str = "report.pdf") -> str:
    """
    분석 결과를 담은 문자열(report_content)을 PDF 파일로 저장합니다.
    
    Args:
        report_content: 보고서 내용 문자열
        filename: 저장할 파일명 (기본값: report.pdf)
    
    Returns:
        저장 결과 메시지
    """
    print(f"[Tool] pdf_report_exporter 실행: filename={filename}")
    
    try:
        # 한글 폰트 등록 시도 (여러 경로 시도)
        font_name = 'Helvetica'
        font_paths = [
            './fonts/NanumGothic.ttf',                          # 프로젝트 내 fonts 폴더
            'fonts/NanumGothic.ttf',                            # 상대 경로
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # Docker/Linux 시스템 폰트
            'NanumGothic.ttf',                                  # 현재 디렉토리
            os.path.join(os.path.dirname(__file__), '..', 'fonts', 'NanumGothic.ttf'),  # app 상위 fonts
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
                    font_name = 'NanumGothic'
                    print(f"[Tool] 폰트 로드 성공: {font_path}")
                    break
            except Exception:
                continue
        
        if font_name == 'Helvetica':
            print("[Tool] Warning: NanumGothic font not found. Using Helvetica (한글 깨짐 가능).")
        
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        c.setFont(font_name, 10)
        
        text_object = c.beginText(40, height - 40)
        for line in report_content.split('\n'):
            # 마크다운 기호 제거
            clean_line = line.replace('#', '').replace('*', '').replace('>', '')
            text_object.textLine(clean_line)
        
        c.drawText(text_object)
        c.save()
        
        return f"PDF 보고서가 '{filename}'로 성공적으로 저장되었습니다."
    except Exception as e:
        return f"PDF 저장 중 오류 발생: {str(e)}"


@tool("word_report_exporter")
def word_report_exporter(report_content: str, filename: str = "report.docx") -> str:
    """
    분석 결과를 담은 문자열(report_content)을 Word 파일로 저장합니다.
    
    Args:
        report_content: 보고서 내용 문자열
        filename: 저장할 파일명 (기본값: report.docx)
    
    Returns:
        저장 결과 메시지
    """
    print(f"[Tool] word_report_exporter 실행: filename={filename}")
    
    try:
        doc = DocxDocument()
        doc.add_heading('HS 코드 및 수입 원가 분석 보고서', level=1)
        
        # 마크다운 기호 제거 후 단락 추가
        for line in report_content.split('\n'):
            clean_line = line.replace('#', '').replace('*', '').replace('>', '').strip()
            if clean_line:
                doc.add_paragraph(clean_line)
        
        doc.save(filename)
        return f"Word 보고서가 '{filename}'로 성공적으로 저장되었습니다."
    except Exception as e:
        return f"Word 저장 중 오류 발생: {str(e)}"


@tool("excel_report_exporter")
def excel_report_exporter(data: Dict[str, Any], filename: str = "report.xlsx") -> str:
    """
    분석 결과 데이터를 Excel 파일로 저장합니다.
    
    Args:
        data: 보고서 데이터 딕셔너리 (예: {"물품명": "스마트워치", "HS코드": "8517.62", ...})
        filename: 저장할 파일명 (기본값: report.xlsx)
    
    Returns:
        저장 결과 메시지
    """
    print(f"[Tool] excel_report_exporter 실행: filename={filename}")
    
    try:
        # 딕셔너리를 DataFrame으로 변환
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return "엑셀 파일 생성 실패: 입력 데이터는 딕셔너리 또는 리스트여야 합니다."
        
        df.to_excel(filename, index=False)
        return f"Excel 보고서가 '{filename}'로 성공적으로 저장되었습니다."
    except Exception as e:
        return f"Excel 저장 중 오류 발생: {str(e)}"


# ===== 도구 그룹 정의 =====

# HS Code & Tax Finder 에이전트용 도구
hs_code_finder_tools = [hs_code_search, tariff_search_by_hs_code]

# Tax Calculator 에이전트용 도구
tax_calculator_tools = [exchange_rate_loader, final_cost_calculator]

# Report Writer 에이전트용 도구
report_writer_tools = [pdf_report_exporter, word_report_exporter, excel_report_exporter]

# 모든 도구 리스트
all_tools = hs_code_finder_tools + tax_calculator_tools + report_writer_tools
