# run_preprocessing.py
"""
FAISS 인덱스 생성 스크립트 (최초 1회 또는 데이터 변경 시 실행)

생성 결과:
  vector_store/faiss_index      — 통합 인덱스 (PDF + CSV)
  vector_store/faiss_index_pdf  — PDF-only 인덱스 (balanced search용)

설정 (RAG 실험 최적값):
  청킹:   RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
  임베딩: nlpai-lab/KURE-v1
  검색:   듀얼 인덱스 PDF 쿼터=3 (app/tools.py에서 적용)
"""
import os
import re
import pandas as pd
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.models import get_embedding_model
from tqdm import tqdm

DATA_PATH = "./data"
VS_PATH   = "./vector_store"

PDF_FILENAME = "hsk_guide.pdf"
CSV_FILENAME = "tariff_by_hs.csv"

# PDF 포맷: (0101.29-1000), (0306.17-1090) 등
_HS_PDF_PATTERN = re.compile(r'\((\d{4})[.\-](\d{2})[.\-](\d{4})\)')


def _extract_hs_from_text(text: str) -> Optional[str]:
    """(XXXX.XX-XXXX) 패턴에서 10자리 HS 코드 추출. 없으면 None."""
    m = _HS_PDF_PATTERN.search(text)
    return (m.group(1) + m.group(2) + m.group(3)) if m else None


def load_pdf_docs() -> list:
    """
    PDF 로드 + 2000자 청킹 + HS 코드 메타데이터 carry-forward 주입.

    주입 우선순위:
      1) 청크 본문에서 (XXXX.XX-XXXX) 직접 추출
      2) 해당 페이지의 첫 번째 HS 코드 (page_hs_map)
      3) carry-forward — 직전 청크/페이지의 HS 코드
    """
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)

    # 페이지 단위로 먼저 로드하여 page_hs_map 구성
    pages = loader.load()
    page_hs_map: dict = {}
    for page_doc in pages:
        pnum = page_doc.metadata.get("page", -1)
        hs = _extract_hs_from_text(page_doc.page_content)
        if hs:
            page_hs_map[pnum] = hs

    # 2000자 청킹
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # HS 코드 메타데이터 주입
    last_hs: Optional[str] = None
    for doc in docs:
        doc.metadata["source"] = "HSK 품명규격 가이드"

        hs = _extract_hs_from_text(doc.page_content)
        if hs:
            last_hs = hs
        else:
            pnum = doc.metadata.get("page", -1)
            if pnum in page_hs_map:
                last_hs = page_hs_map[pnum]

        if last_hs:
            doc.metadata["hs_code"] = last_hs

    injected = sum(1 for d in docs if "hs_code" in d.metadata)
    print(f"  PDF 청크: {len(docs)}개 (HS 코드 주입: {injected}/{len(docs)}개)")
    return docs


def load_csv_docs() -> list:
    """CSV → 행(row)별 Document."""
    csv_path = os.path.join(DATA_PATH, CSV_FILENAME)
    df = pd.read_csv(csv_path, encoding="cp949")
    df = df.dropna(subset=["세번", "한글품명", "기본세율 - A"])

    docs = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="  CSV row → Document"):
        content = (
            f"HS 코드(세번): {row['세번']}\n"
            f"품명: {row['한글품명']} ({row['영문품명']})\n"
            f"기본세율: {row['기본세율 - A']}\n"
            f"WTO 협정세율: {row['WTO협정세율 - C']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "source": "HSK 관세율표",
                "hs_code": str(row["세번"]),
                "item_name_ko": row["한글품명"],
            },
        )
        docs.append(doc)
    print(f"  CSV 문서: {len(docs)}개")
    return docs


def create_vector_store():
    """통합 인덱스(PDF+CSV) + PDF-only 인덱스 생성 후 저장."""
    os.makedirs(VS_PATH, exist_ok=True)

    print("=== 문서 로드 ===")
    pdf_docs = load_pdf_docs()
    csv_docs = load_csv_docs()
    all_docs = pdf_docs + csv_docs
    print(f"  총 문서: {len(all_docs)}개")

    print("\n=== 임베딩 모델 로드 (nlpai-lab/KURE-v1) ===")
    embedding_model = get_embedding_model()

    # 1. 통합 인덱스 (PDF + CSV)
    print("\n=== 통합 인덱스 빌드 (PDF + CSV) ===")
    vector_store = FAISS.from_documents(all_docs, embedding_model)
    combined_path = os.path.join(VS_PATH, "faiss_index")
    vector_store.save_local(combined_path)
    print(f"  저장 완료: {combined_path}")

    # 2. PDF-only 인덱스 (balanced search용)
    print("\n=== PDF-only 인덱스 빌드 ===")
    pdf_store = FAISS.from_documents(pdf_docs, embedding_model)
    pdf_only_path = os.path.join(VS_PATH, "faiss_index_pdf")
    pdf_store.save_local(pdf_only_path)
    print(f"  저장 완료: {pdf_only_path}")

    print("\n=== 완료 ===")
    print(f"  통합 인덱스:   {combined_path}")
    print(f"  PDF-only 인덱스: {pdf_only_path}")
    print("  app 실행 시 balanced search (PDF 쿼터=3) 자동 적용됩니다.")


if __name__ == "__main__":
    create_vector_store()
