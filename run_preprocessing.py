# run_preprocessing.py
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.models import get_embedding_model
from tqdm import tqdm

DATA_PATH = "./data"
VS_PATH = "./vector_store"


def create_vector_store():
    """Load data from PDF and CSV, then create and save a FAISS vector store."""
    if not os.path.exists(VS_PATH):
        os.makedirs(VS_PATH)

    print("Loading documents...")

    # 1. PDF 로드 및 분할
    pdf_path = os.path.join(DATA_PATH, "hsk_guide.pdf")
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
    for doc in pdf_docs:
        doc.metadata['source'] = 'HSK 품명규격 가이드'

    # 2. CSV 로드 및 Document 객체로 변환
    csv_path = os.path.join(DATA_PATH, "tariff_by_hs.csv")

    # [수정된 부분] header=1 옵션을 제거하여 첫 번째 줄을 컬럼명으로 사용하도록 변경
    # 공공데이터 CSV 파일은 'cp949' 인코딩인 경우가 많습니다.
    df = pd.read_csv(csv_path, encoding='cp949')

    # (디버깅 팁) 만약 계속 에러가 발생하면 아래 코드의 주석을 해제하여 실제 컬럼명을 확인해보세요.
    # print("CSV 파일에서 읽어온 컬럼명:", df.columns)

    # 컬럼명이 정확히 일치하는지 확인 후 NaN 값 제거
    df = df.dropna(subset=['세번', '한글품명', '기본세율 - A'])

    csv_docs = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing CSV rows"):
        content = (
            f"HS 코드(세번): {row['세번']}\n"
            f"품명: {row['한글품명']} ({row['영문품명']})\n"
            f"기본세율: {row['기본세율 - A']}\n"
            f"WTO 협정세율: {row['WTO협정세율 - C']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                'source': 'HSK 관세율표',
                'hs_code': str(row['세번']),
                'item_name_ko': row['한글품명']
            }
        )
        csv_docs.append(doc)

    all_docs = pdf_docs + csv_docs
    print(f"Total documents to be embedded: {len(all_docs)}")

    # 3. 임베딩 모델 로드 및 FAISS 벡터 스토어 생성
    print("Creating and saving vector store... (This may take a while)")
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(all_docs, embedding_model)
    vector_store.save_local(os.path.join(VS_PATH, "faiss_index"))
    print("Vector store created and saved successfully at ./vector_store/faiss_index")


if __name__ == "__main__":
    create_vector_store()