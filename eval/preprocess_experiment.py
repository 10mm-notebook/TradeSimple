"""
eval/preprocess_experiment.py
(청킹 전략, 임베딩 모델) 조합별 FAISS 인덱스 빌더

인덱스 저장 경로
────────────────────────────────────────────────────────
  baseline 청킹 + baseline 임베딩 → vector_store/exp_baseline_baseline/faiss_index
  small 청킹    + multilingual_e5  → vector_store/exp_small_multilingual_e5/faiss_index
  ...

docs.json 도 같은 디렉터리에 저장 (BM25 hybrid search 용).

사용법
──────
  # 등록된 전략 목록 확인
  python -m eval.preprocess_experiment --list

  # 특정 조합 빌드
  python -m eval.preprocess_experiment --chunking small large --embedding multilingual_e5 kure_v1

  # 모든 CPU 모델 × 모든 청킹 전략 빌드
  python -m eval.preprocess_experiment --all-cpu

  # 기존 인덱스가 있어도 재빌드
  python -m eval.preprocess_experiment --chunking baseline --embedding baseline --force
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.chunking_strategies import CHUNKING_REGISTRY, list_strategies
from eval.embedding_registry import EMBEDDING_REGISTRY, CPU_MODELS, list_embeddings

DATA_PATH   = ROOT / "data"
VS_BASE     = ROOT / "vector_store"
PDF_FILENAME = "hsk_guide.pdf"
CSV_FILENAME = "tariff_by_hs.csv"


# ── HS 코드 추출 패턴 ────────────────────────────────────────────
# PDF 포맷: (0101.29-1000), (0306.17-1090) 등
# → group(1)+group(2)+group(3) = "0101291000", "0306171090" (10자리 raw code)
import re as _re
_HS_PDF_PATTERN = _re.compile(r'\((\d{4})[.\-](\d{2})[.\-](\d{4})\)')


def _extract_hs_from_text(text: str) -> Optional[str]:
    """(XXXX.XX-XXXX) 패턴에서 10자리 HS 코드 추출. 없으면 None."""
    m = _HS_PDF_PATTERN.search(text)
    return (m.group(1) + m.group(2) + m.group(3)) if m else None


# ── 문서 로드 ────────────────────────────────────────────────────

def load_pdf_docs(chunking_name: str) -> List:
    """PDF 로드 후 청킹 전략 적용. 각 문서에 HS 코드를 메타데이터로 주입.

    주입 우선순위:
      1) 청크 본문에서 직접 추출  (0101.29-1000) 패턴
      2) 해당 페이지의 첫 번째 HS 코드 (page_hs_map)
      3) carry-forward — 직전 청크/페이지의 HS 코드
    """
    from langchain_community.document_loaders import PyPDFLoader

    pdf_path = DATA_PATH / PDF_FILENAME
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일이 없습니다: {pdf_path}")

    cfg = CHUNKING_REGISTRY[chunking_name]
    splitter = cfg.get_splitter()

    loader = PyPDFLoader(str(pdf_path))

    # 항상 페이지 단위로 먼저 로드 (page_hs_map 구성 + 청킹 재사용)
    pages = loader.load()

    # 페이지번호 → 첫 번째 HS 코드 매핑
    page_hs_map: dict = {}
    for page_doc in pages:
        pnum = page_doc.metadata.get("page", -1)
        hs = _extract_hs_from_text(page_doc.page_content)
        if hs:
            page_hs_map[pnum] = hs

    # 청킹 전략 적용 (PDF를 한 번만 로드)
    docs = pages if splitter is None else splitter.split_documents(pages)

    # HS 코드 메타데이터 주입
    last_hs: Optional[str] = None
    for doc in docs:
        doc.metadata["source"] = "HSK 품명규격 가이드"

        # 1) 청크 본문에서 직접 추출
        hs = _extract_hs_from_text(doc.page_content)
        if hs:
            last_hs = hs
        else:
            # 2) 페이지 첫 HS 코드 fallback
            pnum = doc.metadata.get("page", -1)
            if pnum in page_hs_map:
                last_hs = page_hs_map[pnum]
            # 3) carry-forward: last_hs 그대로 유지

        if last_hs:
            doc.metadata["hs_code"] = last_hs

    hs_injected = sum(1 for d in docs if "hs_code" in d.metadata)
    print(f"  HS 코드 메타데이터 주입: {hs_injected}/{len(docs)}개 문서")

    return docs


def load_csv_docs() -> List:
    """CSV → row 별 Document (청킹 전략과 무관, 항상 동일)."""
    import pandas as pd
    from langchain_core.documents import Document

    csv_path = DATA_PATH / CSV_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    df = pd.read_csv(csv_path, encoding="cp949")
    df = df.dropna(subset=["세번", "한글품명", "기본세율 - A"])

    docs = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="CSV row → Document"):
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
    return docs


# ── docs.json 저장/로드 ─────────────────────────────────────────

def save_docs_json(docs: List, out_dir: Path) -> None:
    """BM25 hybrid search 용 docs.json 저장."""
    records = [
        {"content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]
    out_path = out_dir / "docs.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"  docs.json 저장 완료: {out_path}  ({len(records)}건)")


# ── 인덱스 경로 헬퍼 ────────────────────────────────────────────

def vs_dir(chunking_name: str, embedding_name: str) -> Path:
    """실험용 벡터 스토어 저장 디렉터리."""
    return VS_BASE / f"exp_{chunking_name}_{embedding_name}"


def vs_path(chunking_name: str, embedding_name: str) -> Path:
    return vs_dir(chunking_name, embedding_name) / "faiss_index"


# ── 빌더 ─────────────────────────────────────────────────────────

def build_pdf_only_index(
    chunking_name: str,
    embedding_name: str,
    force: bool = False,
) -> Path:
    """
    PDF 문서만으로 구성된 FAISS 인덱스 빌드 (balanced 검색용).

    저장 경로: vector_store/exp_{chunking}_{embedding}_pdf/faiss_index
    """
    from langchain_community.vectorstores import FAISS

    out_dir  = VS_BASE / f"exp_{chunking_name}_{embedding_name}_pdf"
    idx_path = out_dir / "faiss_index"

    if idx_path.exists() and not force:
        print(f"  건너뜀 (이미 존재): {idx_path}  (--force 로 재빌드)")
        return idx_path

    print(f"\n▶ PDF-only 빌드: chunking={chunking_name}, embedding={embedding_name}")
    print(f"   저장 경로: {idx_path}")

    print("  PDF 로드 중...")
    pdf_docs = load_pdf_docs(chunking_name)
    print(f"  PDF 문서: {len(pdf_docs)}개 (CSV 제외)")

    print(f"  임베딩 모델 로드 중: {embedding_name} ...")
    emb_cfg   = EMBEDDING_REGISTRY[embedding_name]
    embedding = emb_cfg.load()

    print("  FAISS 인덱스 빌드 중...")
    vector_store = FAISS.from_documents(pdf_docs, embedding)

    out_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(idx_path))
    print(f"  PDF-only 인덱스 저장 완료: {idx_path}")

    save_docs_json(pdf_docs, out_dir)
    return idx_path


def build_index(
    chunking_name: str,
    embedding_name: str,
    force: bool = False,
) -> Path:
    """
    (청킹, 임베딩) 조합 FAISS 인덱스 빌드.

    Args:
        chunking_name:  CHUNKING_REGISTRY 키
        embedding_name: EMBEDDING_REGISTRY 키
        force:          True면 기존 인덱스 재빌드

    Returns:
        저장된 faiss_index 경로
    """
    from langchain_community.vectorstores import FAISS

    if chunking_name not in CHUNKING_REGISTRY:
        raise ValueError(
            f"알 수 없는 청킹 전략: '{chunking_name}'\n"
            f"사용 가능: {list(CHUNKING_REGISTRY.keys())}"
        )
    if embedding_name not in EMBEDDING_REGISTRY:
        raise ValueError(
            f"알 수 없는 임베딩 모델: '{embedding_name}'\n"
            f"사용 가능: {list(EMBEDDING_REGISTRY.keys())}"
        )

    out_dir  = vs_dir(chunking_name, embedding_name)
    idx_path = out_dir / "faiss_index"

    if idx_path.exists() and not force:
        print(f"  건너뜀 (이미 존재): {idx_path}  (--force 로 재빌드)")
        return idx_path

    print(f"\n▶ 빌드: chunking={chunking_name}, embedding={embedding_name}")
    print(f"   저장 경로: {idx_path}")

    # 문서 로드
    print("  PDF 로드 중...")
    pdf_docs = load_pdf_docs(chunking_name)
    print(f"  PDF 문서: {len(pdf_docs)}개")

    print("  CSV 로드 중...")
    csv_docs = load_csv_docs()
    print(f"  CSV 문서: {len(csv_docs)}개")

    all_docs = pdf_docs + csv_docs
    print(f"  총 문서: {len(all_docs)}개")

    # 임베딩 모델 로드
    print(f"  임베딩 모델 로드 중: {embedding_name} ...")
    emb_cfg   = EMBEDDING_REGISTRY[embedding_name]
    embedding = emb_cfg.load()

    # FAISS 인덱스 빌드
    print("  FAISS 인덱스 빌드 중... (시간이 걸릴 수 있습니다)")
    vector_store = FAISS.from_documents(all_docs, embedding)

    # 저장
    out_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(idx_path))
    print(f"  FAISS 인덱스 저장 완료: {idx_path}")

    # BM25 용 docs.json 저장
    save_docs_json(all_docs, out_dir)

    return idx_path


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="(청킹, 임베딩) 조합별 FAISS 인덱스 빌더",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="등록된 청킹 전략 및 임베딩 모델 목록 출력 후 종료",
    )
    parser.add_argument(
        "--chunking", nargs="+", metavar="NAME",
        help="사용할 청킹 전략 이름 (복수 지정 가능)",
    )
    parser.add_argument(
        "--embedding", nargs="+", metavar="NAME",
        help="사용할 임베딩 모델 이름 (복수 지정 가능)",
    )
    parser.add_argument(
        "--all-cpu", action="store_true",
        help="모든 청킹 전략 × CPU 임베딩 모델 조합 빌드",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="기존 인덱스가 있어도 재빌드",
    )
    parser.add_argument(
        "--pdf-only", action="store_true",
        help="PDF 전용 인덱스 빌드 (balanced 검색 실험용, _pdf 접미사)",
    )
    args = parser.parse_args()

    if args.list:
        list_strategies()
        list_embeddings()
        return

    # 대상 결정
    if args.all_cpu:
        chunking_names  = list(CHUNKING_REGISTRY.keys())
        embedding_names = CPU_MODELS
    else:
        chunking_names  = args.chunking  or ["baseline"]
        embedding_names = args.embedding or ["baseline"]

    # 유효성 검사
    for c in chunking_names:
        if c not in CHUNKING_REGISTRY:
            print(f"오류: 알 수 없는 청킹 전략 '{c}'")
            print(f"사용 가능: {list(CHUNKING_REGISTRY.keys())}")
            return
    for e in embedding_names:
        if e not in EMBEDDING_REGISTRY:
            print(f"오류: 알 수 없는 임베딩 모델 '{e}'")
            print(f"사용 가능: {list(EMBEDDING_REGISTRY.keys())}")
            return

    total = len(chunking_names) * len(embedding_names)
    mode  = "PDF-only" if args.pdf_only else "통합"
    print(f"\n총 {total}개 조합 {mode} 빌드 예정")
    print(f"  청킹:    {chunking_names}")
    print(f"  임베딩:  {embedding_names}")

    built, skipped, failed = [], [], []

    builder = build_pdf_only_index if args.pdf_only else build_index

    for chunking in chunking_names:
        for embedding in embedding_names:
            try:
                idx = builder(chunking, embedding, force=args.force)
                if idx.exists():
                    built.append(f"{chunking}_{embedding}")
                else:
                    skipped.append(f"{chunking}_{embedding}")
            except Exception as exc:
                print(f"  ❌ 오류 ({chunking}/{embedding}): {exc}")
                failed.append(f"{chunking}_{embedding}")

    print(f"\n{'='*50}")
    print(f"  완료: {len(built)}개  건너뜀: {len(skipped)}개  실패: {len(failed)}개")
    if failed:
        print(f"  실패 목록: {failed}")


if __name__ == "__main__":
    main()
