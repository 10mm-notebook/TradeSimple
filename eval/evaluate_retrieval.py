"""
eval/evaluate_retrieval.py
FAISS 벡터 검색 정확도 평가 (LLM 호출 없음 — 무료)

사용법
──────
  # 기본 실행 (k=5, 쿼리 확장 없음)
  python -m eval.evaluate_retrieval

  # k=10, 쿼리 확장 적용, description 필드 사용
  python -m eval.evaluate_retrieval --k 10 --expand --query-field description

  # 커스텀 벡터 스토어 경로 사용 (실험용 인덱스)
  python -m eval.evaluate_retrieval --vs-path vector_store/exp_small_multilingual_e5/faiss_index --embedding multilingual_e5

  # BM25+dense hybrid 검색 활성화
  python -m eval.evaluate_retrieval --hybrid

  # CrossEncoder 리랭킹 활성화
  python -m eval.evaluate_retrieval --rerank --rerank-top-n 10

  # 결과를 JSON 파일로 저장
  python -m eval.evaluate_retrieval --k 5 --output eval/results/baseline.json

  # 케이스별 출력 없이 요약만
  python -m eval.evaluate_retrieval --quiet
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import compute_retrieval_metrics, aggregate_metrics, format_metrics_table, normalize_hs


# ── 데이터 로드 ────────────────────────────────────────────────

def load_test_cases(path: Optional[str] = None) -> List[Dict[str, Any]]:
    if path is None:
        path = ROOT / "eval" / "dataset" / "test_cases.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 문서에서 HS 코드 추출 ──────────────────────────────────────

def extract_hs_from_doc(doc) -> Optional[str]:
    """
    FAISS 문서에서 HS 코드를 추출.

    소스별 포맷
    ──────────────────────────────────────────────────────────
    CSV(HSK 관세율표):
      - metadata['hs_code'] = '306170000'  ← 우선 사용
      - page_content: "HS 코드(세번): 306170000\n품명: ..."

    PDF(HSK 품명규격 가이드):
      - metadata에 hs_code 없음
      - page_content 앞부분: "(0306.17-1090) 냉동한 그 밖의 새우류..."
    ──────────────────────────────────────────────────────────
    """
    # 1. CSV 문서 metadata (가장 신뢰도 높음)
    if hasattr(doc, "metadata") and "hs_code" in doc.metadata:
        raw = str(doc.metadata["hs_code"]).strip()
        if raw and raw.lower() not in ("none", "nan", ""):
            return raw

    content = doc.page_content

    # 2. CSV page_content 포맷: "HS 코드(세번): 303430000"
    m = re.search(r"HS\s*코드\(세번\):\s*(\d[\d.]*)", content)
    if m:
        return m.group(1)

    # 3. PDF 품명규격 가이드 포맷: "(0306.17-1090)" 또는 "(0302.13-0000)"
    #    세번 XXXX.XX-XXXX 가 소괄호로 감싸진 형태
    m = re.search(r"\((\d{4}[.\-]\d{2}[.\-]\d{4})\)", content)
    if m:
        return m.group(1)

    return None


def doc_source_type(doc) -> str:
    """문서 소스 분류 ('csv' | 'pdf' | 'unknown')"""
    src = (doc.metadata or {}).get("source", "")
    if "관세율표" in src:
        return "csv"
    if "품명규격" in src or "가이드" in src:
        return "pdf"
    return "unknown"


# ── BM25 + Hybrid 검색 헬퍼 ───────────────────────────────────

def _load_docs_for_bm25(idx_path: Path) -> List:
    """
    docs.json → Document 리스트.
    docs.json이 없으면 빈 리스트 반환.
    """
    from langchain_core.documents import Document

    docs_json = idx_path.parent / "docs.json"
    if not docs_json.exists():
        return []

    with docs_json.open(encoding="utf-8") as f:
        records = json.load(f)

    return [Document(page_content=r["content"], metadata=r["metadata"]) for r in records]


def _build_bm25_index(docs: List):
    """rank-bm25 BM25Okapi 인덱스 생성. 설치 없으면 None 반환."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None

    # 공백 기반 토큰화 (한국어 형태소 분석 없이도 어느 정도 동작)
    tokenized = [doc.page_content.split() for doc in docs]
    return BM25Okapi(tokenized)


def _hybrid_rrf_search(
    query: str,
    vector_store,
    bm25_index,
    bm25_docs: List,
    k: int,
    rrf_k: int = 60,
) -> List:
    """
    BM25 top-(k*2) + dense top-(k*2) → RRF 스코어 합산 → 상위 k 반환.

    RRF 스코어: Σ 1/(rank + rrf_k)
    """
    candidate_k = min(k * 2, len(bm25_docs)) if bm25_docs else k * 2

    # Dense 검색
    dense_docs = vector_store.similarity_search(query, k=candidate_k)

    # BM25 검색
    bm25_docs_result: List = []
    if bm25_index is not None and bm25_docs:
        tokens = query.split()
        scores = bm25_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_k]
        bm25_docs_result = [bm25_docs[i] for i in top_indices]

    # RRF 융합
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Any] = {}

    def _doc_id(doc) -> str:
        """문서 고유 키 (metadata hs_code 또는 content 앞 50자)."""
        hs = (doc.metadata or {}).get("hs_code", "")
        if hs:
            return f"hs:{hs}"
        return doc.page_content[:50]

    for rank, doc in enumerate(dense_docs, start=1):
        did = _doc_id(doc)
        doc_scores[did] = doc_scores.get(did, 0) + 1.0 / (rank + rrf_k)
        doc_map[did] = doc

    for rank, doc in enumerate(bm25_docs_result, start=1):
        did = _doc_id(doc)
        doc_scores[did] = doc_scores.get(did, 0) + 1.0 / (rank + rrf_k)
        doc_map[did] = doc

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[did] for did, _ in ranked[:k]]


# ── Balanced (듀얼 인덱스) 검색 헬퍼 ──────────────────────────

def _balanced_search(
    query: str,
    vector_store,        # 통합(CSV+PDF) 인덱스 — CSV가 압도적이므로 실질적 CSV 소스
    pdf_store,           # PDF-only 인덱스
    k: int,
    pdf_quota: int,
) -> List:
    """
    통합 인덱스에서 (k - pdf_quota)개 + PDF-only 인덱스에서 pdf_quota개.

    결과 순서: CSV docs (앞) → PDF docs (뒤)
    Hit@k 평가는 순서 무관; MRR에서는 CSV가 앞에 위치.
    """
    csv_k = max(0, k - pdf_quota)
    pdf_k = min(k, pdf_quota)

    csv_docs = vector_store.similarity_search(query, k=csv_k) if csv_k > 0 else []
    pdf_docs = pdf_store.similarity_search(query, k=pdf_k)

    return csv_docs + pdf_docs


# ── CrossEncoder 리랭킹 헬퍼 ───────────────────────────────────

_reranker = None  # 모듈 수준 캐시 (한 번만 로드)


def _rerank_docs(query: str, docs: List, top_n: int) -> List:
    """
    BAAI/bge-reranker-v2-m3 CrossEncoder로 docs 재정렬 후 top_n 반환.
    sentence-transformers 미설치 시 원본 docs 반환.
    """
    global _reranker
    if not docs:
        return docs

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("경고: sentence-transformers 미설치 — 리랭킹 건너뜀.")
        return docs[:top_n]

    if _reranker is None:
        print("CrossEncoder 로드 중: BAAI/bge-reranker-v2-m3 ...")
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    pairs = [[query, doc.page_content] for doc in docs]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


# ── 평가 실행 ──────────────────────────────────────────────────

def run_retrieval_eval(
    test_cases: List[Dict[str, Any]],
    k: int = 5,
    use_expansion: bool = False,
    query_field: str = "item_name",
    verbose: bool = True,
    vs_path: Optional[Path] = None,
    embedding_name: str = "baseline",
    hybrid: bool = False,
    rerank: bool = False,
    rerank_top_n: int = 10,
    pdf_quota: int = 0,
    pdf_vs_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    FAISS 검색 정확도 평가.

    Args:
        test_cases    : test_cases.json 에서 로드한 케이스 목록
        k             : 검색 상위 k개
        use_expansion : LLM 쿼리 확장 사용 여부
        query_field   : 쿼리로 사용할 필드 ('item_name', 'description', 'detailed_description')
        verbose       : 케이스별 결과 출력 여부
        vs_path       : 벡터 스토어 경로 (기본: ROOT/vector_store/faiss_index)
        embedding_name: 임베딩 모델 이름 (EMBEDDING_REGISTRY 키, 기본: 'baseline')
        hybrid        : BM25+dense hybrid 검색 활성화
        rerank        : CrossEncoder 리랭킹 활성화
        rerank_top_n  : 리랭킹 후보 수
        pdf_quota     : PDF-only 인덱스에서 강제 포함할 문서 수 (0=기존 방식)
        pdf_vs_path   : PDF-only FAISS 인덱스 경로 (pdf_quota > 0 시 필요)

    Returns:
        {"config": {...}, "per_case": [...], "aggregate": {...}}
    """
    from langchain_community.vectorstores import FAISS

    # 벡터 스토어 경로 결정
    if vs_path is None:
        vs_path = ROOT / "vector_store" / "faiss_index"
    vs_path = Path(vs_path)

    if not vs_path.exists():
        raise FileNotFoundError(
            f"벡터 스토어가 없습니다: {vs_path}\n"
            "먼저 'python run_preprocessing.py' 또는 "
            "'python -m eval.preprocess_experiment' 를 실행하세요."
        )

    # 임베딩 모델 로드
    print(f"벡터 스토어 로드 중: {vs_path}")
    try:
        from eval.embedding_registry import EMBEDDING_REGISTRY
        emb_cfg = EMBEDDING_REGISTRY[embedding_name]
        embedding_model = emb_cfg.load()
    except (ImportError, KeyError):
        from app.models import get_embedding_model
        embedding_model = get_embedding_model()

    vector_store = FAISS.load_local(
        str(vs_path), embedding_model, allow_dangerous_deserialization=True
    )

    # BM25 인덱스 준비 (hybrid 모드)
    bm25_index = None
    bm25_docs: List = []
    if hybrid:
        print("BM25 인덱스 준비 중...")
        bm25_docs = _load_docs_for_bm25(vs_path)
        if bm25_docs:
            bm25_index = _build_bm25_index(bm25_docs)
            if bm25_index is None:
                print("경고: rank-bm25 미설치 — BM25 없이 dense 검색만 사용합니다.")
                print("  설치: pip install rank-bm25")
            else:
                print(f"BM25 인덱스 준비 완료 ({len(bm25_docs)}개 문서)")
        else:
            print("경고: docs.json 없음 — BM25 없이 dense 검색만 사용합니다.")
            print("  eval.preprocess_experiment 로 인덱스를 빌드하면 docs.json이 생성됩니다.")

    # PDF-only 인덱스 로드 (balanced 모드)
    pdf_store = None
    if pdf_quota > 0:
        if pdf_vs_path is None or not Path(pdf_vs_path).exists():
            print(f"경고: PDF-only 인덱스 없음({pdf_vs_path}) → 기존 검색 방식으로 대체")
            pdf_quota = 0
        else:
            pdf_vs_path = Path(pdf_vs_path)
            print(f"PDF-only 인덱스 로드: {pdf_vs_path}  (pdf_quota={pdf_quota})")
            pdf_store = FAISS.load_local(
                str(pdf_vs_path), embedding_model, allow_dangerous_deserialization=True
            )

    if rerank:
        print(f"CrossEncoder 리랭킹 활성화 (top_n={rerank_top_n})")

    # 쿼리 확장 함수 (선택)
    expand_fn = None
    if use_expansion:
        try:
            from app.tools import expand_hs_search_query
            expand_fn = expand_hs_search_query
            print("쿼리 확장(LLM) 활성화")
        except ImportError:
            print("경고: expand_hs_search_query 를 가져올 수 없어 확장 없이 진행합니다.")

    ks_to_eval = sorted({1, 3, k})
    per_case: List[Dict[str, Any]] = []

    for case in test_cases:
        query = case[query_field]

        # 쿼리 확장
        if expand_fn:
            try:
                expanded = expand_fn(query)
                if expanded and expanded != query:
                    query = f"{query} {expanded}"
            except Exception:
                pass

        # 검색
        if pdf_store is not None and pdf_quota > 0:
            # balanced 모드: 통합 인덱스(CSV 중심) + PDF-only 인덱스
            docs = _balanced_search(query, vector_store, pdf_store, k=k, pdf_quota=pdf_quota)
        elif hybrid and bm25_index is not None:
            search_k = max(k, 5)
            docs = _hybrid_rrf_search(
                query, vector_store, bm25_index, bm25_docs, k=search_k
            )
        else:
            search_k = max(k, 5)
            docs = vector_store.similarity_search(query, k=search_k)

        # 리랭킹
        if rerank and docs:
            docs = _rerank_docs(query, docs, top_n=max(k, rerank_top_n))

        # 최종 상위 k개만 평가 대상
        docs = docs[:k]

        # 문서별 소스 분류 및 HS 코드 추출 (None 제거)
        src_types    = [doc_source_type(d) for d in docs]
        ranked_codes = [c for c in (extract_hs_from_doc(d) for d in docs) if c]

        expected = case["expected_hs6"]
        metrics = compute_retrieval_metrics(ranked_codes, expected, ks=ks_to_eval)

        n_csv = src_types.count("csv")
        n_pdf = src_types.count("pdf")
        case_result = {
            "id":            case["id"],
            "item_name":     case["item_name"],
            "category":      case.get("category", ""),
            "difficulty":    case.get("difficulty", 1),
            "expected_hs6":  expected,
            "query_used":    query,
            "top_retrieved": [normalize_hs(c) for c in ranked_codes[:5]],
            "src_breakdown": {"csv": n_csv, "pdf": n_pdf, "k": len(docs)},
            "metrics":       metrics,
        }
        per_case.append(case_result)

        if verbose:
            hit6 = bool(metrics.get(f"hit@{k}_hs6"))
            hit4 = bool(metrics.get(f"hit@{k}_hs4"))
            icon = "✅" if hit6 else ("🟡" if hit4 else "❌")
            top3 = case_result["top_retrieved"][:3]
            print(
                f"  {icon} [{case['id']:02d}] {case['item_name']:<15} "
                f"exp={normalize_hs(expected)}  top3={top3}  "
                f"(csv={n_csv}/pdf={n_pdf})"
            )

    agg = aggregate_metrics([c["metrics"] for c in per_case])
    return {
        "config": {
            "k": k,
            "use_expansion": use_expansion,
            "query_field": query_field,
            "vs_path": str(vs_path),
            "embedding_name": embedding_name,
            "hybrid": hybrid,
            "rerank": rerank,
            "rerank_top_n": rerank_top_n,
            "pdf_quota": pdf_quota,
        },
        "per_case":  per_case,
        "aggregate": agg,
    }


# ── 결과 요약 출력 ─────────────────────────────────────────────

def print_summary(result: Dict[str, Any], ks: List[int] = (1, 3, 5)) -> None:
    cfg = result["config"]
    n   = len(result["per_case"])
    hybrid_tag = " hybrid" if cfg.get("hybrid") else ""
    rerank_tag = " rerank" if cfg.get("rerank") else ""
    title = (
        f"검색 평가 결과  |  k={cfg['k']}  "
        f"쿼리확장={'O' if cfg['use_expansion'] else 'X'}  "
        f"필드={cfg['query_field']}{hybrid_tag}{rerank_tag}  |  n={n}"
    )
    print(format_metrics_table(result["aggregate"], ks=list(ks), title=title))

    # CSV vs PDF 소스 비율 분석
    if result["per_case"] and "src_breakdown" in result["per_case"][0]:
        total_k   = sum(c["src_breakdown"]["k"]   for c in result["per_case"])
        total_csv = sum(c["src_breakdown"]["csv"] for c in result["per_case"])
        total_pdf = sum(c["src_breakdown"]["pdf"] for c in result["per_case"])
        csv_ratio = total_csv / total_k if total_k else 0
        pdf_ratio = total_pdf / total_k if total_k else 0
        print(f"\n  문서 소스 비율 (전체 검색 k={result['config']['k']}×{len(result['per_case'])}케이스)")
        print(f"    CSV(관세율표): {total_csv:4d}개  ({csv_ratio:.0%})")
        print(f"    PDF(품명규격): {total_pdf:4d}개  ({pdf_ratio:.0%})")

        if pdf_ratio > 0.5:
            print(f"\n  ⚠  PDF 문서 비율이 {pdf_ratio:.0%}으로 높습니다.")
            print("     → run_preprocessing.py 재실행 시 CSV 비중을 늘리는 것을 권장합니다.")

    # 난이도별 분석
    for diff, label in [(1, "쉬움"), (2, "보통"), (3, "어려움")]:
        subset = [c["metrics"] for c in result["per_case"] if c.get("difficulty") == diff]
        if subset:
            sub_agg = aggregate_metrics(subset)
            k_max = cfg["k"]
            h6  = sub_agg.get(f"hit@{k_max}_hs6", 0)
            h4  = sub_agg.get(f"hit@{k_max}_hs4", 0)
            mrr = sub_agg.get("mrr_hs6", 0)
            print(f"  난이도 {diff}({label}, n={len(subset):2d}):  "
                  f"Hit@{k_max}_hs6={h6:.0%}  Hit@{k_max}_hs4={h4:.0%}  MRR={mrr:.4f}")


# ── CLI ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAISS 검색 정확도 평가 (LLM 호출 없음)"
    )
    parser.add_argument("--k", type=int, default=5, help="검색 상위 k개 (기본: 5)")
    parser.add_argument("--expand", action="store_true", help="LLM 쿼리 확장 사용")
    parser.add_argument(
        "--query-field",
        choices=["item_name", "description", "detailed_description"],
        default="item_name",
        help="쿼리로 사용할 케이스 필드 (기본: item_name)",
    )
    parser.add_argument(
        "--vs-path", dest="vs_path",
        help="커스텀 벡터 스토어 경로 (기본: vector_store/faiss_index)",
    )
    parser.add_argument(
        "--embedding", default="baseline",
        help="임베딩 모델 이름 (EMBEDDING_REGISTRY 키, 기본: baseline)",
    )
    parser.add_argument("--hybrid", action="store_true", help="BM25+dense hybrid 검색 활성화")
    parser.add_argument("--rerank", action="store_true", help="CrossEncoder 리랭킹 활성화")
    parser.add_argument(
        "--rerank-top-n", dest="rerank_top_n", type=int, default=10,
        help="리랭킹 후보 수 (기본: 10)",
    )
    parser.add_argument("--dataset", help="테스트 케이스 JSON 경로")
    parser.add_argument("--output", help="결과 저장 JSON 경로")
    parser.add_argument("--quiet", action="store_true", help="케이스별 출력 생략")
    args = parser.parse_args()

    cases = load_test_cases(args.dataset)
    print(f"\n테스트 케이스 {len(cases)}개 로드")

    vs_path_arg = Path(args.vs_path) if args.vs_path else None

    result = run_retrieval_eval(
        cases,
        k=args.k,
        use_expansion=args.expand,
        query_field=args.query_field,
        verbose=not args.quiet,
        vs_path=vs_path_arg,
        embedding_name=args.embedding,
        hybrid=args.hybrid,
        rerank=args.rerank,
        rerank_top_n=args.rerank_top_n,
    )

    print_summary(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
