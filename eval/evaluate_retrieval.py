"""
eval/evaluate_retrieval.py
FAISS 벡터 검색 정확도 평가 (LLM 호출 없음 — 무료)

사용법
──────
  # 기본 실행 (k=5, 쿼리 확장 없음)
  python -m eval.evaluate_retrieval

  # k=10, 쿼리 확장 적용, description 필드 사용
  python -m eval.evaluate_retrieval --k 10 --expand --query-field description

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


# ── 평가 실행 ──────────────────────────────────────────────────

def run_retrieval_eval(
    test_cases: List[Dict[str, Any]],
    k: int = 5,
    use_expansion: bool = False,
    query_field: str = "item_name",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    FAISS 검색 정확도 평가.

    Args:
        test_cases  : test_cases.json 에서 로드한 케이스 목록
        k           : 검색 상위 k개
        use_expansion: LLM 쿼리 확장 사용 여부
        query_field : 쿼리로 사용할 필드 ('item_name' 또는 'description')
        verbose     : 케이스별 결과 출력 여부

    Returns:
        {"config": {...}, "per_case": [...], "aggregate": {...}}
    """
    from langchain_community.vectorstores import FAISS
    from app.models import get_embedding_model

    vs_path = ROOT / "vector_store" / "faiss_index"
    if not vs_path.exists():
        raise FileNotFoundError(
            f"벡터 스토어가 없습니다: {vs_path}\n"
            "먼저 'python run_preprocessing.py' 를 실행하세요."
        )

    print(f"벡터 스토어 로드 중: {vs_path}")
    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(
        str(vs_path), embedding_model, allow_dangerous_deserialization=True
    )

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

        # FAISS 검색 (필요 시 max k 보다 크게 요청)
        search_k = max(k, 5)
        docs = vector_store.similarity_search(query, k=search_k)

        # 문서별 소스 분류 및 HS 코드 추출 (None 제거)
        src_types   = [doc_source_type(d) for d in docs]
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
        "config":    {"k": k, "use_expansion": use_expansion, "query_field": query_field},
        "per_case":  per_case,
        "aggregate": agg,
    }


# ── 결과 요약 출력 ─────────────────────────────────────────────

def print_summary(result: Dict[str, Any], ks: List[int] = (1, 3, 5)) -> None:
    cfg = result["config"]
    n   = len(result["per_case"])
    title = (
        f"검색 평가 결과  |  k={cfg['k']}  "
        f"쿼리확장={'O' if cfg['use_expansion'] else 'X'}  "
        f"필드={cfg['query_field']}  |  n={n}"
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

        # PDF가 과도하면 경고
        if pdf_ratio > 0.5:
            print(f"\n  ⚠  PDF 문서 비율이 {pdf_ratio:.0%}으로 높습니다.")
            print("     → run_preprocessing.py 재실행 시 CSV 비중을 늘리는 것을 권장합니다.")

    # 난이도별 분석
    for diff, label in [(1, "쉬움"), (2, "보통"), (3, "어려움")]:
        subset = [c["metrics"] for c in result["per_case"] if c.get("difficulty") == diff]
        if subset:
            sub_agg = aggregate_metrics(subset)
            k_max = cfg["k"]
            h6 = sub_agg.get(f"hit@{k_max}_hs6", 0)
            h4 = sub_agg.get(f"hit@{k_max}_hs4", 0)
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
        choices=["item_name", "description"],
        default="item_name",
        help="쿼리로 사용할 케이스 필드 (기본: item_name)",
    )
    parser.add_argument("--dataset", help="테스트 케이스 JSON 경로")
    parser.add_argument("--output", help="결과 저장 JSON 경로")
    parser.add_argument("--quiet", action="store_true", help="케이스별 출력 생략")
    args = parser.parse_args()

    cases = load_test_cases(args.dataset)
    print(f"\n테스트 케이스 {len(cases)}개 로드")

    result = run_retrieval_eval(
        cases,
        k=args.k,
        use_expansion=args.expand,
        query_field=args.query_field,
        verbose=not args.quiet,
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
