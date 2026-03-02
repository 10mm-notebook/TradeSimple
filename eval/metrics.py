"""
eval/metrics.py
HS 코드 RAG 평가 지표 계산 모듈

지표 설명
─────────────────────────────────────────────────────────────────
검색(Retrieval) 지표 — FAISS 검색 결과를 직접 평가
  Hit@k_hs6  : 상위 k개 문서 중 정답 HS 6자리(소호)가 있는 비율
  Hit@k_hs4  : 상위 k개 문서 중 정답 HS 4자리(호)가 있는 비율
  Hit@k_ch2  : 상위 k개 문서 중 정답 HS 2자리(류)가 있는 비율
  MRR_hs6    : Mean Reciprocal Rank (6자리 기준)
  MRR_hs4    : Mean Reciprocal Rank (4자리 기준)

에이전트(Agent) 지표 — run_with_candidates() 출력 평가
  candidate_hit      : 후보 3개 중 정답 HS 6자리 포함 여부
  top1_exact_hs6     : 1순위 후보가 HS 6자리 정답
  top1_exact_hs4     : 1순위 후보가 HS 4자리 정답
  chapter_hit        : 1순위 후보가 HS 2자리(류) 정답
─────────────────────────────────────────────────────────────────
HS 코드 정규화 규칙
  CSV 세번 컬럼은 leading zero가 없음 (예: '303430000')
  → zfill(10)[:n] 으로 앞자리 0 복원 후 앞 n자리 비교
"""

import re
from typing import List, Optional, Dict, Any


# ── 정규화 ─────────────────────────────────────────────────────

def normalize_hs(hs_code: str, digits: int = 6) -> str:
    """
    HS 코드를 지정 자릿수로 정규화 (앞자리 0 포함).

    >>> normalize_hs('303430000', 6)   # CSV 저장값 (leading zero 없음)
    '030343'
    >>> normalize_hs('0303.43-0000', 6)  # 표준 표기
    '030343'
    >>> normalize_hs('030343', 4)
    '0303'
    """
    cleaned = re.sub(r'[.\-\s]', '', str(hs_code))
    return cleaned.zfill(10)[:digits]


def hs_match(code_a: str, code_b: str, digits: int = 6) -> bool:
    """두 HS 코드가 지정 자릿수 기준으로 일치하는지 확인."""
    return normalize_hs(code_a, digits) == normalize_hs(code_b, digits)


# ── 검색 지표 ──────────────────────────────────────────────────

def hit_at_k(
    ranked_codes: List[str],
    expected: str,
    k: int,
    match_digits: int = 6,
) -> bool:
    """상위 k개 ranked_codes 중 expected가 있으면 True."""
    return any(hs_match(c, expected, match_digits) for c in ranked_codes[:k])


def reciprocal_rank(
    ranked_codes: List[str],
    expected: str,
    match_digits: int = 6,
) -> float:
    """첫 번째 정답의 1/순위 반환. 없으면 0.0."""
    for i, code in enumerate(ranked_codes, 1):
        if hs_match(code, expected, match_digits):
            return 1.0 / i
    return 0.0


def compute_retrieval_metrics(
    ranked_codes: List[str],
    expected_hs6: str,
    ks: List[int] = (1, 3, 5),
) -> Dict[str, float]:
    """
    단일 케이스에 대한 검색 지표 딕셔너리 반환.

    Args:
        ranked_codes: FAISS 상위 k개 문서에서 추출한 HS 코드 목록 (순서대로)
        expected_hs6: 정답 HS 6자리 코드
        ks: Hit@k 를 계산할 k 값 목록
    """
    result: Dict[str, float] = {}
    for k in ks:
        result[f"hit@{k}_hs6"] = float(hit_at_k(ranked_codes, expected_hs6, k, 6))
        result[f"hit@{k}_hs4"] = float(hit_at_k(ranked_codes, expected_hs6, k, 4))
        result[f"hit@{k}_ch2"] = float(hit_at_k(ranked_codes, expected_hs6, k, 2))
    result["mrr_hs6"] = reciprocal_rank(ranked_codes, expected_hs6, 6)
    result["mrr_hs4"] = reciprocal_rank(ranked_codes, expected_hs6, 4)
    return result


# ── 에이전트 지표 ──────────────────────────────────────────────

def compute_agent_metrics(
    candidates: List[Dict[str, Any]],
    expected_hs6: str,
) -> Dict[str, float]:
    """
    run_with_candidates() 결과에 대한 에이전트 지표 딕셔너리 반환.

    Args:
        candidates: [{"hs_code": str, "tariff_rate": float, ...}, ...]  (순위순)
        expected_hs6: 정답 HS 6자리 코드
    """
    codes = [str(c.get("hs_code", "")) for c in candidates]
    top1 = codes[0] if codes else ""
    return {
        "candidate_hit":   float(any(hs_match(c, expected_hs6, 6) for c in codes)),
        "top1_exact_hs6":  float(hs_match(top1, expected_hs6, 6)),
        "top1_exact_hs4":  float(hs_match(top1, expected_hs6, 4)),
        "chapter_hit":     float(hs_match(top1, expected_hs6, 2)),
    }


# ── 집계 ───────────────────────────────────────────────────────

def aggregate_metrics(per_case: List[Dict[str, float]]) -> Dict[str, float]:
    """케이스별 지표 목록을 평균내어 반환."""
    if not per_case:
        return {}
    keys = per_case[0].keys()
    n = len(per_case)
    return {k: sum(m[k] for m in per_case) / n for k in keys}


def format_metrics_table(
    agg: Dict[str, float],
    ks: List[int] = (1, 3, 5),
    title: str = "",
) -> str:
    """집계 지표를 보기 좋은 테이블 문자열로 반환."""
    lines = []
    if title:
        lines.append(f"\n{'─'*56}")
        lines.append(f"  {title}")
    lines.append(f"{'─'*56}")
    lines.append(f"  {'지표':<22} {'HS 6자리':>9} {'HS 4자리':>9} {'2자리(류)':>9}")
    lines.append(f"  {'─'*50}")
    for k in ks:
        h6 = agg.get(f"hit@{k}_hs6", 0)
        h4 = agg.get(f"hit@{k}_hs4", 0)
        h2 = agg.get(f"hit@{k}_ch2", 0)
        lines.append(f"  {'Hit@'+str(k):<22} {h6:>8.1%} {h4:>9.1%} {h2:>9.1%}")
    lines.append(f"  {'─'*50}")
    mrr6 = agg.get("mrr_hs6", 0)
    mrr4 = agg.get("mrr_hs4", 0)
    lines.append(f"  {'MRR':<22} {mrr6:>8.4f} {mrr4:>9.4f}")

    # 에이전트 지표가 있으면 추가 출력
    if "candidate_hit" in agg:
        lines.append(f"  {'─'*50}")
        lines.append(f"  {'후보 포함(Candidate Hit)':<22} {agg['candidate_hit']:>8.1%}")
        lines.append(f"  {'1위 HS6 정확도':<22} {agg['top1_exact_hs6']:>8.1%}")
        lines.append(f"  {'1위 HS4 정확도':<22} {agg['top1_exact_hs4']:>8.1%}")
        lines.append(f"  {'1위 류(Chapter) 정확도':<22} {agg['chapter_hit']:>8.1%}")
    lines.append(f"{'─'*56}")
    return "\n".join(lines)
