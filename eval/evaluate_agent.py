"""
eval/evaluate_agent.py
HSCodeFinderAgent 엔드-투-엔드 정확도 평가 (LLM 호출 발생 — 비용 주의)

지표
────
  candidate_hit      : 후보 3개 중 정답 HS 6자리 포함 여부 (HITL 전 정확도)
  top1_exact_hs6     : 1순위 후보 HS 6자리 일치
  top1_exact_hs4     : 1순위 후보 HS 4자리 일치
  chapter_hit        : 1순위 후보 HS 2자리(류) 일치

사용법
──────
  # 전체 35케이스 실행 (비용 주의: GPT-4o 호출 35회)
  python -m eval.evaluate_agent

  # 10케이스만 실행
  python -m eval.evaluate_agent --limit 10

  # 비용 예측만 출력 (dry-run)
  python -m eval.evaluate_agent --dry-run

  # 결과 저장
  python -m eval.evaluate_agent --limit 10 --output eval/results/agent_eval.json
"""

import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import compute_agent_metrics, aggregate_metrics, format_metrics_table, normalize_hs


# ── 데이터 로드 ────────────────────────────────────────────────

def load_test_cases(path: Optional[str] = None) -> List[Dict[str, Any]]:
    if path is None:
        path = ROOT / "eval" / "dataset" / "test_cases.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 단일 케이스 평가 ───────────────────────────────────────────

async def evaluate_one(
    case: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    케이스 1개에 대해 run_with_candidates() 를 실행하고 지표를 반환.
    에이전트 호출 카운터를 매 케이스마다 리셋한다.
    """
    from app.agents.hs_code_finder import HSCodeFinderAgent
    from app.tools import reset_hs_code_search_limit

    item_name = case["item_name"]
    expected  = case["expected_hs6"]

    # 에이전트 호출 카운터 리셋 (케이스 간 독립성 보장)
    reset_hs_code_search_limit()

    t0 = time.monotonic()
    try:
        agent = HSCodeFinderAgent()
        result = await agent.run_with_candidates(item_name)
        candidates = result.get("candidates", [])
        elapsed = time.monotonic() - t0
        error = None
    except Exception as e:
        candidates = []
        elapsed = time.monotonic() - t0
        error = str(e)

    metrics = compute_agent_metrics(candidates, expected)
    top1_code = normalize_hs(candidates[0].get("hs_code", "")) if candidates else "—"

    if verbose:
        cand_hit = bool(metrics["candidate_hit"])
        top1_ok  = bool(metrics["top1_exact_hs6"])
        icon = "✅" if top1_ok else ("🟡" if cand_hit else "❌")
        print(
            f"  {icon} [{case['id']:02d}] {item_name:<15} "
            f"exp={normalize_hs(expected)}  top1={top1_code}  "
            f"({elapsed:.1f}s)"
            + (f"  ERR: {error}" if error else "")
        )

    return {
        "id":         case["id"],
        "item_name":  item_name,
        "category":   case.get("category", ""),
        "difficulty": case.get("difficulty", 1),
        "expected_hs6": expected,
        "candidates": [
            {"hs_code": normalize_hs(c.get("hs_code", "")), "품명": c.get("품명", "")}
            for c in candidates
        ],
        "metrics":    metrics,
        "elapsed_s":  round(elapsed, 2),
        "error":      error,
    }


# ── 전체 평가 실행 ─────────────────────────────────────────────

async def run_agent_eval(
    test_cases: List[Dict[str, Any]],
    limit: Optional[int] = None,
    verbose: bool = True,
    delay_s: float = 1.0,
) -> Dict[str, Any]:
    """
    전체 케이스에 대한 에이전트 평가.

    Args:
        delay_s: 케이스 간 대기 시간 (API rate limit 방지)
    """
    if limit:
        test_cases = test_cases[:limit]

    per_case: List[Dict[str, Any]] = []
    for i, case in enumerate(test_cases):
        res = await evaluate_one(case, verbose=verbose)
        per_case.append(res)
        if i < len(test_cases) - 1 and delay_s > 0:
            await asyncio.sleep(delay_s)

    agg = aggregate_metrics([c["metrics"] for c in per_case])
    return {
        "config":    {"limit": limit, "n_evaluated": len(per_case)},
        "per_case":  per_case,
        "aggregate": agg,
    }


# ── 요약 출력 ──────────────────────────────────────────────────

def print_summary(result: Dict[str, Any]) -> None:
    n = result["config"]["n_evaluated"]
    title = f"에이전트 평가 결과  |  n={n}"

    # 에이전트 전용 지표 출력 (Hit@k 없음)
    agg = result["aggregate"]
    print(f"\n{'─'*56}")
    print(f"  {title}")
    print(f"{'─'*56}")
    print(f"  {'지표':<28} {'값':>8}")
    print(f"  {'─'*36}")
    print(f"  {'후보 포함 (Candidate Hit)':<28} {agg.get('candidate_hit',0):>7.1%}")
    print(f"  {'1위 HS6 정확도 (Top-1 Exact)':<28} {agg.get('top1_exact_hs6',0):>7.1%}")
    print(f"  {'1위 HS4 정확도':<28} {agg.get('top1_exact_hs4',0):>7.1%}")
    print(f"  {'1위 류(Chapter) 정확도':<28} {agg.get('chapter_hit',0):>7.1%}")
    print(f"{'─'*56}")

    # 난이도별 분석
    for diff, label in [(1, "쉬움"), (2, "보통"), (3, "어려움")]:
        subset = [c["metrics"] for c in result["per_case"] if c.get("difficulty") == diff]
        if subset:
            sub_agg = aggregate_metrics(subset)
            c_hit  = sub_agg.get("candidate_hit", 0)
            t1_hs6 = sub_agg.get("top1_exact_hs6", 0)
            print(
                f"  난이도 {diff}({label}, n={len(subset):2d}):  "
                f"Candidate Hit={c_hit:.0%}  Top-1 HS6={t1_hs6:.0%}"
            )

    # 오류 케이스
    errors = [c for c in result["per_case"] if c.get("error")]
    if errors:
        print(f"\n  ⚠ 오류 {len(errors)}건:")
        for e in errors:
            print(f"    [{e['id']:02d}] {e['item_name']}: {e['error']}")


# ── CLI ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HSCodeFinderAgent 정확도 평가 (LLM 호출 발생)"
    )
    parser.add_argument("--limit", type=int, default=None, help="평가할 케이스 수 제한")
    parser.add_argument("--dataset", help="테스트 케이스 JSON 경로")
    parser.add_argument("--output", help="결과 저장 JSON 경로")
    parser.add_argument("--quiet", action="store_true", help="케이스별 출력 생략")
    parser.add_argument("--delay", type=float, default=1.0, help="케이스 간 대기 시간(초, 기본 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="비용 예측만 출력 후 종료")
    args = parser.parse_args()

    cases = load_test_cases(args.dataset)
    n = args.limit or len(cases)

    print(f"\n테스트 케이스 {len(cases)}개 중 {n}개 평가 예정")
    print(f"⚠  LLM API 호출 발생 (케이스당 GPT-4o ~2-4회 호출)")

    if args.dry_run:
        print(f"\n[Dry-run] 실제 실행 안함. --dry-run 플래그 제거 후 재실행하세요.")
        return

    result = asyncio.run(
        run_agent_eval(
            cases,
            limit=args.limit,
            verbose=not args.quiet,
            delay_s=args.delay,
        )
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
