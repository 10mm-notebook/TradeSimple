"""
eval/run_experiments.py
RAG 전략 ablation 실험 자동 실행 및 비교 테이블 출력

실험 설정 (EXPERIMENTS 목록에서 자유롭게 추가/수정)
──────────────────────────────────────────────────────
  baseline        k=5, 쿼리 확장 없음, item_name 필드
  high_k          k=10, 쿼리 확장 없음, item_name 필드
  low_k           k=3, 쿼리 확장 없음, item_name 필드
  with_expansion  k=5, 쿼리 확장(LLM), item_name 필드
  desc_field      k=5, 쿼리 확장 없음, description 필드
  desc_expansion  k=5, 쿼리 확장(LLM), description 필드

사용법
──────
  # 모든 실험 실행
  python -m eval.run_experiments

  # 특정 실험만 실행
  python -m eval.run_experiments --only baseline high_k

  # 결과를 eval/results/ 에 자동 저장
  python -m eval.run_experiments --save

  # 이미 저장된 JSON 결과만 불러와 비교 테이블 출력
  python -m eval.run_experiments --compare-only

  # 실험을 건너뛰고 단순 비교
  python -m eval.run_experiments --compare-only --save-dir eval/results
"""

import sys
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.evaluate_retrieval import load_test_cases, run_retrieval_eval
from eval.metrics import aggregate_metrics, normalize_hs


# ── 실험 설정 ──────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    k: int = 5
    use_expansion: bool = False
    query_field: str = "item_name"
    description: str = ""

    @property
    def output_filename(self) -> str:
        return f"{self.name}.json"


EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="baseline",
        k=5, use_expansion=False, query_field="item_name",
        description="k=5, 확장 없음, item_name",
    ),
    ExperimentConfig(
        name="low_k",
        k=3, use_expansion=False, query_field="item_name",
        description="k=3, 확장 없음, item_name",
    ),
    ExperimentConfig(
        name="high_k",
        k=10, use_expansion=False, query_field="item_name",
        description="k=10, 확장 없음, item_name",
    ),
    ExperimentConfig(
        name="with_expansion",
        k=5, use_expansion=True, query_field="item_name",
        description="k=5, LLM 쿼리 확장, item_name",
    ),
    ExperimentConfig(
        name="desc_field",
        k=5, use_expansion=False, query_field="description",
        description="k=5, 확장 없음, description",
    ),
    ExperimentConfig(
        name="desc_expansion",
        k=5, use_expansion=True, query_field="description",
        description="k=5, LLM 쿼리 확장, description",
    ),
]


# ── 단일 실험 실행 ─────────────────────────────────────────────

def run_one(
    cfg: ExperimentConfig,
    cases: List[Dict[str, Any]],
    save_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """실험 1개 실행 후 결과 반환 (필요 시 JSON 저장)."""
    print(f"\n▶ 실험: {cfg.name}  ({cfg.description})")
    result = run_retrieval_eval(
        cases,
        k=cfg.k,
        use_expansion=cfg.use_expansion,
        query_field=cfg.query_field,
        verbose=verbose,
    )
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / cfg.output_filename
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"   저장: {out}")
    return result


# ── 비교 테이블 출력 ───────────────────────────────────────────

_METRIC_COLS = [
    ("hit@1_hs6",  "Hit@1 HS6"),
    ("hit@3_hs6",  "Hit@3 HS6"),
    ("hit@5_hs6",  "Hit@5 HS6"),
    ("hit@10_hs6", "Hit@10 HS6"),
    ("mrr_hs6",    "MRR HS6"),
    ("hit@1_hs4",  "Hit@1 HS4"),
    ("hit@3_hs4",  "Hit@3 HS4"),
    ("hit@5_hs4",  "Hit@5 HS4"),
    ("mrr_hs4",    "MRR HS4"),
]


def print_comparison_table(
    results: List[Dict[str, Any]],
    configs: List[ExperimentConfig],
) -> None:
    """실험별 핵심 지표를 한 줄로 비교하는 테이블 출력."""
    assert len(results) == len(configs)

    # 각 실험에서 실제로 존재하는 지표 컬럼만 사용
    present_keys: set = set()
    for r in results:
        present_keys.update(r["aggregate"].keys())
    cols = [(k, label) for k, label in _METRIC_COLS if k in present_keys]

    # 헤더
    name_w = max(len(c.name) for c in configs) + 2
    col_w  = 9
    header = f"  {'실험명':<{name_w}}" + "".join(f"{label:>{col_w}}" for _, label in cols)
    sep    = "  " + "─" * (name_w + col_w * len(cols))

    print(f"\n{'='*60}")
    print("  실험 결과 비교 테이블 (검색 정확도)")
    print(f"{'='*60}")
    print(header)
    print(sep)

    best = {k: max(r["aggregate"].get(k, 0) for r in results) for k, _ in cols}

    for cfg, result in zip(configs, results):
        agg  = result["aggregate"]
        row  = f"  {cfg.name:<{name_w}}"
        for k, _ in cols:
            val = agg.get(k, 0)
            s   = f"{val:.1%}" if "hit" in k else f"{val:.4f}"
            # 최고값 강조 (*표시)
            marker = "*" if abs(val - best[k]) < 1e-9 and best[k] > 0 else " "
            row += f"{marker+s:>{col_w}}"
        print(row)

    print(sep)
    print("  * = 해당 지표 최고값\n")

    # 종합 순위 (Hit@5_hs6 + MRR_hs6 기준)
    def score(r: Dict) -> float:
        a = r["aggregate"]
        return a.get("hit@5_hs6", 0) * 0.7 + a.get("mrr_hs6", 0) * 0.3

    ranked = sorted(zip(configs, results), key=lambda x: score(x[1]), reverse=True)
    print("  종합 순위 (Hit@5_hs6×0.7 + MRR_hs6×0.3):")
    for rank, (cfg, res) in enumerate(ranked, 1):
        s = score(res)
        print(f"    {rank}위: {cfg.name:<20}  score={s:.4f}  ({cfg.description})")


# ── CLI ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 전략 ablation 실험")
    parser.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="실행할 실험 이름 (예: baseline high_k)",
    )
    parser.add_argument("--save", action="store_true", help="결과를 JSON 파일로 저장")
    parser.add_argument(
        "--save-dir", default="eval/results",
        help="결과 저장 디렉터리 (기본: eval/results)",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="저장된 JSON 결과만 로드하여 비교 테이블 출력 (실험 미실행)",
    )
    parser.add_argument("--dataset", help="테스트 케이스 JSON 경로")
    parser.add_argument("--verbose", action="store_true", help="케이스별 결과 출력")
    args = parser.parse_args()

    save_dir = ROOT / args.save_dir

    # 실험 목록 필터
    chosen_exps = (
        [e for e in EXPERIMENTS if e.name in args.only]
        if args.only
        else EXPERIMENTS
    )

    # ── compare-only 모드 ──────────────────────────────────────
    if args.compare_only:
        results = []
        valid_cfgs = []
        for cfg in chosen_exps:
            fp = save_dir / cfg.output_filename
            if fp.exists():
                with fp.open(encoding="utf-8") as f:
                    results.append(json.load(f))
                valid_cfgs.append(cfg)
                print(f"  로드: {fp}")
            else:
                print(f"  건너뜀 (파일 없음): {fp}")
        if results:
            print_comparison_table(results, valid_cfgs)
        else:
            print("비교할 결과 파일이 없습니다. 먼저 --save 플래그로 실험을 실행하세요.")
        return

    # ── 실험 실행 모드 ─────────────────────────────────────────
    cases = load_test_cases(args.dataset)
    print(f"\n테스트 케이스 {len(cases)}개 로드")
    print(f"실험 {len(chosen_exps)}개 실행 예정: {[e.name for e in chosen_exps]}")

    results = []
    for cfg in chosen_exps:
        result = run_one(
            cfg,
            cases,
            save_dir=save_dir if args.save else None,
            verbose=args.verbose,
        )
        results.append(result)

    if len(results) > 1:
        print_comparison_table(results, chosen_exps)
    else:
        # 단일 실험이면 요약만 출력
        from eval.evaluate_retrieval import print_summary
        print_summary(results[0])


if __name__ == "__main__":
    main()
