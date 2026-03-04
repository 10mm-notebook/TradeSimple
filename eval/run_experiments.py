"""
eval/run_experiments.py
RAG 전략 ablation 실험 자동 실행 및 비교 테이블 출력

실험 설정 (EXPERIMENTS 목록에서 자유롭게 추가/수정)
──────────────────────────────────────────────────────
  [기본 쿼리 전략]
  baseline          k=5, 쿼리 확장 없음, item_name 필드
  high_k            k=10, 쿼리 확장 없음, item_name 필드
  low_k             k=3, 쿼리 확장 없음, item_name 필드
  with_expansion    k=5, 쿼리 확장(LLM), item_name 필드
  desc_field        k=5, 쿼리 확장 없음, description 필드
  desc_expansion    k=5, 쿼리 확장(LLM), description 필드
  detailed_desc     k=5, 쿼리 확장 없음, detailed_description 필드
  detailed_expansion k=5, 쿼리 확장(LLM), detailed_description 필드

  [청킹 전략 — 사전에 eval.preprocess_experiment 로 인덱스 빌드 필요]
  chunk_small       RecursiveChar 500자, baseline 임베딩
  chunk_large       RecursiveChar 2000자, baseline 임베딩
  chunk_sliding     RecursiveChar 300자 50% overlap, baseline 임베딩
  chunk_token       TokenTextSplitter 256토큰, baseline 임베딩
  chunk_paragraph   단락 우선 분리, baseline 임베딩
  chunk_page        PDF 페이지 단위, baseline 임베딩

  [임베딩 모델 — 사전에 eval.preprocess_experiment 로 인덱스 빌드 필요]
  embed_multilingual_e5   baseline 청킹, multilingual-e5-large-instruct
  embed_kure_v1           baseline 청킹, KURE-v1
  embed_snowflake_ko      baseline 청킹, snowflake-arctic-embed-l-v2.0-ko

  [검색 방법 — 기본 인덱스 사용, 알고리즘만 변경]
  hybrid_baseline   BM25+dense hybrid (RRF), baseline 인덱스
  rerank_baseline   CrossEncoder 리랭킹, baseline 인덱스
  hybrid_rerank     BM25+dense hybrid + CrossEncoder 리랭킹

사용법
──────
  # 모든 실험 실행
  python -m eval.run_experiments

  # 특정 실험만 실행
  python -m eval.run_experiments --only baseline chunk_small embed_multilingual_e5

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
    # 인덱스 / 임베딩 / 검색 방법 선택
    vs_name: str = "default"          # "default" or "{chunking}_{embedding}"
    embedding_name: str = "baseline"  # EMBEDDING_REGISTRY 키
    hybrid: bool = False
    rerank: bool = False
    rerank_top_n: int = 10
    # balanced 듀얼 인덱스 검색
    pdf_quota: int = 0        # 0 = 기존 방식, 1+ = PDF-only 인덱스에서 강제 포함 수
    pdf_vs_name: str = ""     # PDF-only 인덱스 이름 (exp_{pdf_vs_name}_pdf/faiss_index)

    @property
    def output_filename(self) -> str:
        return f"{self.name}.json"

    def resolve_vs_path(self) -> Optional[Path]:
        """
        vs_name으로부터 실제 벡터 스토어 경로를 반환.
        "default" → None (run_retrieval_eval 기본값 사용)
        그 외   → ROOT/vector_store/exp_{vs_name}/faiss_index
        """
        if self.vs_name == "default":
            return None
        return ROOT / "vector_store" / f"exp_{self.vs_name}" / "faiss_index"

    def resolve_pdf_vs_path(self) -> Optional[Path]:
        """PDF-only 인덱스 경로. pdf_vs_name 없으면 None."""
        if not self.pdf_vs_name:
            return None
        return ROOT / "vector_store" / f"exp_{self.pdf_vs_name}_pdf" / "faiss_index"


# ── 실험 목록 ──────────────────────────────────────────────────

EXPERIMENTS: List[ExperimentConfig] = [

    # ── 기본 쿼리 전략 ─────────────────────────────────────────
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
    ExperimentConfig(
        name="detailed_desc",
        k=5, use_expansion=False, query_field="detailed_description",
        description="k=5, 확장 없음, detailed_description (원재료·가공·형태·소재)",
    ),
    ExperimentConfig(
        name="detailed_expansion",
        k=5, use_expansion=True, query_field="detailed_description",
        description="k=5, LLM 쿼리 확장, detailed_description",
    ),

    # ── 청킹 전략 (baseline 임베딩 고정, PDF 청킹만 변경) ──────
    ExperimentConfig(
        name="chunk_small",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=small(500자), baseline 임베딩",
        vs_name="small_baseline", embedding_name="baseline",
    ),
    ExperimentConfig(
        name="chunk_large",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=large(2000자), baseline 임베딩",
        vs_name="large_baseline", embedding_name="baseline",
    ),
    ExperimentConfig(
        name="chunk_sliding",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=sliding_300(300자 50% overlap), baseline 임베딩",
        vs_name="sliding_300_baseline", embedding_name="baseline",
    ),
    ExperimentConfig(
        name="chunk_token",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=token_256(256토큰), baseline 임베딩",
        vs_name="token_256_baseline", embedding_name="baseline",
    ),
    ExperimentConfig(
        name="chunk_paragraph",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=paragraph(단락 우선), baseline 임베딩",
        vs_name="paragraph_baseline", embedding_name="baseline",
    ),
    ExperimentConfig(
        name="chunk_page",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=page(PDF 페이지 단위), baseline 임베딩",
        vs_name="page_baseline", embedding_name="baseline",
    ),

    # ── 임베딩 모델 (baseline 청킹 고정, 임베딩만 변경) ─────────
    ExperimentConfig(
        name="embed_multilingual_e5",
        k=5, use_expansion=False, query_field="item_name",
        description="baseline 청킹, multilingual-e5-large-instruct",
        vs_name="baseline_multilingual_e5", embedding_name="multilingual_e5",
    ),
    ExperimentConfig(
        name="embed_kure_v1",
        k=5, use_expansion=False, query_field="item_name",
        description="baseline 청킹, KURE-v1",
        vs_name="baseline_kure_v1", embedding_name="kure_v1",
    ),
    ExperimentConfig(
        name="embed_snowflake_ko",
        k=5, use_expansion=False, query_field="item_name",
        description="baseline 청킹, snowflake-arctic-embed-l-v2.0-ko",
        vs_name="baseline_snowflake_ko", embedding_name="snowflake_ko",
    ),
    ExperimentConfig(
        name="embed_pixie_spell",
        k=5, use_expansion=False, query_field="item_name",
        description="baseline 청킹, PIXIE-Spell-1.7B-fp16 (GPU)",
        vs_name="baseline_pixie_spell", embedding_name="pixie_spell",
    ),

    # ── 조합 실험 (청킹 + 임베딩 교차) ──────────────────────────
    ExperimentConfig(
        name="large_snowflake_ko",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=large(2000자) + snowflake_ko",
        vs_name="large_snowflake_ko", embedding_name="snowflake_ko",
    ),
    ExperimentConfig(
        name="large_kure_v1",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=large(2000자) + KURE-v1",
        vs_name="large_kure_v1", embedding_name="kure_v1",
    ),
    ExperimentConfig(
        name="page_snowflake_ko",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=page(페이지단위) + snowflake_ko",
        vs_name="page_snowflake_ko", embedding_name="snowflake_ko",
    ),
    ExperimentConfig(
        name="page_kure_v1",
        k=5, use_expansion=False, query_field="item_name",
        description="청킹=page(페이지단위) + KURE-v1",
        vs_name="page_kure_v1", embedding_name="kure_v1",
    ),

    # ── Balanced 듀얼 인덱스 (통합 인덱스 + PDF-only 인덱스) ──────
    # 사전 빌드 필요:
    #   python -m eval.preprocess_experiment --chunking large --embedding kure_v1 --pdf-only
    ExperimentConfig(
        name="balanced_1pdf_large_kure",
        k=5, use_expansion=False, query_field="item_name",
        description="balanced PDF=1 + CSV=4 (large+kure_v1 인덱스)",
        vs_name="large_kure_v1", embedding_name="kure_v1",
        pdf_quota=1, pdf_vs_name="large_kure_v1",
    ),
    ExperimentConfig(
        name="balanced_2pdf_large_kure",
        k=5, use_expansion=False, query_field="item_name",
        description="balanced PDF=2 + CSV=3 (large+kure_v1 인덱스)",
        vs_name="large_kure_v1", embedding_name="kure_v1",
        pdf_quota=2, pdf_vs_name="large_kure_v1",
    ),
    ExperimentConfig(
        name="balanced_3pdf_large_kure",
        k=5, use_expansion=False, query_field="item_name",
        description="balanced PDF=3 + CSV=2 (large+kure_v1 인덱스)",
        vs_name="large_kure_v1", embedding_name="kure_v1",
        pdf_quota=3, pdf_vs_name="large_kure_v1",
    ),
    ExperimentConfig(
        name="balanced_4pdf_large_kure",
        k=5, use_expansion=False, query_field="item_name",
        description="balanced PDF=4 + CSV=1 (large+kure_v1 인덱스)",
        vs_name="large_kure_v1", embedding_name="kure_v1",
        pdf_quota=4, pdf_vs_name="large_kure_v1",
    ),
    ExperimentConfig(
        name="balanced_5pdf_large_kure",
        k=5, use_expansion=False, query_field="item_name",
        description="balanced PDF=5 + CSV=0 (PDF-only 검색)",
        vs_name="large_kure_v1", embedding_name="kure_v1",
        pdf_quota=5, pdf_vs_name="large_kure_v1",
    ),

    # ── 검색 방법 (baseline_baseline 인덱스 사용 — docs.json 포함) ──
    ExperimentConfig(
        name="hybrid_baseline",
        k=5, use_expansion=False, query_field="item_name",
        description="BM25+dense hybrid(RRF), baseline 인덱스",
        vs_name="baseline_baseline", embedding_name="baseline",
        hybrid=True, rerank=False,
    ),
    ExperimentConfig(
        name="rerank_baseline",
        k=5, use_expansion=False, query_field="item_name",
        description="CrossEncoder 리랭킹, baseline 인덱스",
        vs_name="baseline_baseline", embedding_name="baseline",
        hybrid=False, rerank=True, rerank_top_n=10,
    ),
    ExperimentConfig(
        name="hybrid_rerank",
        k=5, use_expansion=False, query_field="item_name",
        description="BM25+dense hybrid + CrossEncoder 리랭킹",
        vs_name="baseline_baseline", embedding_name="baseline",
        hybrid=True, rerank=True, rerank_top_n=10,
    ),
]


# ── 단일 실험 실행 ─────────────────────────────────────────────

def run_one(
    cfg: ExperimentConfig,
    cases: List[Dict[str, Any]],
    save_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """실험 1개 실행 후 결과 반환 (필요 시 JSON 저장). 인덱스 없으면 None 반환."""
    resolved_vs = cfg.resolve_vs_path()

    # 인덱스 존재 여부 사전 확인 (non-default 실험)
    if resolved_vs is not None and not resolved_vs.exists():
        print(
            f"\n  ⏭ 건너뜀: {cfg.name}  (인덱스 없음: {resolved_vs})\n"
            f"     → python -m eval.preprocess_experiment --chunking ... --embedding ... 로 먼저 빌드하세요."
        )
        return None

    # balanced 모드: PDF-only 인덱스 경로 확인
    resolved_pdf_vs = cfg.resolve_pdf_vs_path()
    if cfg.pdf_quota > 0 and resolved_pdf_vs is not None and not resolved_pdf_vs.exists():
        print(
            f"\n  ⏭ 건너뜀: {cfg.name}  (PDF-only 인덱스 없음: {resolved_pdf_vs})\n"
            f"     → python -m eval.preprocess_experiment --chunking ... --embedding ... --pdf-only 로 빌드하세요."
        )
        return None

    print(f"\n▶ 실험: {cfg.name}  ({cfg.description})")
    result = run_retrieval_eval(
        cases,
        k=cfg.k,
        use_expansion=cfg.use_expansion,
        query_field=cfg.query_field,
        verbose=verbose,
        vs_path=resolved_vs,
        embedding_name=cfg.embedding_name,
        hybrid=cfg.hybrid,
        rerank=cfg.rerank,
        rerank_top_n=cfg.rerank_top_n,
        pdf_quota=cfg.pdf_quota,
        pdf_vs_path=resolved_pdf_vs,
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
        print(f"    {rank}위: {cfg.name:<25}  score={s:.4f}  ({cfg.description})")


# ── CLI ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 전략 ablation 실험")
    parser.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="실행할 실험 이름 (예: baseline chunk_small embed_multilingual_e5)",
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
    valid_cfgs = []
    for cfg in chosen_exps:
        result = run_one(
            cfg,
            cases,
            save_dir=save_dir if args.save else None,
            verbose=args.verbose,
        )
        if result is not None:
            results.append(result)
            valid_cfgs.append(cfg)

    if len(results) > 1:
        print_comparison_table(results, valid_cfgs)
    elif len(results) == 1:
        from eval.evaluate_retrieval import print_summary
        print_summary(results[0])
    else:
        print("\n실행된 실험이 없습니다. 인덱스를 먼저 빌드하세요.")


if __name__ == "__main__":
    main()
