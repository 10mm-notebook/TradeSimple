"""
GPU 임베딩 실험 자동화 스크립트 (overnight 실행용)

실행 (tradesimple 환경 activate 후):
    python run_gpu_experiments.py

순서:
    1. bitsandbytes 설치 (qwen3_4b_int8 INT8 양자화용)
    2. FAISS 인덱스 빌드 (baseline + large 청킹 × 3개 GPU 모델)
    3. 실험 평가 실행 + 결과 저장
"""
import subprocess
import sys
from datetime import datetime


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: str, abort_on_fail: bool = False) -> bool:
    log(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> {cmd}")
    sys.stdout.flush()
    # 자식 프로세스 stdout/stderr를 그대로 터미널에 출력 (버퍼링 없이)
    result = subprocess.run(cmd, shell=True, stdout=None, stderr=None)
    ok = result.returncode == 0
    if not ok:
        log(f"[FAIL] returncode={result.returncode}: {cmd}")
        if abort_on_fail:
            sys.exit(1)
    return ok


log("=" * 60)
log("  GPU 임베딩 실험 시작")
log(f"  시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ── Step 1: bitsandbytes 설치 (qwen3_4b_int8 INT8 양자화용) ──
log("\n[Step 1] bitsandbytes 설치")
run("pip install bitsandbytes --prefer-binary")

# ── Step 2: FAISS 인덱스 빌드 ────────────────────────────────
log("\n[Step 2-A] 인덱스 빌드: pixie_spell, pixie_rune")
run(
    "python -u -m eval.preprocess_experiment "
    "--chunking baseline large "
    "--embedding pixie_spell pixie_rune",
    abort_on_fail=True,
)

log("\n[Step 2-B] 인덱스 빌드: qwen3_4b_int8 (INT8 양자화)")
qwen_ok = run(
    "python -u -m eval.preprocess_experiment "
    "--chunking baseline large "
    "--embedding qwen3_4b_int8"
)

# ── Step 3: 실험 평가 ────────────────────────────────────────
log("\n[Step 3-A] 평가: pixie_spell, pixie_rune")
run(
    "python -u -m eval.run_experiments "
    "--only embed_pixie_spell embed_pixie_rune large_pixie_spell large_pixie_rune "
    "--save",
    abort_on_fail=True,
)

if qwen_ok:
    log("\n[Step 3-B] 평가: qwen3_4b_int8")
    run(
        "python -u -m eval.run_experiments "
        "--only embed_qwen3_4b_int8 large_qwen3_4b_int8 "
        "--save"
    )
else:
    log("\n[Step 3-B] qwen3_4b_int8 인덱스 빌드 실패 → 건너뜀")

# ── 최종 비교 테이블 출력 ─────────────────────────────────────
log("\n[Step 4] 전체 결과 비교")
run("python -u -m eval.run_experiments --compare-only")

log("\n" + "=" * 60)
log(f"  완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("  결과: eval/results/ 폴더 확인")
log("=" * 60)
