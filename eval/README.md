# eval/ — RAG 정확도 평가 파이프라인

TradeSimple의 FAISS 벡터 검색 정확도를 체계적으로 측정하고 개선하기 위한 평가 인프라.
**LLM 호출 없이** 순수 검색 정확도만 평가한다 (무료·빠름).

---

## 목차

1. [디렉터리 구조](#1-디렉터리-구조)
2. [평가 지표](#2-평가-지표)
3. [테스트 데이터셋](#3-테스트-데이터셋)
4. [빠른 시작](#4-빠른-시작)
5. [실험 로드맵](#5-실험-로드맵)
6. [Phase 1 — 청킹 전략 탐색](#6-phase-1--청킹-전략-탐색)
7. [Phase 2 — 임베딩 모델 탐색](#7-phase-2--임베딩-모델-탐색)
8. [Phase 3 — 청킹 × 임베딩 최적 조합](#8-phase-3--청킹--임베딩-최적-조합)
9. [Phase 4 — 검색 후처리 (BM25 / Rerank)](#9-phase-4--검색-후처리)
10. [Phase 5 — 듀얼 인덱스 균형 검색](#10-phase-5--듀얼-인덱스-균형-검색)
11. [추가 실험 — GPU 임베딩 모델](#11-추가-실험--gpu-임베딩-모델)
12. [기술 노트 — PDF HS 코드 메타데이터 주입](#12-기술-노트--pdf-hs-코드-메타데이터-주입)
13. [실험 실행 가이드](#13-실험-실행-가이드)
14. [결론 및 최적 설정](#14-결론-및-최적-설정)

---

## 1. 디렉터리 구조

```
eval/
├── README.md                   # 이 파일
├── __init__.py
│
├── metrics.py                  # HS 코드 정규화 + 지표 계산 (Hit@k, MRR)
├── evaluate_retrieval.py       # FAISS 검색 평가 실행기 (메인 평가 모듈)
├── evaluate_agent.py           # 에이전트 전체 파이프라인 평가 (LLM 호출)
├── run_experiments.py          # 복수 실험 일괄 실행 + 비교 테이블 출력
│
├── chunking_strategies.py      # 청킹 전략 레지스트리 (7가지)
├── embedding_registry.py       # 임베딩 모델 레지스트리 (9가지)
├── preprocess_experiment.py    # (청킹 × 임베딩) 조합별 FAISS 인덱스 빌더
│
├── dataset/                    # ← .gitignore 제외
│   └── test_cases.json         # 35개 테스트 케이스
└── results/                    # ← .gitignore 제외 (재현 가능)
    ├── baseline.json
    ├── large_kure_v1.json
    ├── balanced_3pdf_large_kure.json
    └── ...
```

---

## 2. 평가 지표

### 검색(Retrieval) 지표

FAISS가 반환한 상위 k개 문서에서 HS 코드를 추출해 정답과 비교한다.

| 지표 | 설명 |
|------|------|
| **Hit@k HS6** | 상위 k개 중 정답 **HS 6자리(소호)** 포함 비율 |
| **Hit@k HS4** | 상위 k개 중 정답 **HS 4자리(호)** 포함 비율 |
| **Hit@k CH2** | 상위 k개 중 정답 **HS 2자리(류/챕터)** 포함 비율 |
| **MRR HS6** | Mean Reciprocal Rank — 정답이 처음 등장한 순위의 역수 평균 |
| **MRR HS4** | 4자리 기준 MRR |

### HS 코드 정규화 규칙

```
입력 형식          →  정규화(6자리 기준)
──────────────────────────────────────────
"303430000" (CSV)  →  "030343"   zfill(10)[:6]
"0306.17-1090"     →  "030617"   구분자 제거 후 zfill(10)[:6]
"030343"           →  "030343"   그대로
"0303"   (4자리)   →  "030300"   오른쪽 0 패딩
```

### 기준 지표

실험 비교의 주 지표는 **Hit@5 HS6** (상위 5개 문서 중 정답 소호 포함 비율).
보조 지표로 **MRR HS6** (랭킹 품질)과 **Hit@5 HS4** (소호 오분류 허용)를 함께 본다.

---

## 3. 테스트 데이터셋

**파일**: `eval/dataset/test_cases.json`
**규모**: 35개 케이스, 18류(챕터) 포함

### 케이스 구조

```json
{
  "id": 1,
  "item_name": "냉동 새우",
  "description": "냉동 처리된 양식 흰다리새우",
  "detailed_description": "냉동한 흰다리새우(Litopenaeus vannamei), 양식산, 껍질 있음",
  "expected_hs6": "030617",
  "category": "수산물",
  "difficulty": 2
}
```

| 필드 | 설명 |
|------|------|
| `item_name` | 단순 품명 (기본 쿼리) |
| `description` | 간단한 설명 포함 쿼리 |
| `detailed_description` | 상세 규격 포함 쿼리 |
| `expected_hs6` | 정답 HS 6자리 |
| `difficulty` | 난이도 1(쉬움)~3(어려움) |

---

## 4. 빠른 시작

```bash
# 기본 평가 (k=5, 기본 인덱스)
python -m eval.evaluate_retrieval

# 특정 실험 인덱스 평가
python -m eval.evaluate_retrieval \
  --vs-path vector_store/exp_large_kure_v1/faiss_index \
  --embedding kure_v1

# 모든 실험 일괄 실행 + 비교 테이블
python -m eval.run_experiments

# 특정 실험만 선택 + 결과 저장
python -m eval.run_experiments --only baseline large_kure_v1 balanced_3pdf_large_kure --save
```

---

## 5. 실험 로드맵

한 번에 하나의 변수만 변경하는 **단계별 ablation** 방식으로 진행했다.
각 Phase에서 최적을 확정한 뒤, 다음 Phase에서 그 결과를 기준으로 삼는다.

```
[베이스라인]  Hit@5 HS6 = 37.1%
  └─ ko-sroberta / 1000자 청킹 / 단일 인덱스

  Phase 1: 청킹 전략 탐색  (임베딩=baseline 고정)
  └─ large(2000자) 선택 → 42.9%  (+5.8%p)

  Phase 2: 임베딩 모델 탐색  (청킹=baseline 고정)
  └─ KURE-v1 선택 → 45.7%  (+8.6%p)

  Phase 3: 최적 청킹 × 최적 임베딩 조합
  └─ large × KURE-v1 → 54.3%  (시너지 +8.6%p 추가)

  Phase 4: 검색 후처리 실험  (BM25 hybrid / CrossEncoder rerank)
  └─ Hit@5 변화 없음 → 알고리즘보다 데이터 불균형 문제 확인

  Phase 5: 듀얼 인덱스 균형 검색  (PDF 할당량 조정)
  └─ PDF 쿼터=3 → 62.9%  (+8.6%p) ★ 최종 채택

[최종]  Hit@5 HS6 = 62.9%  (베이스라인 대비 +25.8%p, 1.7배)
```

> **추가**: GPU 임베딩 모델(pixie_spell, qwen3_4b_int8)도 실험했으나
> 듀얼 인덱스 적용 후에도 최대 60.0%로 CPU KURE-v1(62.9%)을 넘지 못함.

---

## 6. Phase 1 — 청킹 전략 탐색

**목표**: PDF 문서의 최적 분할 크기/방식을 찾는다.
**통제**: 임베딩=baseline(ko-sroberta), 인덱스=단일

### 등록된 전략 (`eval/chunking_strategies.py`)

| 키 | 방식 | 파라미터 |
|----|------|---------|
| `baseline` | RecursiveCharacter | chunk=1000, overlap=100 |
| `small` | RecursiveCharacter | chunk=500, overlap=50 |
| `large` | RecursiveCharacter | chunk=2000, overlap=200 |
| `sliding_300` | RecursiveCharacter | chunk=300, overlap=150 (50% 슬라이딩) |
| `token_256` | TokenTextSplitter | chunk=256 tokens, overlap=32 |
| `paragraph` | RecursiveCharacter | `\n\n` 우선 분리, max=1000 |
| `page` | PyPDFLoader 페이지 단위 | 추가 분할 없음 |

CSV 문서는 어느 전략에서도 row별 Document로 고정 (청킹 불필요).

### 실험 결과

| 실험 | Hit@1 HS6 | Hit@5 HS6 | MRR HS6 | Hit@5 HS4 |
|------|-----------|-----------|---------|-----------|
| `baseline` (1000자) | 22.9% | 37.1% | 0.2843 | 54.3% |
| `chunk_small` (500자) | 22.9% | 34.3% | 0.2771 | 48.6% |
| **`chunk_large`** (2000자) ★ | **22.9%** | **42.9%** | **0.3057** | **62.9%** |
| `chunk_sliding` (300자/50%) | 22.9% | 31.4% | 0.2629 | 45.7% |
| `chunk_token` (256 tokens) | 20.0% | 40.0% | 0.2867 | 54.3% |
| `chunk_paragraph` | 20.0% | 37.1% | 0.2700 | 54.3% |
| `chunk_page` (페이지 단위) | 22.9% | **42.9%** | **0.3057** | 62.9% |

**결론**: large(2000자) = page 전략으로 동점. 이후 실험에 **large** 채택.
sliding_300이 가장 나쁨 — 짧은 청크에 맥락이 부족.

---

## 7. Phase 2 — 임베딩 모델 탐색

**목표**: 한국어 RAG에 최적인 임베딩 모델을 찾는다.
**통제**: 청킹=baseline(1000자), 인덱스=단일

### 등록된 모델 (`eval/embedding_registry.py`)

#### CPU 실행 가능

| 키 | 모델 | 크기 | 비고 |
|----|------|------|------|
| `baseline` | `jhgan/ko-sroberta-multitask` | ~500MB | 기존 production |
| `multilingual_e5` | `intfloat/multilingual-e5-large-instruct` | ~560MB | instruction prefix 있음 |
| `kure_v1` | `nlpai-lab/KURE-v1` | ~500MB | 한국어 특화 ★ |
| `snowflake_ko` | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` | ~335MB | 경량 |

#### GPU 필요 (registry 등록, 선택 실행)

| 키 | 모델 | VRAM | 로딩 방식 |
|----|------|------|----------|
| `pixie_spell` | `telepix/PIXIE-Spell-Preview-1.7B` | ~3.4GB (fp16 강제) | SentenceTransformer |
| `pixie_rune` | `telepix/PIXIE-Rune-Preview` | ~4GB | SentenceTransformer |
| `qwen3_4b_int8` | `Qwen/Qwen3-Embedding-4B` | ~4GB (INT8) | AutoModel + last-token pooling |
| `qwen3_4b` | `Qwen/Qwen3-Embedding-4B-bf16` | ~8GB+ | AutoModel + last-token pooling |
| `qwen3_8b` | `Qwen/Qwen3-Embedding-8B-bf16` | ~16GB+ | AutoModel + last-token pooling |
| `gte_qwen2_7b` | `Alibaba-NLP/gte-Qwen2-7B-instruct-fp16` | ~14GB+ | - |

> **참고**: Qwen3-Embedding은 LLM 기반 last-token pooling 아키텍처로 SentenceTransformer 미지원.
> `_AutoModelEmbeddings` 래퍼로 `transformers.AutoModel` 직접 로드 + L2 정규화.

### CPU 모델 실험 결과 (청킹=baseline 고정)

| 실험 | Hit@1 HS6 | Hit@5 HS6 | MRR HS6 | Hit@5 HS4 | MRR HS4 |
|------|-----------|-----------|---------|-----------|---------|
| `baseline` (ko-sroberta) | 22.9% | 37.1% | 0.2843 | 54.3% | 0.4329 |
| `embed_snowflake_ko` | 14.3% | 51.4% | 0.2743 | 68.6% | 0.4652 |
| **`embed_kure_v1`** ★ | **25.7%** | **45.7%** | **0.3248** | **74.3%** | **0.5676** |

**결론**: KURE-v1이 Hit@1·MRR 균형 최적. Snowflake는 Hit@5 높지만 정밀도(Hit@1) 낮음.
이후 실험에 **KURE-v1** 채택.

---

## 8. Phase 3 — 청킹 × 임베딩 최적 조합

**목표**: Phase 1 최적(large 2000자) × Phase 2 최적(KURE-v1)의 시너지를 확인한다.
**통제**: 인덱스=단일

### 실험 결과

| 실험 | 청킹 | 임베딩 | Hit@5 HS6 | MRR HS6 | Hit@5 HS4 |
|------|------|--------|-----------|---------|-----------|
| Phase 1 최적 단독 | large | ko-sroberta | 42.9% | 0.3057 | 62.9% |
| Phase 2 최적 단독 | baseline | KURE-v1 | 45.7% | 0.3248 | 74.3% |
| `large_snowflake_ko` | large | snowflake-ko | 48.6% | 0.2690 | 71.4% |
| `page_kure_v1` | page | KURE-v1 | 51.4% | 0.3471 | 74.3% |
| **`large_kure_v1`** ★ | **large** | **KURE-v1** | **54.3%** | **0.3543** | **77.1%** |

**결론**: 조합 효과 확인. 각 Phase 단독 최적(42.9%, 45.7%)보다 조합(54.3%)이 높음.

---

## 9. Phase 4 — 검색 후처리

**목표**: BM25 hybrid / CrossEncoder rerank 적용 효과를 측정한다.
**통제**: 인덱스=baseline_baseline (기존 단일 인덱스)

### BM25 Hybrid (Dense + Sparse)

`rank-bm25` 패키지 필요: `pip install rank-bm25`

RRF(Reciprocal Rank Fusion) 방식으로 Dense 검색 결과와 BM25 결과를 결합.
docs.json이 인덱스와 같은 디렉터리에 있어야 BM25가 동작함.

### CrossEncoder Reranking

`sentence-transformers` 필요. 모델: `BAAI/bge-reranker-v2-m3`

### 실험 결과

| 실험 | Hit@5 HS6 | MRR HS6 | Hit@1 HS4 | 비고 |
|------|-----------|---------|-----------|------|
| `baseline` (dense only) | 37.1% | 0.2843 | 25.7% | 기준 |
| `hybrid_baseline` | 37.1% | 0.2429 | 17.1% | MRR -0.04 (악화) |
| `rerank_baseline` | 37.1% | 0.2524 | 17.1% | 미미한 개선 |
| `hybrid_rerank` | 37.1% | **0.2952** | **22.9%** | MRR +0.011 |

**결론**: Hit@5 변화 없음. BM25 토큰화가 한국어에 최적화되지 않아 hybrid 효과 미미.
알고리즘 후처리보다 **데이터 불균형(CSV 지배 현상)** 이 근본 문제임을 확인 → Phase 5로 전환.

---

## 10. Phase 5 — 듀얼 인덱스 균형 검색

### 배경: CSV 지배 현상

기존 단일 인덱스(PDF+CSV 혼합)에서 상위 5개 문서 중 **CSV가 약 86%** 차지.
PDF(HSK 품명규격 가이드)는 HS 코드 항목별 상세 설명을 담고 있어 분류 단서가 풍부하지만,
CSV 문서 수(12,243건) >> PDF 문서 수(~2,741건) 불균형으로 검색에서 밀려남.

### 해결책: 듀얼 인덱스 + PDF 쿼터

```
통합 인덱스 (PDF+CSV)  ← (k - pdf_quota)개 검색
PDF-only 인덱스        ← pdf_quota개 검색 (강제 할당)
                          ↓
              결과 합산 → 총 k개 반환
```

인덱스 저장 구조:
```
vector_store/
├── exp_large_kure_v1/faiss_index      # 통합 인덱스 (PDF+CSV)
└── exp_large_kure_v1_pdf/faiss_index  # PDF-only 인덱스
```

### 실험 결과 — large × KURE-v1 (k=5)

| PDF 쿼터 | CSV개수 | PDF개수 | Hit@1 HS6 | Hit@5 HS6 | MRR HS6 | Hit@5 HS4 |
|---------|---------|---------|-----------|-----------|---------|-----------|
| 0 (단일) | ~4 | ~1 | 25.7% | 54.3% | 0.3543 | 77.1% |
| 1 | 4 | 1 | 25.7% | 57.1% | 0.3600 | 85.7% |
| 2 | 3 | 2 | 25.7% | 60.0% | 0.3671 | 82.9% |
| **3** ★ | **2** | **3** | **25.7%** | **62.9%** | **0.3795** | **88.6%** |
| 4 | 1 | 4 | 25.7% | 62.9% | 0.3986 | 85.7% |
| 5 (CSV=0) | 0 | 5 | 34.3% | 60.0% | 0.4595 | 88.6% |

**결론**: **PDF=3이 최적** — CSV 2개를 함께 포함해 "정확한 HS코드가 있는 관세율표 row"의 보완 효과.
PDF=5(CSV 완전 제거)는 Hit@5가 오히려 하락.

### PDF 쿼터 분석

```
Hit@5 HS6 기준
PDF=0 ──────────────────── 54.3%
PDF=1 ──────────────────────── 57.1%  +2.8%p
PDF=2 ──────────────────────────── 60.0%  +5.7%p
PDF=3 ──────────────────────────────── 62.9%  +8.6%p  ← 최적
PDF=4 ──────────────────────────────── 62.9%  (Hit@5 동일, MRR만 소폭↑)
PDF=5 ──────────────────────────── 60.0%  -2.9%p (CSV 제거로 역효과)
```

---

## 11. 추가 실험 — GPU 임베딩 모델

Phase 2~5에서 확립한 최적 조합(large 청킹 + 듀얼 인덱스)을 GPU 모델에도 적용해 성능 한계를 확인했다.

### GPU 모델 단일 인덱스 (청킹=baseline 고정)

| 실험 | 환경 | Hit@1 HS6 | Hit@5 HS6 | MRR HS6 | Hit@5 HS4 |
|------|------|-----------|-----------|---------|-----------|
| `embed_pixie_rune` | GPU | 20.0% | 42.9% | 0.2810 | 62.9% |
| `embed_qwen3_4b_int8` | GPU | 8.6% | 31.4% | 0.1538 | 48.6% |
| **`embed_pixie_spell`** | **GPU** | **28.6%** | **60.0%** | **0.3900** | **82.9%** |

### GPU 모델 × large 청킹 조합

| 실험 | 환경 | Hit@1 HS6 | Hit@5 HS6 | MRR HS6 | Hit@5 HS4 |
|------|------|-----------|-----------|---------|-----------|
| `large_pixie_rune` | GPU | 17.1% | 40.0% | 0.2510 | 60.0% |
| `large_pixie_spell` | GPU | 28.6% | 57.1% | 0.3829 | 77.1% |
| **`large_qwen3_4b_int8`** | **GPU** | **28.6%** | **60.0%** | **0.3900** | **82.9%** |

### GPU × 듀얼 인덱스 — pixie_spell + balanced search (k=5)

| PDF 쿼터 | Hit@1 HS6 | Hit@5 HS6 | MRR HS6 | Hit@5 HS4 |
|---------|-----------|-----------|---------|-----------|
| 0 (단일) | 28.6% | 57.1% | 0.3829 | 77.1% |
| 1 | 28.6% | 57.1% | 0.3829 | 77.1% |
| 2 | 28.6% | 60.0% | 0.3900 | 82.9% |
| 3 | 28.6% | 60.0% | 0.3962 | 77.1% |
| 4 | 28.6% | 60.0% | 0.4176 | 80.0% |
| **5 (CSV=0)** | **37.1%** | **60.0%** | **0.4519** | **74.3%** |

pixie_spell + 듀얼 인덱스: MRR은 0.3829 → 0.4519로 향상되지만 Hit@5는 **60.0%에서 정체**.

### GPU 실험 총평

| 모델 | VRAM | 최고 Hit@5 HS6 | 조건 |
|------|------|---------------|------|
| KURE-v1 (CPU) ★ | 0 | **62.9%** | large + balanced PDF=3 |
| PIXIE-Spell | ~3.4GB | 60.0% | baseline 청킹 or balanced PDF≥2 |
| Qwen3-4B INT8 | ~4GB | 60.0% | large 청킹 |
| PIXIE-Rune | ~4GB | 42.9% | baseline 청킹 |

- GPU가 항상 유리하지 않음 — PIXIE-Rune(GPU)은 CPU KURE-v1보다 낮음
- PIXIE-Spell은 단일 인덱스에서도 60.0% 달성했지만, 듀얼 인덱스 조합에서도 KURE-v1(62.9%)를 넘지 못함
- **CPU KURE-v1 + 듀얼 인덱스 조합이 비용 효율 최고**

---

## 12. 기술 노트 — PDF HS 코드 메타데이터 주입

### 문제

PDF를 청킹하면 continuation chunk에 `(XXXX.XX-XXXX)` 패턴이 없어 HS 코드 추출이 실패한다.

```
페이지 원본:  "(0101.29-1000) 경주말  품명  작성요령  거래품명 ..."
               ↓ 2000자 청크 분할
청크1: "(0101.29-1000) 경주말..."     → HS 추출 ✅
청크2: "품명 관세율표상 영문 기재..."  → HS 추출 ❌ None
청크3: "수입요건 가축전염병예방법..."  → HS 추출 ❌ None
```

### 해결책 (`preprocess_experiment.py`, `scripts/run_preprocessing.py`)

PDF를 먼저 페이지 단위로 로드해 `page_hs_map` 구성,
청킹 후 각 청크에 HS 코드를 메타데이터로 주입한다.

```
주입 우선순위:
  1) 청크 본문에서 (XXXX.XX-XXXX) 직접 추출
  2) 해당 페이지의 첫 번째 HS 코드 (page_hs_map)
  3) carry-forward — 직전 청크/페이지의 HS 코드
```

### 효과

`large` + KURE-v1 기준: 주입 전 51.4% → **주입 후 54.3%** (+2.9%p)
large 전략은 5456/5462 청크(99.9%)에 hs_code 메타데이터 주입 성공.

---

## 13. 실험 실행 가이드

### Step 1: 실험용 인덱스 빌드

```bash
# 특정 조합 빌드 (통합 인덱스)
python -m eval.preprocess_experiment --chunking large page --embedding kure_v1 snowflake_ko

# PDF-only 인덱스 빌드 (balanced search용)
python -m eval.preprocess_experiment --chunking large --embedding kure_v1 --pdf-only

# 등록된 전략/모델 목록 확인
python -m eval.preprocess_experiment --list

# 기존 인덱스 강제 재빌드
python -m eval.preprocess_experiment --chunking large --embedding kure_v1 --force

# CPU 가능 모델 × 전체 청킹 조합 (all-in-one)
python -m eval.preprocess_experiment --all-cpu
```

인덱스 저장 경로:
- 통합: `vector_store/exp_{chunking}_{embedding}/faiss_index`
- PDF-only: `vector_store/exp_{chunking}_{embedding}_pdf/faiss_index`

### Step 2: 실험 실행

```bash
# 전체 실험 (등록된 모든 ExperimentConfig)
python -m eval.run_experiments

# 특정 실험만
python -m eval.run_experiments --only baseline large_kure_v1 balanced_3pdf_large_kure

# 결과 JSON 저장
python -m eval.run_experiments --save

# 저장된 결과만 비교 (재실행 없이)
python -m eval.run_experiments --compare-only
```

### Step 3: 단일 실험 직접 평가

```bash
# 특정 인덱스 직접 평가
python -m eval.evaluate_retrieval \
  --vs-path vector_store/exp_large_kure_v1/faiss_index \
  --embedding kure_v1 \
  --k 5

# hybrid + rerank
python -m eval.evaluate_retrieval \
  --vs-path vector_store/exp_baseline_baseline/faiss_index \
  --embedding baseline \
  --hybrid --rerank --rerank-top-n 10

# 결과 JSON 저장
python -m eval.evaluate_retrieval --output eval/results/my_experiment.json
```

---

## 14. 결론 및 최적 설정

### 최종 최적 설정 (Production 반영 완료)

**`balanced_3pdf_large_kure`** — Hit@5 HS6 **62.9%**, MRR **0.3795**, Hit@5 HS4 **88.6%**

```
청킹:      RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
임베딩:    nlpai-lab/KURE-v1
검색:      듀얼 인덱스 균형 검색 (PDF 쿼터=3, CSV 쿼터=2)
인덱스:    vector_store/faiss_index          (통합, PDF+CSV)
           vector_store/faiss_index_pdf      (PDF-only)
HS 메타:   PDF 청킹 시 carry-forward 주입 적용
```

### 베이스라인 대비 전체 개선 경로

```
baseline (37.1%)
   ① Phase 1: 청킹 large (2000자)          → 42.9%  (+5.8%p)
   ② Phase 2: 임베딩 KURE-v1 교체          → 45.7%  (+8.6%p, 단독)
   ③ Phase 3: large × KURE-v1 시너지       → 54.3%  (①+② 조합 효과)
   ④ Phase 4: 검색 후처리                  → 변화 없음 (방향 전환)
   ⑤ Phase 5: 듀얼 인덱스 PDF 쿼터=3       → 62.9%  (+8.6%p)

최종: 37.1% → 62.9%  (총 +25.8%p, 1.7배 향상)
```

### 난이도별 최종 성능

| 난이도 | n | Hit@5 HS6 | Hit@5 HS4 |
|--------|---|-----------|-----------|
| 1 (쉬움) | 16 | 75.0% | 100.0% |
| 2 (보통) | 15 | 60.0% | 80.0% |
| 3 (어려움) | 4 | 50.0% | 75.0% |

### 한계 및 향후 개선 방향

1. **임계점 존재**: 62.9%에서 정체 — HS 코드 자체의 분류 모호성 (가공 상태·재질에 따라 코드 분기)
2. **형태소 기반 BM25**: 공백 토큰화 대신 한국어 형태소 분석기 적용 → hybrid 효과 개선 가능
3. **에이전트 레이어 강화**: HSCodeFinder reflection (확신 낮을 시 다른 키워드 재검색)
4. **Layout-aware PDF 파싱**: LlamaParse 등 활용 시 표 구조 보존 → 품목 설명 품질 향상
