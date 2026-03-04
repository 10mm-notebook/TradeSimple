"""
eval/embedding_registry.py
임베딩 모델 레지스트리

즉시 실험 가능 (CPU / 경량 GPU)
────────────────────────────────────────────────────────
  baseline          jhgan/ko-sroberta-multitask             ~500MB  CPU
  multilingual_e5   intfloat/multilingual-e5-large-instruct ~560MB  CPU
  kure_v1           nlpai-lab/KURE-v1                       CPU
  snowflake_ko      dragonkue/snowflake-arctic-embed-l-v2.0-ko ~335MB CPU

GPU 필요 (registry 등록, 실행은 선택)
────────────────────────────────────────────────────────
  pixie_spell       telepix/PIXIE-Spell-Preview-1.7B-fp16
  pixie_rune        telepix/PIXIE-Rune-Preview
  qwen3_4b          Qwen/Qwen3-Embedding-4B-bf16
  qwen3_8b          Qwen/Qwen3-Embedding-8B-bf16
  gte_qwen2_7b      Alibaba-NLP/gte-Qwen2-7B-instruct-fp16
────────────────────────────────────────────────────────

사용 예시
─────────
  from eval.embedding_registry import EMBEDDING_REGISTRY, list_embeddings

  cfg = EMBEDDING_REGISTRY["multilingual_e5"]
  embedder = cfg.load()   # HuggingFaceEmbeddings 인스턴스

  list_embeddings()       # 등록 목록 출력
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── 설정 클래스 ─────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    name: str
    model_id: str
    description: str
    requires_gpu: bool = False
    # instruction-following 모델용 쿼리 prefix (빈 문자열이면 미사용)
    query_instruction: str = ""

    def load(self):
        """
        HuggingFaceEmbeddings 인스턴스를 반환.

        requires_gpu=True인 모델은 CUDA 없이 호출 시 경고를 출력하지만
        fallback으로 CPU 실행 시도 (매우 느릴 수 있음).
        """
        import torch
        from langchain_community.embeddings import HuggingFaceEmbeddings

        if self.requires_gpu and not torch.cuda.is_available():
            print(
                f"[EmbeddingRegistry] 경고: '{self.name}' 모델은 GPU를 권장하지만 "
                "CUDA를 사용할 수 없습니다. CPU로 실행합니다 (매우 느릴 수 있음)."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}

        kwargs: dict = dict(
            model_name=self.model_id,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # query_instruction이 있으면 query/passage 분리 인코딩
        if self.query_instruction:
            kwargs["query_instruction"] = self.query_instruction

        return HuggingFaceEmbeddings(**kwargs)


# ── 레지스트리 ──────────────────────────────────────────────────

EMBEDDING_REGISTRY: dict[str, EmbeddingConfig] = {
    # ── 즉시 실험 가능 (CPU) ──────────────────────────────────
    "baseline": EmbeddingConfig(
        name="baseline",
        model_id="jhgan/ko-sroberta-multitask",
        description="한국어 sRoBERTa (현재 기본값, ~500MB, CPU)",
        requires_gpu=False,
    ),
    "multilingual_e5": EmbeddingConfig(
        name="multilingual_e5",
        model_id="intfloat/multilingual-e5-large-instruct",
        description="Multilingual E5-large-instruct (~560MB, CPU, instruction prefix)",
        requires_gpu=False,
        query_instruction="Represent this sentence for searching relevant passages: ",
    ),
    "kure_v1": EmbeddingConfig(
        name="kure_v1",
        model_id="nlpai-lab/KURE-v1",
        description="KURE-v1 한국어 특화 임베딩 (CPU)",
        requires_gpu=False,
    ),
    "snowflake_ko": EmbeddingConfig(
        name="snowflake_ko",
        model_id="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        description="Snowflake Arctic Embed ko (~335MB, CPU)",
        requires_gpu=False,
    ),

    # ── GPU 필요 (registry 등록, 실행은 CUDA 환경에서만 권장) ──
    "pixie_spell": EmbeddingConfig(
        name="pixie_spell",
        model_id="telepix/PIXIE-Spell-Preview-1.7B-fp16",
        description="PIXIE-Spell 1.7B fp16 (GPU 권장)",
        requires_gpu=True,
    ),
    "pixie_rune": EmbeddingConfig(
        name="pixie_rune",
        model_id="telepix/PIXIE-Rune-Preview",
        description="PIXIE-Rune Preview (GPU 권장)",
        requires_gpu=True,
    ),
    "qwen3_4b": EmbeddingConfig(
        name="qwen3_4b",
        model_id="Qwen/Qwen3-Embedding-4B-bf16",
        description="Qwen3-Embedding 4B bf16 (GPU 필요)",
        requires_gpu=True,
    ),
    "qwen3_8b": EmbeddingConfig(
        name="qwen3_8b",
        model_id="Qwen/Qwen3-Embedding-8B-bf16",
        description="Qwen3-Embedding 8B bf16 (GPU 필요)",
        requires_gpu=True,
    ),
    "gte_qwen2_7b": EmbeddingConfig(
        name="gte_qwen2_7b",
        model_id="Alibaba-NLP/gte-Qwen2-7B-instruct-fp16",
        description="GTE-Qwen2-7B-instruct fp16 (GPU 필요)",
        requires_gpu=True,
    ),
}

# CPU 즉시 실험 가능한 모델 목록 (편의 상수)
CPU_MODELS: list[str] = [k for k, v in EMBEDDING_REGISTRY.items() if not v.requires_gpu]
GPU_MODELS: list[str] = [k for k, v in EMBEDDING_REGISTRY.items() if v.requires_gpu]


def list_embeddings() -> None:
    """등록된 임베딩 모델 목록 출력."""
    print(f"\n{'임베딩 모델 레지스트리':=<55}")
    print(f"  {'이름':<20} {'GPU':<5} {'설명'}")
    print(f"  {'-'*20} {'-'*5} {'-'*30}")
    for key, cfg in EMBEDDING_REGISTRY.items():
        gpu_mark = "✓" if cfg.requires_gpu else "-"
        print(f"  {key:<20} {gpu_mark:<5} {cfg.description}")
    print()


if __name__ == "__main__":
    list_embeddings()
