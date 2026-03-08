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
  pixie_spell       telepix/PIXIE-Spell-Preview-1.7B   (fp16 강제 로드, ~3.4GB)
  pixie_rune        telepix/PIXIE-Rune-Preview
  qwen3_4b          Qwen/Qwen3-Embedding-4B-bf16
  qwen3_4b_int8     Qwen/Qwen3-Embedding-4B  (INT8 양자화, ~4GB)
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
from typing import List, Optional

from langchain_core.embeddings import Embeddings


# ── SentenceTransformer 직접 래퍼 ────────────────────────────────
# quantization_config / torch_dtype 등 HuggingFaceEmbeddings가
# 지원하지 않는 옵션이 필요한 모델용.
#
# ── AutoModel 래퍼 (Qwen3-Embedding 등 비-SBERT LLM 기반 모델용) ──
# Qwen3-Embedding은 SentenceTransformer 미지원 → AutoModel + last-token pooling

class _STEmbeddings(Embeddings):
    """SentenceTransformer 래퍼 — LangChain Embeddings 인터페이스 구현."""

    def __init__(self, model, normalize: bool = True, query_instruction: str = ""):
        self._model = model
        self._normalize = normalize
        self._query_instruction = query_instruction

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            batch_size=16,          # VRAM 절약을 위해 작은 배치
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        if self._query_instruction:
            text = self._query_instruction + text
        return self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        ).tolist()


class _AutoModelEmbeddings(Embeddings):
    """
    AutoModel 기반 last-token pooling 래퍼.
    Qwen3-Embedding 등 SentenceTransformer 미지원 LLM 임베딩 모델용.

    - 마지막 유효 토큰의 hidden state를 임베딩으로 사용
    - L2 정규화 적용
    - INT8 / bf16 / fp16 quantization 지원 (quantization_config or torch_dtype)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        load_in_8bit: bool = False,
        query_instruction: str = "",
        batch_size: int = 4,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._device = device
        self._query_instruction = query_instruction
        self._batch_size = batch_size
        self._max_length = max_length

        print(f"  [AutoModelEmbeddings] {model_id} 로드 중"
              f" (int8={load_in_8bit}, device={device})")

        model_kwargs: dict = {}
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id, **model_kwargs)
        if not load_in_8bit:
            self._model = self._model.to(device)
        self._model.eval()

    def _last_token_pool(self, hidden_states, attention_mask):
        import torch
        # 패딩이 왼쪽(left-padding)인 경우 마지막 토큰이 EOS
        left_pad = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_pad:
            return hidden_states[:, -1]
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lens]

    def _encode(self, texts: List[str]) -> List[List[float]]:
        import torch
        import torch.nn.functional as F
        import numpy as np

        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**encoded)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        if self._query_instruction:
            text = self._query_instruction + text
        return self._encode([text])[0]


# ── 설정 클래스 ─────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    name: str
    model_id: str
    description: str
    requires_gpu: bool = False
    # instruction-following 모델용 쿼리 prefix (빈 문자열이면 미사용)
    query_instruction: str = ""
    # 추가 옵션: load_in_8bit=True, torch_dtype="float16" 등
    extra_model_kwargs: dict = field(default_factory=dict)
    # "sentence_transformer"(기본) or "automodel_last_token"(Qwen3-Embedding 등)
    loader: str = "sentence_transformer"

    def load(self):
        """
        임베딩 모델 인스턴스를 반환.

        loader="automodel_last_token" → _AutoModelEmbeddings (Qwen3-Embedding용)
        extra_model_kwargs에 load_in_8bit 또는 torch_dtype이 있으면
          SentenceTransformer 직접 로드 후 _STEmbeddings로 래핑.
        그 외 → HuggingFaceEmbeddings (기본).
        """
        import torch
        from langchain_community.embeddings import HuggingFaceEmbeddings

        if self.requires_gpu and not torch.cuda.is_available():
            print(
                f"[EmbeddingRegistry] 경고: '{self.name}' 모델은 GPU를 권장하지만 "
                "CUDA를 사용할 수 없습니다. CPU로 실행합니다 (매우 느릴 수 있음)."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        extra = dict(self.extra_model_kwargs)

        # ── AutoModel last-token pooling 경로 (Qwen3-Embedding 등) ──
        if self.loader == "automodel_last_token":
            load_in_8bit = extra.pop("load_in_8bit", False)
            return _AutoModelEmbeddings(
                model_id=self.model_id,
                device=device,
                load_in_8bit=load_in_8bit,
                query_instruction=self.query_instruction,
            )

        # ── 특수 로딩 경로: load_in_8bit 또는 torch_dtype ──────
        load_in_8bit = extra.pop("load_in_8bit", False)
        torch_dtype_str = extra.pop("torch_dtype", None)

        if load_in_8bit or torch_dtype_str:
            from sentence_transformers import SentenceTransformer

            st_model_kwargs: dict = {}

            if load_in_8bit:
                from transformers import BitsAndBytesConfig
                st_model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            if torch_dtype_str:
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                st_model_kwargs["torch_dtype"] = dtype_map.get(torch_dtype_str, torch.float16)

            print(f"  [STEmbeddings] {self.model_id} 로드 중 (model_kwargs={list(st_model_kwargs.keys())})")
            st = SentenceTransformer(
                self.model_id,
                model_kwargs=st_model_kwargs,
                device=device,
            )
            return _STEmbeddings(st, normalize=True, query_instruction=self.query_instruction)

        # ── 일반 경로: HuggingFaceEmbeddings ───────────────────
        model_kwargs = {"device": device, **extra}
        encode_kwargs = {"normalize_embeddings": True}

        kwargs: dict = dict(
            model_name=self.model_id,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

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
        model_id="telepix/PIXIE-Spell-Preview-1.7B",
        description="PIXIE-Spell 1.7B fp16 강제 로드 (~3.4GB VRAM, 8GB GPU 가능)",
        requires_gpu=True,
        extra_model_kwargs={"torch_dtype": "float16"},  # fp32→fp16, VRAM 절반
    ),
    "pixie_rune": EmbeddingConfig(
        name="pixie_rune",
        model_id="telepix/PIXIE-Rune-Preview",
        description="PIXIE-Rune Preview (~2-4GB VRAM 추정, 8GB GPU 가능)",
        requires_gpu=True,
    ),
    "qwen3_4b": EmbeddingConfig(
        name="qwen3_4b",
        model_id="Qwen/Qwen3-Embedding-4B-bf16",
        description="Qwen3-Embedding 4B bf16 (~8GB VRAM, 8GB GPU OOM 위험)",
        requires_gpu=True,
    ),
    "qwen3_4b_int8": EmbeddingConfig(
        name="qwen3_4b_int8",
        model_id="Qwen/Qwen3-Embedding-4B",
        description="Qwen3-Embedding 4B INT8 양자화 (~4GB VRAM, 8GB GPU 가능, bitsandbytes 필요)",
        requires_gpu=True,
        extra_model_kwargs={"load_in_8bit": True},
        loader="automodel_last_token",
    ),
    "qwen3_8b": EmbeddingConfig(
        name="qwen3_8b",
        model_id="Qwen/Qwen3-Embedding-8B-bf16",
        description="Qwen3-Embedding 8B bf16 (~16GB VRAM 필요)",
        requires_gpu=True,
    ),
    "gte_qwen2_7b": EmbeddingConfig(
        name="gte_qwen2_7b",
        model_id="Alibaba-NLP/gte-Qwen2-7B-instruct-fp16",
        description="GTE-Qwen2-7B-instruct fp16 (~14GB VRAM 필요)",
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
