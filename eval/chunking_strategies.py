"""
eval/chunking_strategies.py
PDF 청킹 전략 레지스트리

지원 전략 (CSV 문서는 어느 전략에서도 row별 Document 유지 — 청킹 불필요)
────────────────────────────────────────────────────────
  baseline    RecursiveCharacterTextSplitter(1000, 100)  현재 기본값
  small       RecursiveCharacterTextSplitter(500,  50)
  large       RecursiveCharacterTextSplitter(2000, 200)
  sliding_300 RecursiveCharacterTextSplitter(300,  150)  50% 오버랩
  token_256   TokenTextSplitter(256, 32)
  paragraph   "\n\n" 우선 분리 (ParagraphConfig)
  page        PDF 페이지 단위 — 추가 분할 없음 (None 반환)
────────────────────────────────────────────────────────

사용 예시
─────────
  from eval.chunking_strategies import CHUNKING_REGISTRY, list_strategies

  cfg = CHUNKING_REGISTRY["small"]
  splitter = cfg.get_splitter()   # None이면 분할 없음 (page 전략)
  print(cfg.description)

  list_strategies()               # 등록 목록 출력
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ── 베이스 클래스 ───────────────────────────────────────────────

@dataclass
class ChunkingConfig(ABC):
    name: str
    description: str

    @abstractmethod
    def get_splitter(self):
        """
        텍스트 스플리터 인스턴스를 반환.
        None을 반환하면 분할하지 않음 (page 전략 등).
        """


# ── 구체 전략 ───────────────────────────────────────────────────

@dataclass
class RecursiveConfig(ChunkingConfig):
    """RecursiveCharacterTextSplitter 기반."""
    chunk_size: int = 1000
    chunk_overlap: int = 100

    def get_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )


@dataclass
class TokenConfig(ChunkingConfig):
    """TokenTextSplitter 기반 (토큰 수 기준)."""
    chunk_size: int = 256
    chunk_overlap: int = 32

    def get_splitter(self):
        from langchain_text_splitters import TokenTextSplitter
        return TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )


@dataclass
class ParagraphConfig(ChunkingConfig):
    """'\n\n' 우선 분리 후 RecursiveCharacterTextSplitter 적용."""
    max_chunk_size: int = 1000
    chunk_overlap: int = 100

    def get_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", ". ", " ", ""],
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
        )


@dataclass
class PageConfig(ChunkingConfig):
    """PDF 페이지 단위 유지 — 추가 분할 없음."""

    def get_splitter(self):
        return None  # preprocess_experiment.py 에서 None이면 분할 생략


# ── 레지스트리 ──────────────────────────────────────────────────

CHUNKING_REGISTRY: dict[str, ChunkingConfig] = {
    "baseline": RecursiveConfig(
        name="baseline",
        description="RecursiveCharacter 1000자, overlap=100 (현재 기본값)",
        chunk_size=1000,
        chunk_overlap=100,
    ),
    "small": RecursiveConfig(
        name="small",
        description="RecursiveCharacter 500자, overlap=50",
        chunk_size=500,
        chunk_overlap=50,
    ),
    "large": RecursiveConfig(
        name="large",
        description="RecursiveCharacter 2000자, overlap=200",
        chunk_size=2000,
        chunk_overlap=200,
    ),
    "sliding_300": RecursiveConfig(
        name="sliding_300",
        description="RecursiveCharacter 300자, overlap=150 (50% 슬라이딩)",
        chunk_size=300,
        chunk_overlap=150,
    ),
    "token_256": TokenConfig(
        name="token_256",
        description="TokenTextSplitter 256 tokens, overlap=32",
        chunk_size=256,
        chunk_overlap=32,
    ),
    "paragraph": ParagraphConfig(
        name="paragraph",
        description="단락(\\n\\n) 우선 분리, max_chunk=1000",
        max_chunk_size=1000,
        chunk_overlap=100,
    ),
    "page": PageConfig(
        name="page",
        description="PDF 페이지 단위 유지 (추가 분할 없음)",
    ),
}


def list_strategies() -> None:
    """등록된 청킹 전략 목록 출력."""
    print(f"\n{'청킹 전략 레지스트리':=<50}")
    for key, cfg in CHUNKING_REGISTRY.items():
        print(f"  {key:<15} : {cfg.description}")
    print()


if __name__ == "__main__":
    list_strategies()
    # 간단 smoke test
    for name, cfg in CHUNKING_REGISTRY.items():
        splitter = cfg.get_splitter()
        if splitter is not None:
            sample = "안녕하세요. " * 200
            chunks = splitter.split_text(sample)
            print(f"  {name:<15}: {len(chunks)}개 청크 (샘플)")
        else:
            print(f"  {name:<15}: 분할 없음 (페이지 단위)")
