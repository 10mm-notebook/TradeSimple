# app/models.py
"""
모델 로더 - OpenAI LLM 및 임베딩 모델
"""
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# 환경 변수 로드
load_dotenv()


def get_embedding_model():
    """
    한국어 임베딩 모델 로더
    - jhgan/ko-sroberta-multitask 모델 사용
    - CPU에서 실행 (GPU 없이도 동작)
    """
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def get_llm(
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: Optional[int] = None
) -> ChatOpenAI:
    """
    OpenAI LLM 인스턴스 생성
    
    Args:
        model: 사용할 모델명 (기본값: gpt-4o)
        temperature: 생성 다양성 (0-2, 기본값: 0)
        max_tokens: 최대 토큰 수 (선택사항)
    
    Returns:
        ChatOpenAI 인스턴스
    """
    kwargs = {
        "model": model,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    
    return ChatOpenAI(**kwargs)


def get_creative_llm() -> ChatOpenAI:
    """
    창의적 응답용 LLM (Report Writer 등에서 사용)
    - temperature: 0.3
    """
    return get_llm(temperature=0.3)


def get_router_llm() -> ChatOpenAI:
    """
    결정론적 라우팅용 LLM (Supervisor에서 사용)
    - temperature: 0
    """
    return get_llm(temperature=0)
