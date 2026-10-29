"""Embeddings multilíngues (bom para português) com sentence-transformers.

O modelo e5 espera os prefixos 'query: ' e 'passage: '.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_passages(textos: list[str]) -> list[list[float]]:
    return _model().encode([f"passage: {t}" for t in textos], normalize_embeddings=True).tolist()


def embed_queries(textos: list[str]) -> list[list[float]]:
    return _model().encode([f"query: {t}" for t in textos], normalize_embeddings=True).tolist()
