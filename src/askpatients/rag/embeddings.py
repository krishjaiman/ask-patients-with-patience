"""
Embedding backends for the RAG index. Second arrow of:
  documents -> chunking -> embeddings -> FAISS -> retrieval -> LLM
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from askpatients.config import RAGConfig


class BaseEmbedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of embeddings."""
        raise NotImplementedError


class MockEmbedder(BaseEmbedder):
    """
    Deterministic hashed bag-of-words embedder. No network, no model
    weights. Good enough to make retrieval *functional* (semantically
    similar / overlapping-vocabulary text ends up closer) for tests and
    offline demos, but it is not a substitute for a real embedding model
    in a genuine reproduction run -- switch rag.embedding_provider to
    "openai" or "sentence-transformers" for that.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._token_re = re.compile(r"[a-zA-Z]+")

    def _hash_bucket(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16) % self.dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._token_re.findall(text.lower())
            for tok in tokens:
                out[i, self._hash_bucket(tok)] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        import os
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai to use rag.embedding_provider=openai") from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = 1536 if "small" in model else 3072

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class SentenceTransformersEmbedder(BaseEmbedder):
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "pip install sentence-transformers to use rag.embedding_provider="
                "sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(cfg: RAGConfig) -> BaseEmbedder:
    if cfg.embedding_provider == "mock":
        return MockEmbedder()
    if cfg.embedding_provider == "openai":
        return OpenAIEmbedder(model=cfg.embedding_model)
    if cfg.embedding_provider == "sentence-transformers":
        return SentenceTransformersEmbedder(model=cfg.embedding_model)
    raise ValueError(f"Unknown rag.embedding_provider: {cfg.embedding_provider!r}")
