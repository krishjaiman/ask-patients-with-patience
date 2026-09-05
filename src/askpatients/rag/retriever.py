"""
Retriever: builds and queries a similarity index over knowledge-base chunks.

  documents -> chunking -> embeddings -> FAISS -> retrieval -> LLM

Uses FAISS (IndexFlatIP over L2-normalized vectors == cosine similarity)
when available. Falls back to a pure-numpy brute-force cosine search if
`faiss` is not installed, so the pipeline still runs (this fallback is
functionally equivalent for the small knowledge bases used here; FAISS
matters for scale, not for correctness).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .chunking import Chunk, chunk_text
from .embeddings import BaseEmbedder
from askpatients.medical.knowledge_base import KnowledgeBase


class Retriever:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.chunks: List[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._faiss_index = None

    # ---------------------------------------------------------------- build
    @classmethod
    def build_from_knowledge_base(cls, kb: KnowledgeBase, embedder: BaseEmbedder,
                                   chunk_size: int = 400, overlap: int = 60) -> "Retriever":
        r = cls(embedder)
        all_chunks: List[Chunk] = []
        for entry in kb.entries:
            all_chunks.extend(
                chunk_text(entry.disease, entry.as_grounding_text(), chunk_size, overlap)
            )
        r.chunks = all_chunks
        if all_chunks:
            vectors = embedder.embed([c.text for c in all_chunks])
            r._vectors = vectors
            r._build_index(vectors)
        return r

    def _build_index(self, vectors: np.ndarray) -> None:
        try:
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors.astype(np.float32))
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None  # numpy fallback used in `query`

    # ----------------------------------------------------------------- I/O
    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({"chunk_id": c.chunk_id, "disease": c.disease, "text": c.text}) + "\n")
        if self._vectors is not None:
            np.save(str(Path(index_path).with_suffix(".npy")), self._vectors)

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path,
              embedder: BaseEmbedder) -> "Retriever":
        r = cls(embedder)
        metadata_path = Path(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                r.chunks.append(Chunk(chunk_id=d["chunk_id"], disease=d["disease"], text=d["text"]))
        npy_path = Path(index_path).with_suffix(".npy")
        if npy_path.exists():
            r._vectors = np.load(str(npy_path))
            r._build_index(r._vectors)
        return r

    # -------------------------------------------------------------- query
    def query(self, text: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.chunks or self._vectors is None:
            return []
        qvec = self.embedder.embed([text])[0]
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(qvec[None, :].astype(np.float32), top_k)
            pairs = [(self.chunks[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]
            return pairs
        # numpy fallback: cosine similarity (vectors are already normalized)
        sims = self._vectors @ qvec
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def query_disease_scoped(self, text: str, disease: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Retrieve only within a given disease's chunks -- used when we already
        know which disease's Gamma(d_i) we need (Eq. 2's per-disease grounding)."""
        results = self.query(text, top_k=max(top_k * 5, 10))
        filtered = [(c, s) for c, s in results if c.disease == disease]
        return filtered[:top_k]
