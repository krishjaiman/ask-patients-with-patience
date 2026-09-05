"""
Simple, reproducible fixed-window chunker for the knowledge base.

Medical documents / knowledge/-> chunking -> embeddings -> FAISS -> retrieval -> LLM.
This module implements the first arrow. Each DiseaseEntry's grounding text
(askpatients.medical.knowledge_base.DiseaseEntry.as_grounding_text) is split
into overlapping character windows so retrieval can operate at sub-document
granularity even for longer real MSD-derived entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    disease: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_text(disease: str, text: str, chunk_size: int = 400, overlap: int = 60) -> List[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    text = text.strip()
    if not text:
        return []
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(chunk_id=f"{disease}::{idx}", disease=disease, text=piece))
            idx += 1
        if end == len(text):
            break
        start += step
    return chunks
