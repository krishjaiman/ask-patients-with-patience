import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askpatients.medical.knowledge_base import KnowledgeBase
from askpatients.rag.chunking import chunk_text
from askpatients.rag.embeddings import MockEmbedder
from askpatients.rag.retriever import Retriever

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base" / "sample_diseases.json"


def test_chunk_text_respects_size_and_overlap():
    text = "a" * 1000
    chunks = chunk_text("disease", text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200


def test_chunk_text_empty_returns_empty():
    assert chunk_text("d", "", chunk_size=100, overlap=10) == []


def test_chunk_overlap_must_be_smaller_than_size():
    try:
        chunk_text("d", "abc", chunk_size=10, overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mock_embedder_shape_and_normalization():
    embedder = MockEmbedder(dim=64)
    vecs = embedder.embed(["hello world", "foo bar baz"])
    assert vecs.shape == (2, 64)
    import numpy as np
    norms = np.linalg.norm(vecs, axis=1)
    for n in norms:
        assert abs(n - 1.0) < 1e-5 or n == 0.0


def test_retriever_build_and_query_returns_relevant_disease():
    kb = KnowledgeBase.from_json_file(KB_PATH)
    embedder = MockEmbedder(dim=128)
    retriever = Retriever.build_from_knowledge_base(kb, embedder, chunk_size=300, overlap=40)
    assert len(retriever.chunks) > 0

    results = retriever.query("milk ice cream stomach cramps bloating dairy", top_k=5)
    assert len(results) > 0
    top_diseases = {c.disease for c, _ in results}
    # Lactose Intolerance's grounding text is dairy/cramping-heavy, so it
    # should show up among the top results for a dairy-related query.
    assert "Lactose Intolerance" in top_diseases


def test_retriever_disease_scoped_query():
    kb = KnowledgeBase.from_json_file(KB_PATH)
    embedder = MockEmbedder(dim=128)
    retriever = Retriever.build_from_knowledge_base(kb, embedder)
    results = retriever.query_disease_scoped("nasal", "Allergic Rhinitis", top_k=3)
    for chunk, _ in results:
        assert chunk.disease == "Allergic Rhinitis"


def test_retriever_save_and_load_roundtrip(tmp_path):
    kb = KnowledgeBase.from_json_file(KB_PATH)
    embedder = MockEmbedder(dim=64)
    retriever = Retriever.build_from_knowledge_base(kb, embedder)
    index_path = tmp_path / "idx.index"
    meta_path = tmp_path / "chunks.jsonl"
    retriever.save(index_path, meta_path)

    loaded = Retriever.load(index_path, meta_path, embedder)
    assert len(loaded.chunks) == len(retriever.chunks)
    results = loaded.query("dizziness spinning", top_k=3)
    assert len(results) > 0
