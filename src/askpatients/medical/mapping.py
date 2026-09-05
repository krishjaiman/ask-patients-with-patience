"""
Section 2.2 "Mapping into MSD": maps free-text dialogue into a normalized
symptom set S_MSD, using RAG retrieval over the knowledge base to ground
the mapping in the (professional + consumer) symptom vocabulary, exactly
as the paper describes ("we store symptom-description pairs locally and
apply RAG to retrieve the most relevant symptoms").
"""
from __future__ import annotations

from typing import List

from askpatients.llm.base import BaseLLM, LLMMessage
from askpatients.medical.schema import SymptomMappingResult
from askpatients.rag.retriever import Retriever

_SYSTEM_PROMPT = (
    "You are a clinical symptom-extraction assistant. Given a patient's "
    "free-text description and a controlled vocabulary of known symptoms, "
    "identify which vocabulary symptoms are being described (map colloquial "
    "language to clinical terms). Only use symptoms from the provided "
    "vocabulary plus close paraphrases. Respond with JSON: "
    '{"symptoms": ["symptom one", "symptom two", ...]}.'
)


class SymptomMapper:
    """Implements S_MSD = mapping(S_1) using RAG-retrieved vocabulary context."""

    def __init__(self, llm: BaseLLM, retriever: Retriever, top_k: int = 8):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k

    def map_dialogue_to_symptoms(self, dialogue_text: str) -> List[str]:
        retrieved = self.retriever.query(dialogue_text, top_k=self.top_k)
        vocab = sorted({c.disease for c, _ in retrieved})  # disease names as weak proxy
        # Pull the actual symptom vocabulary from retrieved chunk text as well,
        # since chunks are built from DiseaseEntry.as_grounding_text().
        retrieved_text = "\n---\n".join(c.text for c, _ in retrieved)

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"Patient dialogue so far:\n{dialogue_text}\n\n"
                    f"Relevant clinical-guideline context (retrieved via RAG):\n{retrieved_text}\n\n"
                    "Extract the symptoms the patient is describing."
                ),
            ),
        ]
        result: SymptomMappingResult = self.llm.complete_structured(
            messages, SymptomMappingResult,
            mock_context={"symptom_vocab": self._vocab_from_chunks(retrieved_text)},
        )
        return result.symptoms

    @staticmethod
    def _vocab_from_chunks(retrieved_text: str) -> List[str]:
        # Best-effort extraction of "Associated symptoms: a, b, c" lines for the mock LLM.
        vocab: List[str] = []
        for line in retrieved_text.splitlines():
            if line.lower().startswith("associated symptoms:"):
                items = line.split(":", 1)[1].split(",")
                vocab.extend(s.strip() for s in items if s.strip())
        return vocab
