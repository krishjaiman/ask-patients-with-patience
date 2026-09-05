"""
Section 2.3 "Diagnosis Probability Prediction": builds the candidate
disease set D and the initial/updated LLM-proposed distribution
P_t(D | Gamma(S_MSD), S_t) (Equation 1).

The LLM here supplies a *proposal* distribution grounded in retrieved
causes/pathophysiology text (Gamma(S_MSD)); askpatients.bayesian.probability
.normalize_distribution then enforces the sum-to-1 constraint from Eq. 1
and applies the numerical floor used throughout the rest of the pipeline.
Subsequent turns update this distribution via Bayesian posterior updates
(askpatients.bayesian.probability.bayesian_update_with_observed_response),
NOT by re-asking the LLM for a fresh guess each turn -- that is what
keeps the "confidence" trajectory (Section 3.4.2 / Figure 4) meaningful.
"""
from __future__ import annotations

from typing import Dict, List

from askpatients.bayesian.probability import normalize_distribution
from askpatients.llm.base import BaseLLM, LLMMessage
from askpatients.medical.knowledge_base import KnowledgeBase
from askpatients.medical.schema import DiseaseProbabilityResult
from askpatients.rag.retriever import Retriever

_SYSTEM_PROMPT = (
    "You are a clinical reasoning assistant. Given a patient's reported "
    "symptoms and grounded clinical-guideline excerpts describing candidate "
    "diseases (their causes and pathophysiology), estimate a probability "
    "for EACH listed candidate disease that it explains the patient's "
    "presentation. Respond ONLY with JSON: "
    '{"probabilities": {"Disease A": 0.4, "Disease B": 0.3, ...}}. '
    "Values do not need to already sum to 1 (they will be renormalized), "
    "but should reflect relative plausibility given the grounded evidence."
)


class DiagnosisEngine:
    def __init__(self, llm: BaseLLM, kb: KnowledgeBase, retriever: Retriever,
                 min_prob_floor: float = 1e-6, top_n_diseases: int = 8):
        self.llm = llm
        self.kb = kb
        self.retriever = retriever
        self.min_prob_floor = min_prob_floor
        self.top_n_diseases = top_n_diseases

    def candidate_disease_set(self, symptoms: List[str]) -> List[str]:
        """Builds D = {d_i} by retrieving diseases whose grounding text is
        relevant to the extracted/observed symptoms so far."""
        if not symptoms:
            return self.kb.all_disease_names()[: self.top_n_diseases]
        query = "; ".join(symptoms)
        retrieved = self.retriever.query(query, top_k=self.top_n_diseases * 3)
        seen: List[str] = []
        for chunk, _ in retrieved:
            if chunk.disease not in seen:
                seen.append(chunk.disease)
            if len(seen) >= self.top_n_diseases:
                break
        if not seen:
            seen = self.kb.all_disease_names()[: self.top_n_diseases]
        return seen

    def initial_distribution(self, dialogue_text: str, symptoms: List[str],
                              diseases: List[str]) -> Dict[str, float]:
        """Equation (1): P_t(D | Gamma(S_MSD), S_t) at t = current turn."""
        grounding = self._grounding_text(diseases)
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"Dialogue so far:\n{dialogue_text}\n\n"
                    f"Extracted symptoms: {', '.join(symptoms) if symptoms else '(none yet)'}\n\n"
                    f"Candidate diseases and their grounded clinical context:\n{grounding}\n\n"
                    f"Candidate disease list: {diseases}"
                ),
            ),
        ]
        result: DiseaseProbabilityResult = self.llm.complete_structured(
            messages, DiseaseProbabilityResult, mock_context={"diseases": diseases},
        )
        # Ensure every candidate disease has an entry even if the LLM omitted one.
        raw = {d: result.probabilities.get(d, 0.0) for d in diseases}
        return normalize_distribution(raw, floor=self.min_prob_floor)

    def _grounding_text(self, diseases: List[str]) -> str:
        parts = []
        for d in diseases:
            entry = self.kb.get(d)
            if entry:
                parts.append(entry.as_grounding_text())
        return "\n---\n".join(parts)
