"""
Sections 2.4 "Question Generation" and 2.5 "One-more Step Thinking &
Conditional Probability Generation".

This module is the LLM-facing half of the active-learning loop: it
produces the *inputs* (candidate questions Q_{t+1}, plausible responses
R_k per candidate question, and per-disease response likelihoods
P(r_k^l | Gamma(d_i))) that askpatients.bayesian consumes to compute
Eq. 2-5. No probability arithmetic happens in this module -- it is
strictly "ask the LLM for grounded candidates", never "ask the LLM to
rank/select".
"""
from __future__ import annotations

from typing import Dict, List

from askpatients.config import AppAlgoConfig
from askpatients.llm.base import BaseLLM, LLMMessage
from askpatients.medical.knowledge_base import KnowledgeBase
from askpatients.medical.schema import (
    CandidateQuestionsResult, PossibleResponsesResult, ConditionalProbabilityResult,
)

_QUESTION_GEN_SYSTEM = (
    "You are a clinical follow-up question generator. Given the dialogue so "
    "far and grounded clinical-guideline excerpts (their 'Diagnosis' and "
    "'What a doctor does' sections) for the current leading candidate "
    "diseases, propose a pool of distinct, clinically useful follow-up "
    "questions a doctor could ask next to discriminate between the "
    "candidate diseases. Questions should be specific and grounded, not "
    "generic. Respond ONLY with JSON: {\"questions\": [\"...\", ...]}."
)

_RESPONSE_SIM_SYSTEM = (
    "You are simulating plausible patient answers. Given a single follow-up "
    "question and the dialogue context, generate several DISTINCT, "
    "realistic ways a patient might answer it (covering a spread of "
    "plausible outcomes, not just paraphrases of one answer). Respond ONLY "
    "with JSON: {\"responses\": [\"...\", ...]}."
)

_COND_PROB_SYSTEM = (
    "You are a clinical likelihood estimator. Given ONE specific hypothetical "
    "patient response to a follow-up question, and a list of candidate "
    "diseases with their grounded clinical context (causes/pathophysiology), "
    "estimate P(response | disease) for EACH candidate disease: how likely "
    "a patient with that disease would give exactly this response, on a "
    "0.0-1.0 scale (independent per disease -- these do NOT need to sum to "
    "1). Respond ONLY with JSON: "
    "{\"probabilities\": {\"Disease A\": 0.4, \"Disease B\": 0.1, ...}}."
)


class QuestionGenerator:
    def __init__(self, llm: BaseLLM, kb: KnowledgeBase, algo_cfg: AppAlgoConfig):
        self.llm = llm
        self.kb = kb
        self.cfg = algo_cfg

    # ---------------------------------------------------------- Section 2.4
    def generate_candidate_questions(self, dialogue_text: str,
                                      leading_diseases: List[str]) -> List[str]:
        """Q_{t+1} = {q_1, ..., q_K}, grounded in Upsilon(S_MSD)."""
        grounding = self._diagnosis_grounding(leading_diseases)
        messages = [
            LLMMessage(role="system", content=_QUESTION_GEN_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    f"Dialogue so far:\n{dialogue_text}\n\n"
                    f"Leading candidate diseases: {leading_diseases}\n\n"
                    f"Grounded 'Diagnosis' / 'What a doctor does' context:\n{grounding}\n\n"
                    f"Propose up to {self.cfg.num_candidate_questions} distinct follow-up questions."
                ),
            ),
        ]
        focus_symptom = self._focus_symptom(leading_diseases)
        result: CandidateQuestionsResult = self.llm.complete_structured(
            messages, CandidateQuestionsResult,
            mock_context={
                "num_questions": self.cfg.num_candidate_questions,
                "focus_symptom": focus_symptom,
            },
        )
        return result.questions[: self.cfg.num_candidate_questions]

    # ---------------------------------------------------------- Section 2.5
    def generate_possible_responses(self, question: str, dialogue_text: str) -> List[str]:
        """R_k = {r_k^1, ..., r_k^L}, 2 <= L <= 5 per Section 3.3."""
        messages = [
            LLMMessage(role="system", content=_RESPONSE_SIM_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    f"Dialogue so far:\n{dialogue_text}\n\n"
                    f"Candidate follow-up question: \"{question}\"\n\n"
                    f"Generate between {self.cfg.min_responses_per_question} and "
                    f"{self.cfg.max_responses_per_question} distinct plausible patient answers."
                ),
            ),
        ]
        result: PossibleResponsesResult = self.llm.complete_structured(
            messages, PossibleResponsesResult,
            mock_context={
                "min_l": self.cfg.min_responses_per_question,
                "max_l": self.cfg.max_responses_per_question,
            },
        )
        responses = result.responses[: self.cfg.max_responses_per_question]
        if len(responses) < self.cfg.min_responses_per_question:
            # Guarantee the L >= 2 lower bound even if the LLM under-produced.
            responses += [f"(no further detail beyond: {question})"] * (
                self.cfg.min_responses_per_question - len(responses)
            )
        return responses

    def estimate_conditional_probabilities(self, question: str, response: str,
                                            diseases: List[str]) -> Dict[str, float]:
        """P(r_k^l | Gamma(d_i)) for every candidate disease -- the raw input
        to Eq. 2's joint-probability computation (askpatients.bayesian)."""
        grounding = self._diagnosis_grounding(diseases, causes_only=True)
        messages = [
            LLMMessage(role="system", content=_COND_PROB_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    f"Question asked: \"{question}\"\n"
                    f"Hypothetical patient response: \"{response}\"\n\n"
                    f"Candidate diseases and grounded context:\n{grounding}\n\n"
                    f"Candidate disease list: {diseases}"
                ),
            ),
        ]
        result: ConditionalProbabilityResult = self.llm.complete_structured(
            messages, ConditionalProbabilityResult,
            mock_context={"diseases": diseases, "response": response},
        )
        return {d: float(result.probabilities.get(d, 0.0)) for d in diseases}

    # ------------------------------------------------------------- helpers
    def _diagnosis_grounding(self, diseases: List[str], causes_only: bool = False) -> str:
        parts = []
        for d in diseases:
            entry = self.kb.get(d)
            if not entry:
                continue
            if causes_only:
                parts.append(f"Disease: {entry.disease}\nCauses: {entry.causes}\n"
                              f"Pathophysiology: {entry.pathophysiology}")
            else:
                parts.append(entry.as_grounding_text())
        return "\n---\n".join(parts)

    def _focus_symptom(self, diseases: List[str]) -> str:
        for d in diseases:
            entry = self.kb.get(d)
            if entry and entry.symptoms:
                return entry.symptoms[0]
        return "symptoms"
