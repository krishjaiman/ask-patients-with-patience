"""
Dr.APP: the full per-turn pipeline of Section 2 / Figure 2 of the paper,
wired together from the (independently testable) building blocks:

  askpatients.medical.mapping.SymptomMapper        -- Section 2.2
  askpatients.medical.diagnosis.DiagnosisEngine     -- Section 2.3, Eq. 1
  askpatients.questions.generator.QuestionGenerator -- Sections 2.4-2.5
  askpatients.bayesian.probability / entropy        -- Eq. 2-5
  askpatients.dialogue.humanize.HumanCentricRewriter -- Section 2.7

IMPLEMENTATION DECISION (undocumented in the paper): the candidate
disease set D = {d_i} is fixed once at dialogue initialization from the
symptoms present in the opening exchange S_1, rather than being
recomputed from scratch every turn. Only the *probabilities* P_t(D)
evolve turn-to-turn via Bayesian updating. This matches Eq. 1's framing
of D as "the set of possible diseases" (a fixed index set) and keeps
the entropy trajectory (Fig. 4) meaningful -- entropy over a shifting
support set is not directly comparable across turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from askpatients.bayesian.entropy import expected_entropy_for_question, select_min_entropy_question
from askpatients.bayesian.probability import (
    bayesian_update_with_observed_response, joint_probability, Distribution,
)
from askpatients.config import AppAlgoConfig
from askpatients.dialogue.humanize import HumanCentricRewriter
from askpatients.llm.base import BaseLLM
from askpatients.medical.diagnosis import DiagnosisEngine
from askpatients.medical.knowledge_base import KnowledgeBase
from askpatients.medical.mapping import SymptomMapper
from askpatients.questions.generator import QuestionGenerator
from askpatients.rag.retriever import Retriever


@dataclass
class DrAPPState:
    dialogue_text: str = ""
    symptoms: List[str] = field(default_factory=list)
    diseases: List[str] = field(default_factory=list)
    distribution: Distribution = field(default_factory=dict)
    turn: int = 0
    entropy_history: List[float] = field(default_factory=list)  # H_t at each iteration


@dataclass
class DrAPPStepResult:
    clinical_question: str
    human_question: str
    question_entropies: Dict[str, float]          # H_{q_k} for every candidate
    candidate_diseases: List[str]
    prior_distribution: Distribution
    top_diagnosis: str
    top_diagnosis_prob: float


class DrAPP:
    def __init__(self, llm: BaseLLM, kb: KnowledgeBase, retriever: Retriever,
                 algo_cfg: AppAlgoConfig):
        self.llm = llm
        self.kb = kb
        self.retriever = retriever
        self.cfg = algo_cfg
        self.symptom_mapper = SymptomMapper(llm, retriever)
        self.diagnosis_engine = DiagnosisEngine(
            llm, kb, retriever, min_prob_floor=algo_cfg.min_prob_floor,
            top_n_diseases=algo_cfg.top_n_diseases_for_questions,
        )
        self.question_generator = QuestionGenerator(llm, kb, algo_cfg)
        self.humanizer = HumanCentricRewriter(llm)

    # --------------------------------------------------------- initialize
    def initialize(self, opening_question: str, opening_response: str) -> DrAPPState:
        """S_1 = (q_1, r_1) -> initial symptoms, disease set D, and P_1(D)."""
        dialogue_text = f"Doctor: {opening_question}\nPatient: {opening_response}"
        symptoms = self.symptom_mapper.map_dialogue_to_symptoms(dialogue_text)
        diseases = self.diagnosis_engine.candidate_disease_set(symptoms)
        distribution = self.diagnosis_engine.initial_distribution(dialogue_text, symptoms, diseases)
        return DrAPPState(
            dialogue_text=dialogue_text, symptoms=symptoms, diseases=diseases,
            distribution=distribution, turn=1,
            entropy_history=[self._current_entropy(distribution)],
        )

    # -------------------------------------------------------------- step
    def propose_next_question(self, state: DrAPPState) -> DrAPPStepResult:
        """
        Runs Sections 2.4-2.6 for the current state:
          1. generate candidate questions Q_{t+1} (Section 2.4)
          2. for each q_k, simulate responses R_k and estimate their
             per-disease conditional likelihoods (Section 2.5)
          3. compute the expected posterior + its entropy H_{q_k} (Eq. 2-4)
          4. select q*_{t+1} = argmin H_{q_k} (Eq. 5)
          5. rewrite q*_{t+1} in human-centric language (Section 2.7)
        """
        leading = self._top_diseases(state.distribution, self.cfg.top_n_diseases_for_questions)
        candidate_questions = self.question_generator.generate_candidate_questions(
            state.dialogue_text, leading,
        )
        if not candidate_questions:
            raise RuntimeError("Question generator produced no candidate questions.")

        entropies: Dict[str, float] = {}
        for q in candidate_questions:
            responses = self.question_generator.generate_possible_responses(q, state.dialogue_text)
            per_response_joints = []
            for r in responses:
                cond_probs = self.question_generator.estimate_conditional_probabilities(
                    q, r, state.diseases,
                )
                per_response_joints.append(joint_probability(cond_probs, state.distribution))
            h, _posterior = expected_entropy_for_question(per_response_joints)
            entropies[q] = h

        q_star = select_min_entropy_question(entropies)
        human_q = self.humanizer.rewrite(q_star, state.dialogue_text)

        top_disease, top_prob = self._argmax(state.distribution)
        return DrAPPStepResult(
            clinical_question=q_star,
            human_question=human_q,
            question_entropies=entropies,
            candidate_diseases=state.diseases,
            prior_distribution=dict(state.distribution),
            top_diagnosis=top_disease,
            top_diagnosis_prob=top_prob,
        )

    # ---------------------------------------------------- observe response
    def observe_response(self, state: DrAPPState, question_asked: str,
                          patient_response: str) -> DrAPPState:
        """
        Incorporates the ACTUAL patient response r_{t+1}: forms S_{t+1} and
        updates P_{t+1}(D) via Bayes' rule (see
        bayesian.probability.bayesian_update_with_observed_response).
        """
        cond_probs = self.question_generator.estimate_conditional_probabilities(
            question_asked, patient_response, state.diseases,
        )
        new_distribution = bayesian_update_with_observed_response(
            state.distribution, cond_probs, floor=self.cfg.min_prob_floor,
        )
        new_dialogue = state.dialogue_text + f"\nDoctor: {question_asked}\nPatient: {patient_response}"
        state.dialogue_text = new_dialogue
        state.distribution = new_distribution
        state.turn += 1
        state.entropy_history.append(self._current_entropy(new_distribution))
        return state

    # -------------------------------------------------------------- final
    def final_diagnosis(self, state: DrAPPState) -> tuple[str, float]:
        return self._argmax(state.distribution)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _top_diseases(distribution: Distribution, n: int) -> List[str]:
        return [d for d, _ in sorted(distribution.items(), key=lambda kv: -kv[1])[:n]]

    @staticmethod
    def _argmax(distribution: Distribution) -> tuple[str, float]:
        if not distribution:
            return "Unknown", 0.0
        d, p = max(distribution.items(), key=lambda kv: kv[1])
        return d, p

    @staticmethod
    def _current_entropy(distribution: Distribution) -> float:
        from askpatients.bayesian.entropy import shannon_entropy
        return shannon_entropy(distribution)
