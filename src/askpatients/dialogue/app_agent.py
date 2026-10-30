"""Dr.APP pipeline: grounded symptom mapping, disease estimates, candidate
questions, hypothetical answers, printed paper scoring, and patient-facing rewrite.

The candidate disease support is fixed at initialization, a documented limitation.
Observed answers default to full-dialogue probability estimation. The legacy
recursive Bayesian approximation remains available through belief_update.
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
    question_evidence: Dict[str, dict] = field(default_factory=dict)
    grounding: List[dict] = field(default_factory=list)


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
        diseases = self.diagnosis_engine.candidate_disease_set(symptoms, opening_response)
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
        evidence = {}
        for q in candidate_questions:
            responses = self.question_generator.generate_possible_responses(q, state.dialogue_text)
            per_response_joints = []
            likelihoods = []
            for r in responses:
                cond_probs = self.question_generator.estimate_conditional_probabilities(
                    q, r, state.diseases, state.dialogue_text,
                )
                likelihoods.append(cond_probs)
                per_response_joints.append(joint_probability(cond_probs, state.distribution))
            h, _posterior = expected_entropy_for_question(per_response_joints)
            entropies[q] = h
            evidence[q] = {'responses': responses, 'likelihoods': likelihoods,
                           'virtual_posterior': _posterior, 'paper_entropy': h}

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
            question_evidence=evidence,
            grounding=[self.kb.get(d).__dict__.copy() for d in state.diseases if self.kb.get(d)],
        )

    # ---------------------------------------------------- observe response
    def observe_response(self, state: DrAPPState, question_asked: str,
                          patient_response: str) -> DrAPPState:
        """
        Incorporate the actual question as seen by the patient and the answer.
        Default: estimate from full dialogue (Section 2.3 conditioning).
        Optional legacy mode: recursive Bayesian likelihood multiplication.
        """
        new_dialogue = state.dialogue_text + f"\nDoctor: {question_asked}\nPatient: {patient_response}"
        if self.cfg.belief_update == 'dialogue':
            state.symptoms = self.symptom_mapper.map_dialogue_to_symptoms(new_dialogue)
            new_distribution = self.diagnosis_engine.initial_distribution(
                new_dialogue, state.symptoms, state.diseases)
        else:
            cond_probs = self.question_generator.estimate_conditional_probabilities(
                question_asked, patient_response, state.diseases, state.dialogue_text)
            new_distribution = bayesian_update_with_observed_response(
                state.distribution, cond_probs, floor=self.cfg.min_prob_floor)
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
