"""
Orchestrates a full multi-turn conversation between Dr.APP (or a
baseline) and the patient simulator, for `max_turns` iterations
(Section 3.3: "the multi-turn setup involving six rounds of iterative
questioning").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from askpatients.dialogue.app_agent import DrAPP, DrAPPState
from askpatients.dialogue.baselines import NaiveMultiTurnBaseline, NaiveMultiTurnState
from askpatients.patient.simulator import PatientSimulator


@dataclass
class DialogueTurn:
    turn: int
    clinical_question: str
    human_question: str
    patient_response: str
    entropy_after: Optional[float] = None
    top_diagnosis_before: Optional[str] = None
    top_diagnosis_prob_before: Optional[float] = None


@dataclass
class DialogueResult:
    case_id: str
    turns: List[DialogueTurn] = field(default_factory=list)
    entropy_trajectory: List[float] = field(default_factory=list)
    final_diagnosis: str = ""
    final_diagnosis_prob: float = 0.0
    final_distribution: dict = field(default_factory=dict)
    ground_truth: str = ""


class DialogueOrchestrator:
    """Runs Dr.APP's turn loop against a PatientSimulator."""

    def __init__(self, app: DrAPP, max_turns: int = 6):
        self.app = app
        self.max_turns = max_turns

    def run(self, case_id: str, opening_question: str, patient: PatientSimulator,
            ground_truth: str) -> DialogueResult:
        opening_response = patient.opening_statement()
        state = self.app.initialize(opening_question, opening_response)
        result = DialogueResult(case_id=case_id, ground_truth=ground_truth,
                                 entropy_trajectory=list(state.entropy_history))

        for turn in range(1, self.max_turns + 1):
            step = self.app.propose_next_question(state)
            patient_answer = patient.answer(step.human_question, state.dialogue_text)
            state = self.app.observe_response(state, step.clinical_question, patient_answer)
            result.turns.append(DialogueTurn(
                turn=turn,
                clinical_question=step.clinical_question,
                human_question=step.human_question,
                patient_response=patient_answer,
                entropy_after=state.entropy_history[-1],
                top_diagnosis_before=step.top_diagnosis,
                top_diagnosis_prob_before=step.top_diagnosis_prob,
            ))
            result.entropy_trajectory.append(state.entropy_history[-1])

        diagnosis, prob = self.app.final_diagnosis(state)
        result.final_diagnosis = diagnosis
        result.final_diagnosis_prob = prob
        result.final_distribution = dict(state.distribution)
        return result


class NaiveMultiTurnOrchestrator:
    """Runs the ungrounded, non-Bayesian NaiveMultiTurnBaseline for comparison."""

    def __init__(self, baseline: NaiveMultiTurnBaseline, max_turns: int = 6):
        self.baseline = baseline
        self.max_turns = max_turns

    def run(self, case_id: str, opening_question: str, patient: PatientSimulator,
            ground_truth: str) -> dict:
        opening_response = patient.opening_statement()
        state = self.baseline.initialize(opening_question, opening_response)
        transcript = []
        for turn in range(1, self.max_turns + 1):
            q = self.baseline.propose_next_question(state)
            a = patient.answer(q, state.dialogue_text)
            state = self.baseline.observe_response(state, q, a)
            transcript.append({"turn": turn, "question": q, "response": a})
        ranked = self.baseline.final_diagnosis(state)
        return {
            "case_id": case_id, "ground_truth": ground_truth,
            "transcript": transcript, "ranked_diagnoses": ranked,
        }
