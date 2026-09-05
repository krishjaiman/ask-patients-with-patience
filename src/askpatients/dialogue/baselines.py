"""
Baselines matching Figure 1(a) and 1(b) of the paper, used as points of
comparison against Dr.APP in evaluation (Section 3.5, Table 1):

  OneShotBaseline       -- Fig. 1(a): "existing LLMs follow a one-shot
                            diagnostic approach, generating multiple
                            possible diseases without asking follow-up
                            questions."
  NaiveMultiTurnBaseline -- Fig. 1(b): "LLMs can be prompted for
                            multi-turn dialogues [but] often overwhelm
                            users with excessive mechanical inquiries" --
                            i.e. multi-turn WITHOUT grounded reasoning or
                            Bayesian active-learning question selection;
                            the LLM just picks its own next question.

Both baselines deliberately do NOT touch askpatients.bayesian -- that is
the point of comparison Dr.APP is being evaluated against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from askpatients.llm.base import BaseLLM, LLMMessage
from askpatients.medical.knowledge_base import KnowledgeBase
from askpatients.medical.schema import OneShotDiagnosisResult, NaiveFollowUpResult

_ONE_SHOT_SYSTEM = (
    "You are a medical assistant. Based only on the patient's initial "
    "message (no follow-up questions), list your ranked candidate "
    "diagnoses, most likely first. Respond ONLY with JSON: "
    '{"ranked_diagnoses": ["Diagnosis A", "Diagnosis B", ...]}.'
)

_NAIVE_FOLLOWUP_SYSTEM = (
    "You are a medical assistant conducting a multi-turn consultation. "
    "Based on the conversation so far, ask ONE follow-up question to "
    "gather more information, in plain clinical style. Respond ONLY with "
    'JSON: {"question": "..."}.'
)

_NAIVE_FINAL_SYSTEM = (
    "You are a medical assistant. Based on the full conversation, give "
    "your ranked candidate diagnoses, most likely first. Respond ONLY "
    'with JSON: {"ranked_diagnoses": ["Diagnosis A", ...]}.'
)


class OneShotBaseline:
    def __init__(self, llm: BaseLLM, kb: KnowledgeBase):
        self.llm = llm
        self.kb = kb

    def diagnose(self, opening_message: str) -> List[str]:
        diseases = self.kb.all_disease_names()
        messages = [
            LLMMessage(role="system", content=_ONE_SHOT_SYSTEM),
            LLMMessage(
                role="user",
                content=f"Patient message: {opening_message}\n\nCandidate diseases known to you: {diseases}",
            ),
        ]
        result: OneShotDiagnosisResult = self.llm.complete_structured(
            messages, OneShotDiagnosisResult, mock_context={"diseases": diseases},
        )
        return result.ranked_diagnoses


@dataclass
class NaiveMultiTurnState:
    dialogue_text: str = ""
    turn: int = 0


class NaiveMultiTurnBaseline:
    """Multi-turn but WITHOUT Bayesian active-learning question selection --
    the LLM just asks whatever it wants next, unranked and ungrounded."""

    def __init__(self, llm: BaseLLM, kb: KnowledgeBase, max_turns: int = 6):
        self.llm = llm
        self.kb = kb
        self.max_turns = max_turns

    def initialize(self, opening_question: str, opening_response: str) -> NaiveMultiTurnState:
        return NaiveMultiTurnState(
            dialogue_text=f"Doctor: {opening_question}\nPatient: {opening_response}", turn=1,
        )

    def propose_next_question(self, state: NaiveMultiTurnState) -> str:
        messages = [
            LLMMessage(role="system", content=_NAIVE_FOLLOWUP_SYSTEM),
            LLMMessage(role="user", content=f"Conversation so far:\n{state.dialogue_text}"),
        ]
        result: NaiveFollowUpResult = self.llm.complete_structured(
            messages, NaiveFollowUpResult,
            mock_context={"turn": state.turn},
        )
        return result.question or "Can you tell me more about your symptoms?"

    def observe_response(self, state: NaiveMultiTurnState, question: str, response: str) -> NaiveMultiTurnState:
        state.dialogue_text += f"\nDoctor: {question}\nPatient: {response}"
        state.turn += 1
        return state

    def final_diagnosis(self, state: NaiveMultiTurnState) -> List[str]:
        diseases = self.kb.all_disease_names()
        messages = [
            LLMMessage(role="system", content=_NAIVE_FINAL_SYSTEM),
            LLMMessage(
                role="user",
                content=f"Conversation so far:\n{state.dialogue_text}\n\nCandidate diseases known to you: {diseases}",
            ),
        ]
        result: OneShotDiagnosisResult = self.llm.complete_structured(
            messages, OneShotDiagnosisResult, mock_context={"diseases": diseases},
        )
        return result.ranked_diagnoses
