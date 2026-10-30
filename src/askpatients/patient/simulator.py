"""Extractive patient simulation: the LLM selects facts and code renders them.

This restricts the paper's free-form role-play to avoid invented clinical
findings. Profile extraction and fact relevance still require human review.
"""
from __future__ import annotations

import json
from typing import Annotated
from pydantic import BaseModel, Field, StrictInt, ValidationInfo, model_validator
from askpatients.llm.base import BaseLLM, LLMMessage
from askpatients.patient.profile import PatientProfile


class PatientAnswerResult(BaseModel):
    fact_indices: list[Annotated[StrictInt, Field(ge=0)]]
    unknown: bool

    @model_validator(mode="after")
    def valid_indices(self, info: ValidationInfo):
        count = (info.context or {}).get('fact_count')
        if count is not None and any(i >= count for i in self.fact_indices):
            raise ValueError('Selected fact index is outside the supplied profile')
        if len(set(self.fact_indices)) != len(self.fact_indices):
            raise ValueError('Selected fact indices must be unique')
        return self


_SYSTEM = (
    "Select recorded patient facts relevant to the doctor's latest question. "
    "Only the numbered facts are evidence; questions and hypothetical answers "
    "are not facts. Set unknown=true when any requested detail is not recorded. "
    "Absence of a symptom from the profile does NOT mean the patient denies it. "
    "For example, bloating with no stool information means diarrhea is unknown. "
    "Select no facts if none are relevant. Return ONLY JSON: "
    '{"fact_indices": [0], "unknown": true}. Indices start at zero.'
)


class PatientSimulator:
    def __init__(self, llm: BaseLLM, profile: PatientProfile):
        self.llm = llm
        self.profile = profile

    def answer(self, doctor_question: str, dialogue_so_far: str,
               candidate_responses_hint: list[str] | None = None) -> str:
        # Hypothetical response hints are intentionally not patient evidence.
        facts = self.profile.clinical_facts
        messages = [
            LLMMessage(role="system", content=_SYSTEM),
            LLMMessage(role="user", content=(
                f"Numbered recorded facts:\n{json.dumps(dict(enumerate(facts)), ensure_ascii=False)}"
                f"\nConversation (context only):\n{dialogue_so_far}"
                f"\nDoctor asks: {doctor_question}")),
        ]
        result = self.llm.complete_structured(
            messages, PatientAnswerResult,
            mock_context={"fact_count": len(facts)},
            validation_context={"fact_count": len(facts)},
        )
        statements = [facts[i]['statement'].strip().rstrip('.') for i in result.fact_indices]
        answer = "What I can tell you is: " + "; ".join(statements) + "." if statements else ""
        if result.unknown or not statements:
            answer += " I don't know about the other details you asked." if statements else "I don't know."
        gt = self.profile.ground_truth_diagnosis.strip().casefold()
        if gt and gt in answer.casefold():
            return "I don't know."
        return answer

    def opening_statement(self) -> str:
        return self.profile.presenting_complaint
