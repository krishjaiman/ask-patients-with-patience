"""
Section 3.2: the simulated patient agent. Answers doctor questions using
ONLY the constructed persona (never the original transcript or the
ground-truth diagnosis), and is instructed to "reasonably incorporate
daily life details that align with the patient's personality and
background," mirroring the paper's patient-simulation prompt.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from askpatients.llm.base import BaseLLM, LLMMessage
from askpatients.patient.profile import PatientProfile


class PatientAnswerResult(BaseModel):
    answer: str = ""
    reveals_diagnosis: bool = False  # self-check flag; True triggers a scrub


_PATIENT_SYSTEM_TEMPLATE = (
    "{persona}\n\n"
    "You are role-playing as this patient in a medical consultation. Answer the "
    "doctor's question naturally and in character, using only what a real patient "
    "in your situation would know or say. Reasonably incorporate daily life "
    "details that align with your personality and background. Do NOT name or "
    "hint at any medical diagnosis, disease name, or clinical term -- you do not "
    "know what is wrong with you. Do NOT mention that you are an AI or a "
    "simulation. Respond ONLY with JSON: "
    '{{"answer": "...", "reveals_diagnosis": false}}.'
)


class PatientSimulator:
    def __init__(self, llm: BaseLLM, profile: PatientProfile):
        self.llm = llm
        self.profile = profile

    def answer(self, doctor_question: str, dialogue_so_far: str,
               candidate_responses_hint: list[str] | None = None) -> str:
        """
        Answers `doctor_question`. If `candidate_responses_hint` is provided
        (e.g. when reusing the plausible-response pool already generated for
        Bayesian scoring), the simulator is nudged to pick/adapt one of those
        for consistency with the active-learning step, but is not forced to
        -- it can still answer freely in character.
        """
        hint = ""
        if candidate_responses_hint:
            hint = (
                "\nFor reference, here are some example ways a patient might phrase "
                f"an answer (you may use one as inspiration, or answer differently "
                f"if it fits your persona better): {candidate_responses_hint}"
            )
        system = _PATIENT_SYSTEM_TEMPLATE.format(persona=self.profile.as_persona_text())
        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(
                role="user",
                content=f"Conversation so far:\n{dialogue_so_far}\n\nDoctor asks: {doctor_question}{hint}",
            ),
        ]
        result: PatientAnswerResult = self.llm.complete_structured(
            messages, PatientAnswerResult,
            mock_context={"candidate_responses": candidate_responses_hint},
        )
        answer = result.answer.strip()
        if not answer:
            answer = "I'm not sure how to describe it, but it's been bothering me."
        return self._scrub_diagnosis_leak(answer)

    def opening_statement(self) -> str:
        return self.profile.presenting_complaint

    def _scrub_diagnosis_leak(self, answer: str) -> str:
        gt = self.profile.ground_truth_diagnosis.lower()
        if gt and gt in answer.lower():
            return "I'm not sure exactly what it is, but here's what I've noticed: " + \
                answer.lower().replace(gt, "this")
        return answer
