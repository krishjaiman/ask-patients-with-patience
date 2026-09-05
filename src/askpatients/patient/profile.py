"""
Section 3.2 "Patient Simulator": for each real-world ReMeDi case, the
paper reconstructs "a patient profile ... summarizing structured
characteristics such as symptoms, age, intention and personality"
using an LLM (DeepSeek-v3 in the paper; provider-agnostic here), which
is then used to drive a simulated patient agent that "responds using
only the constructed persona, without access to the original ReMeDi
dialogue."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field

from askpatients.llm.base import BaseLLM, LLMMessage


class PatientProfileLLMResult(BaseModel):
    age: int = 40
    gender: str = "unspecified"
    intention: str = "wants to understand what's wrong"
    personality: str = "matter-of-fact"
    background_details: List[str] = Field(default_factory=list)


@dataclass
class PatientProfile:
    case_id: str
    ground_truth_diagnosis: str
    presenting_complaint: str
    symptoms: List[str]
    age: int
    gender: str
    intention: str
    personality: str
    background_details: List[str] = field(default_factory=list)

    def as_persona_text(self) -> str:
        return (
            f"You are a patient, age {self.age}, gender {self.gender}. "
            f"Personality: {self.personality}. Your goal in this conversation: {self.intention}. "
            f"Background: {'; '.join(self.background_details) if self.background_details else 'none specified'}. "
            f"Your presenting complaint: {self.presenting_complaint}. "
            f"Symptoms you are actually experiencing (do not volunteer all of them at once, "
            f"answer naturally as a real patient would): {', '.join(self.symptoms)}."
        )


_PROFILE_SYSTEM = (
    "You are constructing a realistic patient persona from a real doctor-patient "
    "consultation transcript, for use in a patient simulator. Summarize structured "
    "characteristics: a plausible age, gender, the patient's likely INTENTION in "
    "this consultation (what they want out of it), their PERSONALITY (how they "
    "communicate), and a few background/lifestyle details consistent with the "
    "transcript. Do not include the diagnosis itself in these fields. Respond ONLY "
    'with JSON: {"age": int, "gender": "...", "intention": "...", '
    '"personality": "...", "background_details": ["...", "..."]}.'
)


def build_profile_from_case(llm: BaseLLM, case_id: str, transcript: str,
                             presenting_complaint: str, symptoms: List[str],
                             ground_truth_diagnosis: str) -> PatientProfile:
    """Builds a PatientProfile via LLM summarization of a ReMeDi transcript,
    mirroring Section 3.2's persona-construction step."""
    messages = [
        LLMMessage(role="system", content=_PROFILE_SYSTEM),
        LLMMessage(role="user", content=f"Consultation transcript:\n{transcript}"),
    ]
    result: PatientProfileLLMResult = llm.complete_structured(
        messages, PatientProfileLLMResult, mock_context={},
    )
    return PatientProfile(
        case_id=case_id,
        ground_truth_diagnosis=ground_truth_diagnosis,
        presenting_complaint=presenting_complaint,
        symptoms=symptoms,
        age=result.age,
        gender=result.gender,
        intention=result.intention,
        personality=result.personality,
        background_details=result.background_details,
    )
