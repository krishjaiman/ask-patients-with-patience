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
from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from askpatients.llm.base import BaseLLM, LLMMessage


class ClinicalFact(BaseModel):
    category: Literal['symptom', 'timing', 'trigger', 'negative', 'history']
    statement: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description='Exact supporting quote from the transcript')


class PatientProfileLLMResult(BaseModel):
    age: Optional[int] = Field(default=None, ge=0, le=120)
    gender: str = "unspecified"
    intention: str = "wants to understand what's wrong"
    personality: str = "matter-of-fact"
    background_details: List[str] = Field(default_factory=list)
    clinical_facts: List[ClinicalFact] = Field(min_length=1)


@dataclass
class PatientProfile:
    case_id: str
    ground_truth_diagnosis: str
    presenting_complaint: str
    symptoms: List[str]
    age: Optional[int]
    gender: str
    intention: str
    personality: str
    background_details: List[str] = field(default_factory=list)
    clinical_facts: List[dict] = field(default_factory=list)

    def as_persona_text(self) -> str:
        return (
            f"You are a patient, age {self.age}, gender {self.gender}. "
            f"Personality: {self.personality}. Your goal in this conversation: {self.intention}. "
            f"Background: {'; '.join(self.background_details) if self.background_details else 'none specified'}. "
            f"Your presenting complaint: {self.presenting_complaint}. "
            f"Symptoms you are actually experiencing (do not volunteer all of them at once, "
            f"answer naturally as a real patient would): {', '.join(self.symptoms)}. "
            f"Known clinical facts: {self.clinical_facts}. "
            "Do not invent symptoms, triggers, negative findings, age, or history. "
            "If a clinical detail is not in this profile, say you do not know."
        )


_PROFILE_SYSTEM = (
    "Extract a patient persona from the consultation. Preserve ALL patient-confirmed "
    "symptoms, timing, triggers, negative findings and history. A doctor's question "
    "is not an established fact unless the patient confirms it. Do not infer facts "
    "from a diagnosis, invent demographics, or include disease names. Unknown age "
    "must be null; other unknown fields may be unspecified or empty. Each clinical "
    "fact needs an exact supporting quote from the transcript. Respond ONLY with JSON: "
    '{"age": null, "gender": "unspecified", "intention": "...", '
    '"personality": "...", "background_details": [], "clinical_facts": '
    '[{"category": "symptom|timing|trigger|negative|history", '
    '"statement": "...", "evidence": "exact quote"}]}.'
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
        messages, PatientProfileLLMResult, mock_context={"transcript": transcript},
    )
    for fact in result.clinical_facts:
        if fact.evidence not in transcript:
            raise ValueError(f"Profile fact for {case_id} has no matching source quote")
    # Check the generated persona before it is supplied to the simulator. This
    # catches literal label leakage; semantic fidelity still requires review.
    if ground_truth_diagnosis and ground_truth_diagnosis.casefold() in result.model_dump_json().casefold():
        raise ValueError(f"Profile for {case_id} contains the ground-truth diagnosis")
    extracted = [f.statement for f in result.clinical_facts if f.category == 'symptom']
    return PatientProfile(
        case_id=case_id,
        ground_truth_diagnosis=ground_truth_diagnosis,
        presenting_complaint=presenting_complaint,
        symptoms=list(dict.fromkeys(symptoms + extracted)),
        age=result.age,
        gender=result.gender,
        intention=result.intention,
        personality=result.personality,
        background_details=result.background_details,
        clinical_facts=[f.model_dump() for f in result.clinical_facts],
    )
