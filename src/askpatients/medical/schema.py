"""
Structured-output schemas for every LLM call in the Dr.APP pipeline.

Each schema corresponds to one step of Figure 2 / Section 2 of the paper.
Using pydantic here means every LLM call is forced through
`BaseLLM.complete_structured`, which validates the shape of the model's
output before any of it touches the Bayesian math in
`askpatients.bayesian`. This is what keeps "the LLM's opinion" and
"the paper's probabilistic model" cleanly separated: the LLM only ever
supplies *inputs* (symptom lists, a prior distribution, candidate
questions, plausible responses, and per-disease response likelihoods);
every subsequent probability computation (Eq. 1-5) is deterministic
Python, not an LLM call.
"""
from __future__ import annotations

from typing import Dict, List, Annotated, Literal

from pydantic import BaseModel, Field, field_validator, ValidationInfo


class ProbabilityMap(BaseModel):
    probabilities: Dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(min_length=1)

    @field_validator('probabilities')
    @classmethod
    def check_coverage(cls, values, info: ValidationInfo):
        expected = (info.context or {}).get('diseases')
        if expected is not None and set(values) != set(expected):
            raise ValueError('Return exactly one probability for every requested disease')
        return values


class SymptomMappingResult(BaseModel):
    """Section 2.2 'Mapping into MSD': dialogue -> symptom list S_MSD."""
    symptoms: List[str] = Field(default_factory=list)


class DiseaseProbabilityResult(ProbabilityMap):
    """
    Section 2.3 'Diagnosis Probability Prediction': the initial LLM-proposed
    disease probability distribution P_t(D) (Eq. 1), before it is
    renormalized by askpatients.bayesian.probability.normalize_distribution.
    """
    @field_validator('probabilities')
    @classmethod
    def nonzero_mass(cls, values):
        if sum(values.values()) <= 0:
            raise ValueError('Disease weights must have positive total mass')
        return values


class CandidateQuestionsResult(BaseModel):
    """Section 2.4 'Question Generation': candidate pool Q_{t+1} = {q_1..q_K}."""
    questions: List[str] = Field(default_factory=list)

    @field_validator("questions")
    @classmethod
    def _dedup(cls, v: List[str]) -> List[str]:
        seen, out = set(), []
        for q in v:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(q.strip())
        if not out:
            raise ValueError('At least one nonempty question is required')
        return out


class PossibleResponsesResult(BaseModel):
    """
    Section 2.5 'One-more Step Thinking': plausible patient responses
    R_k = {r_k^1 .. r_k^L} for a single candidate question.
    """
    responses: List[str]

    @field_validator('responses')
    @classmethod
    def distinct_answers(cls, values, info: ValidationInfo):
        values = list({v.strip().casefold(): v.strip() for v in values if v.strip()}.values())
        context = info.context or {}
        if not context.get('min_responses', 2) <= len(values) <= context.get('max_responses', 5):
            raise ValueError('Return the requested number of distinct nonempty answers')
        return values


class OneShotDiagnosisResult(BaseModel):
    """Baseline (Fig. 1(a)): ranked list of candidate diagnoses generated
    from the initial patient input alone, with no follow-up questions."""
    ranked_diagnoses: List[str] = Field(default_factory=list)


class NaiveFollowUpResult(BaseModel):
    """Baseline (Fig. 1(b)): a single follow-up question generated directly
    by prompting the LLM, with no Bayesian active-learning question
    selection -- used as the "mechanical multi-turn" comparison point."""
    question: str = ""


class ConditionalProbabilityResult(ProbabilityMap):
    """
    Section 2.5: for ONE specific response r_k^l, the LLM-estimated
    likelihood P(r_k^l | Gamma(d_i)) for each disease d_i under
    consideration. Values need not sum to 1 across diseases (they are
    independent likelihoods, not a distribution) -- this mirrors Eq. 2,
    where P(r_k^l | Gamma(d_i)) is a per-disease conditional likelihood.
    """
    probabilities: Dict[str, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]] = Field(min_length=1)


class DiagnosisJudgeResult(BaseModel):
    """
    Section 3.4.1 / Appendix A.1: LLM-as-judge approximation of the
    paper's 3-expert 5-point diagnosis-accuracy survey (0 = unrelated,
    2 = related but unlikely useful, 3 = closely related, 4 = very
    close, 5 = exact match; note the paper's scale has no score of 1).
    This is an IMPLEMENTATION APPROXIMATION -- the paper used human
    biomedical experts, which this codebase cannot reproduce.
    """
    score: Literal[0, 2, 3, 4, 5]
    rationale: str = ""


class HumanCentricJudgeResult(BaseModel):
    """
    Section 3.4.3 / Appendix A.2: LLM-as-judge approximation of the
    paper's 5-participant human-centric survey (accessibility, empathy,
    relevant response rate, relationship fostering), each 1-5.
    APPROXIMATION -- the paper used real non-medical human raters.
    """
    accessibility: int = Field(ge=1, le=5)
    empathy: int = Field(ge=1, le=5)
    relevant_response: int = Field(ge=1, le=5)
    fosters_relationship: int = Field(ge=1, le=5)
