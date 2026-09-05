"""
Section 3.4 "Evaluation Matrix":
  3.4.1 Accuracy   -- 5-point diagnosis-accuracy score (paper: 3 human
                       biomedical experts). Here: LLM-as-judge with the
                       same rubric (Appendix A.1), an explicit,
                       documented APPROXIMATION of the paper's human
                       evaluation.
  3.4.2 Entropy    -- H_t trajectory across turns; reduction in entropy
                       indicates increased diagnostic confidence. This
                       part IS faithfully reproduced: it directly reuses
                       askpatients.bayesian.entropy.shannon_entropy, the
                       same function driving question selection.
  3.4.3 Human-Centric -- accessibility / empathy / relevant-response /
                       relationship-fostering (paper: 5 non-medical human
                       raters, Appendix A.2). Here: LLM-as-judge
                       APPROXIMATION.
"""
from __future__ import annotations

from typing import List

from askpatients.llm.base import BaseLLM, LLMMessage
from askpatients.medical.schema import DiagnosisJudgeResult, HumanCentricJudgeResult

_DIAGNOSIS_JUDGE_SYSTEM = (
    "You are a biomedical expert rating a predicted diagnosis against the ground "
    "truth, using this exact 5-point rubric (Note: there is no score of 1): "
    "0 = completely unrelated to the actual diagnosis; "
    "2 = related, but unlikely to be useful; "
    "3 = closely related, may still be helpful; "
    "4 = very close, minor difference but clinically similar; "
    "5 = exact match with the ground truth diagnosis. "
    'Respond ONLY with JSON: {"score": int, "rationale": "..."}.'
)

_HUMAN_CENTRIC_JUDGE_SYSTEM = (
    "You are a non-medical patient rating a doctor-patient consultation transcript "
    "on four dimensions, each on a 1-5 scale: "
    "accessibility (how clear/jargon-free the language was for a non-expert), "
    "empathy (how compassionate and understanding the doctor's tone was), "
    "relevant_response (whether the doctor directly answered follow-ups before moving on), "
    "fosters_relationship (whether the doctor built trust and rapport). "
    'Respond ONLY with JSON: {"accessibility": int, "empathy": int, '
    '"relevant_response": int, "fosters_relationship": int}.'
)


def judge_diagnosis_accuracy(llm: BaseLLM, predicted: str, ground_truth: str) -> DiagnosisJudgeResult:
    messages = [
        LLMMessage(role="system", content=_DIAGNOSIS_JUDGE_SYSTEM),
        LLMMessage(
            role="user",
            content=f"Ground truth diagnosis: {ground_truth}\nPredicted diagnosis: {predicted}",
        ),
    ]
    return llm.complete_structured(
        messages, DiagnosisJudgeResult,
        mock_context={"predicted": predicted, "ground_truth": ground_truth},
    )


def judge_human_centric_quality(llm: BaseLLM, transcript_text: str) -> HumanCentricJudgeResult:
    messages = [
        LLMMessage(role="system", content=_HUMAN_CENTRIC_JUDGE_SYSTEM),
        LLMMessage(role="user", content=f"Transcript:\n{transcript_text}"),
    ]
    return llm.complete_structured(messages, HumanCentricJudgeResult, mock_context={})


def entropy_reduction(entropy_trajectory: List[float]) -> float:
    """H_1 - H_final; positive values indicate the dialogue became more
    confident over time (Section 3.4.2 / Figure 4)."""
    if len(entropy_trajectory) < 2:
        return 0.0
    return entropy_trajectory[0] - entropy_trajectory[-1]
