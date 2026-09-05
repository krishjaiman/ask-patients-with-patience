"""
Section 2.7 "Human-Centric Communication": rewrites the entropy-selected
question q*_{t+1} into clear, empathetic, accessible language -- simplified
terminology, yes/no or multiple-choice framing where possible, and
concrete descriptive examples instead of clinical jargon, exactly as the
three worked examples in the paper describe.

This step runs strictly AFTER question selection (Eq. 5): the
information-theoretic content of which question to ask is decided first
by the Bayesian active-learning step; this module only changes surface
phrasing, never which underlying question was chosen.
"""
from __future__ import annotations

from askpatients.llm.base import BaseLLM, LLMMessage

_HUMANIZE_SYSTEM = (
    "You are Dr.APP, a warm and empathetic medical assistant. Rewrite the "
    "given clinical follow-up question for a patient without a medical "
    "background. Simplify medical terminology and jargon into everyday "
    "language. Where natural, frame the question as simple yes/no or "
    "multiple-choice, and use concrete descriptive examples rather than "
    "abstract clinical terms (e.g. instead of 'have you had any dietary "
    "triggers', ask about specific foods like milk or ice cream). Keep a "
    "warm, understanding, conversational tone, as if talking with a "
    "trusted friend. Respond with ONLY the rewritten question/message, no "
    "preamble, no quotation marks."
)


class HumanCentricRewriter:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def rewrite(self, clinical_question: str, dialogue_text: str, empathetic_opener: bool = True) -> str:
        messages = [
            LLMMessage(role="system", content=_HUMANIZE_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    f"Dialogue so far:\n{dialogue_text}\n\n"
                    f"Clinically-selected next question: \"{clinical_question}\"\n\n"
                    f"{'Include a brief empathetic acknowledgement before the question.' if empathetic_opener else ''}\n"
                    "Rewrite this for the patient now."
                ),
            ),
        ]
        resp = self.llm.complete(messages)
        text = resp.text.strip()
        return text if text else clinical_question
