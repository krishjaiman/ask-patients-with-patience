"""
Deterministic mock LLM.

This lets the ENTIRE AskPatients pipeline (symptom mapping, question
generation, response simulation, conditional-probability estimation,
Bayesian update, entropy selection, rewriting, patient simulation) run
end-to-end with no network access and no API key, which is required
for the unit tests / smoke test and for offline demos.

The mock is intentionally simple and deterministic (seeded from the
prompt content) -- it is NOT a stand-in for the paper's actual
reasoning quality, only for the data *shapes* the rest of the pipeline
needs so the Bayesian / entropy math can be exercised end-to-end.
Swapping `llm.provider` to "openai" or "anthropic" in the config
replaces this mock with a real model without touching any other code.

Callers pass `mock_context` (a plain dict of the entities relevant to
the current call -- e.g. the candidate disease names, the candidate
question text, ...) alongside the natural-language prompt so the mock
can build meaningful, schema-conformant dummy data instead of trying
to parse free text.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel

from .base import BaseLLM, LLMMessage, LLMResponse

T = TypeVar("T", bound=BaseModel)


def _stable_seed(*parts: str) -> random.Random:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


class MockLLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(model="mock-llm", **{k: v for k, v in kwargs.items() if k != "model"})

    def _complete(self, messages: List[LLMMessage]) -> LLMResponse:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        rnd = _stable_seed(self.model, last_user[:200])
        return LLMResponse(text=f"[mock free-text reply #{rnd.randint(0, 10**6)}]")

    def complete_structured(self, messages: List[LLMMessage], schema: Type[T],
                             retries: int = 2, mock_context: Optional[dict] = None,
                             validation_context: Optional[dict] = None) -> T:
        prompt_key = "||".join(m.content for m in messages)
        rnd = _stable_seed(self.model, prompt_key)
        result = _mock_fill_schema(schema, rnd, mock_context or {})
        return schema.model_validate(result.model_dump(), context=validation_context)


def _dirichlet_like(rnd: random.Random, n: int) -> List[float]:
    """Deterministic pseudo-random simplex sample without a numpy dependency here."""
    xs = [-1.0 * (rnd.random() ** 1.5) for _ in range(n)]
    weights = [math.exp(-x) for x in xs]
    total = sum(weights)
    return [w / total for w in weights]


def _mock_fill_schema(schema: Type[T], rnd: random.Random, ctx: dict) -> T:
    name = schema.__name__

    if name == "SymptomMappingResult":
        pool = ctx.get("symptom_vocab") or [
            "abdominal pain", "nausea", "fatigue", "headache", "fever",
            "dizziness", "cough", "nasal congestion",
        ]
        pool = list(pool)
        rnd.shuffle(pool)
        k = min(len(pool), rnd.randint(1, 3))
        return schema(symptoms=pool[:k])

    if name == "DiseaseProbabilityResult":
        diseases = ctx.get("diseases", [])
        if not diseases:
            return schema(probabilities={})
        probs = _dirichlet_like(rnd, len(diseases))
        return schema(probabilities=dict(zip(diseases, probs)))

    if name == "CandidateQuestionsResult":
        k = ctx.get("num_questions", 5)
        templates = [
            "Can you describe when the {sym} started and how it has changed?",
            "Have you noticed anything that makes the {sym} better or worse?",
            "Do you have any other symptoms along with the {sym}?",
            "How would you rate the severity of the {sym} on a typical day?",
            "Has anything in your diet, activity, or environment changed recently?",
            "Have you taken any medication or treatment for this so far?",
        ]
        sym = ctx.get("focus_symptom", "symptoms")
        qs = [t.format(sym=sym) for t in templates][:k]
        return schema(questions=qs)

    if name == "PossibleResponsesResult":
        n = rnd.randint(ctx.get("min_l", 2), ctx.get("max_l", 5))
        pool = [
            "Yes, that's right.", "No, not really.", "It comes and goes.",
            "It's been constant.", "Only sometimes, mostly in the morning.",
            "It gets worse after eating.", "I'm not sure, maybe a little.",
            "It started a few days ago.",
        ]
        rnd.shuffle(pool)
        return schema(responses=pool[:n])

    if name == "ConditionalProbabilityResult":
        diseases = ctx.get("diseases", [])
        if not diseases:
            return schema(probabilities={})
        vals = [round(0.05 + 0.9 * rnd.random(), 4) for _ in diseases]
        return schema(probabilities=dict(zip(diseases, vals)))

    if name == "PatientProfileLLMResult":
        facts = [
            {"category": "symptom", "statement": line[9:].strip(), "evidence": line[9:].strip()}
            for line in ctx.get("transcript", "").splitlines()
            if line.startswith("Patient: ") and line[9:].strip()
        ]
        return schema(
            age=None,
            gender="unspecified",
            intention=rnd.choice([
                "wants a quick answer", "anxious and wants reassurance",
                "curious about the cause", "wants to know if this is serious",
            ]),
            personality=rnd.choice([
                "terse and to the point", "chatty and detail-oriented",
                "anxious", "calm and matter-of-fact",
            ]),
            background_details=[],
            clinical_facts=facts,
        )

    if name == "PatientAnswerResult":
        return schema(fact_indices=[0] if ctx.get('fact_count', 0) else [], unknown=True)

    if name == "NaiveFollowUpResult":
        pool = [
            "Can you tell me more about your symptoms?",
            "How long has this been going on?",
            "Is there anything that makes it better or worse?",
            "Do you have any other symptoms?",
        ]
        return schema(question=rnd.choice(pool))

    if name == "OneShotDiagnosisResult":
        diseases = ctx.get("diseases") or ["Unspecified condition"]
        ranked = list(diseases)
        rnd.shuffle(ranked)
        return schema(ranked_diagnoses=ranked[: min(5, len(ranked))])

    if name == "DiagnosisJudgeResult":
        predicted = (ctx.get("predicted") or "").strip().lower()
        truth = (ctx.get("ground_truth") or "").strip().lower()
        if predicted and truth and predicted == truth:
            score = 5
        elif predicted and truth and (predicted in truth or truth in predicted):
            score = 4
        else:
            score = rnd.choice([0, 2, 3])
        return schema(score=score, rationale="[mock heuristic string-overlap judge]")

    if name == "HumanCentricJudgeResult":
        base = rnd.randint(3, 5)
        return schema(
            accessibility=base, empathy=max(1, base - rnd.randint(0, 1)),
            relevant_response=base, fosters_relationship=max(1, base - rnd.randint(0, 1)),
        )

    raise NotImplementedError(f"MockLLM has no filler for schema {name}")
