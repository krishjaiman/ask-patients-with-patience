"""
Ties everything together for Section 3 experiments: runs one-shot,
naive multi-turn, and Dr.APP over a set of ReMeDi cases, scores each
with the (approximated) evaluation metrics, and writes CSV/JSON
results plus the entropy trajectory data used for Figure-4-style plots.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from askpatients.config import Config
from askpatients.data.remedi_loader import ReMeDiCase
from askpatients.dialogue.app_agent import DrAPP
from askpatients.dialogue.baselines import NaiveMultiTurnBaseline, OneShotBaseline
from askpatients.dialogue.orchestrator import DialogueOrchestrator, NaiveMultiTurnOrchestrator
from askpatients.evaluation.metrics import (
    judge_diagnosis_accuracy, judge_human_centric_quality, entropy_reduction,
)
from askpatients.llm.base import BaseLLM
from askpatients.medical.knowledge_base import KnowledgeBase
from askpatients.patient.profile import build_profile_from_case
from askpatients.patient.simulator import PatientSimulator
from askpatients.rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    case_id: str
    specialty: str
    ground_truth: str
    method: str
    predicted_diagnosis: str
    accuracy_score: int
    entropy_trajectory: List[float] = field(default_factory=list)
    entropy_reduction: float = 0.0
    accessibility: Optional[int] = None
    empathy: Optional[int] = None
    relevant_response: Optional[int] = None
    fosters_relationship: Optional[int] = None


class EvaluationRunner:
    def __init__(self, config: Config, llm: BaseLLM, judge_llm: BaseLLM,
                 kb: KnowledgeBase, retriever: Retriever):
        self.config = config
        self.llm = llm
        self.judge_llm = judge_llm
        self.kb = kb
        self.retriever = retriever

    # ------------------------------------------------------------ one case
    def _build_patient(self, case: ReMeDiCase) -> PatientSimulator:
        profile = build_profile_from_case(
            self.llm, case_id=case.case_id, transcript=case.full_transcript_text(),
            presenting_complaint=case.opening_patient_turn(),
            symptoms=[], ground_truth_diagnosis=case.ground_truth_diagnosis,
        )
        return PatientSimulator(self.llm, profile)

    def run_one_shot(self, case: ReMeDiCase) -> CaseResult:
        patient = self._build_patient(case)
        baseline = OneShotBaseline(self.llm, self.kb)
        ranked = baseline.diagnose(patient.opening_statement())
        predicted = ranked[0] if ranked else "Unknown"
        judged = judge_diagnosis_accuracy(self.judge_llm, predicted, case.ground_truth_diagnosis)
        return CaseResult(
            case_id=case.case_id, specialty=case.specialty, ground_truth=case.ground_truth_diagnosis,
            method="one_shot", predicted_diagnosis=predicted, accuracy_score=judged.score,
        )

    def run_naive_multi_turn(self, case: ReMeDiCase) -> CaseResult:
        patient = self._build_patient(case)
        baseline = NaiveMultiTurnBaseline(self.llm, self.kb, max_turns=self.config.algo.max_dialogue_turns)
        orch = NaiveMultiTurnOrchestrator(baseline, max_turns=self.config.algo.max_dialogue_turns)
        result = orch.run(case.case_id, case.opening_doctor_turn(), patient, case.ground_truth_diagnosis)
        predicted = result["ranked_diagnoses"][0] if result["ranked_diagnoses"] else "Unknown"
        judged = judge_diagnosis_accuracy(self.judge_llm, predicted, case.ground_truth_diagnosis)
        transcript_text = "\n".join(
            f"Doctor: {t['question']}\nPatient: {t['response']}" for t in result["transcript"]
        )
        hc = judge_human_centric_quality(self.judge_llm, transcript_text)
        return CaseResult(
            case_id=case.case_id, specialty=case.specialty, ground_truth=case.ground_truth_diagnosis,
            method="naive_multi_turn", predicted_diagnosis=predicted, accuracy_score=judged.score,
            accessibility=hc.accessibility, empathy=hc.empathy,
            relevant_response=hc.relevant_response, fosters_relationship=hc.fosters_relationship,
        )

    def run_app(self, case: ReMeDiCase) -> CaseResult:
        patient = self._build_patient(case)
        app = DrAPP(self.llm, self.kb, self.retriever, self.config.algo)
        orch = DialogueOrchestrator(app, max_turns=self.config.algo.max_dialogue_turns)
        result = orch.run(case.case_id, case.opening_doctor_turn(), patient, case.ground_truth_diagnosis)
        judged = judge_diagnosis_accuracy(self.judge_llm, result.final_diagnosis, case.ground_truth_diagnosis)
        transcript_text = "\n".join(
            f"Doctor: {t.human_question}\nPatient: {t.patient_response}" for t in result.turns
        )
        hc = judge_human_centric_quality(self.judge_llm, transcript_text)
        return CaseResult(
            case_id=case.case_id, specialty=case.specialty, ground_truth=case.ground_truth_diagnosis,
            method="app", predicted_diagnosis=result.final_diagnosis, accuracy_score=judged.score,
            entropy_trajectory=result.entropy_trajectory,
            entropy_reduction=entropy_reduction(result.entropy_trajectory),
            accessibility=hc.accessibility, empathy=hc.empathy,
            relevant_response=hc.relevant_response, fosters_relationship=hc.fosters_relationship,
        )

    # ---------------------------------------------------------------- all
    def run_all(self, cases: List[ReMeDiCase], methods: List[str]) -> List[CaseResult]:
        results: List[CaseResult] = []
        dispatch = {
            "one_shot": self.run_one_shot,
            "naive_multi_turn": self.run_naive_multi_turn,
            "app": self.run_app,
        }
        for i, case in enumerate(cases):
            for method in methods:
                logger.info("Case %d/%d [%s] method=%s", i + 1, len(cases), case.case_id, method)
                try:
                    results.append(dispatch[method](case))
                except Exception:
                    logger.exception("Failed on case %s method %s", case.case_id, method)
        return results

    @staticmethod
    def write_csv(results: List[CaseResult], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(asdict(results[0]).keys()) if results else []
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = asdict(r)
                row["entropy_trajectory"] = json.dumps(row["entropy_trajectory"])
                writer.writerow(row)

    @staticmethod
    def write_json(results: List[CaseResult], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    @staticmethod
    def summarize(results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
        by_method: Dict[str, List[CaseResult]] = {}
        for r in results:
            by_method.setdefault(r.method, []).append(r)
        summary = {}
        for method, rs in by_method.items():
            n = len(rs)
            summary[method] = {
                "n_cases": n,
                "mean_accuracy": sum(r.accuracy_score for r in rs) / n if n else 0.0,
                "mean_entropy_reduction": sum(r.entropy_reduction for r in rs) / n if n else 0.0,
                "mean_accessibility": _safe_mean([r.accessibility for r in rs]),
                "mean_empathy": _safe_mean([r.empathy for r in rs]),
                "mean_relevant_response": _safe_mean([r.relevant_response for r in rs]),
                "mean_fosters_relationship": _safe_mean([r.fosters_relationship for r in rs]),
            }
        return summary


def _safe_mean(values: List[Optional[int]]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
