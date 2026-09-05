import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askpatients.config import Config
from askpatients.data.remedi_loader import load_remedi_raw
from askpatients.dialogue.app_agent import DrAPP
from askpatients.dialogue.baselines import NaiveMultiTurnBaseline, OneShotBaseline
from askpatients.dialogue.orchestrator import DialogueOrchestrator, NaiveMultiTurnOrchestrator
from askpatients.llm.factory import build_llm
from askpatients.medical.knowledge_base import KnowledgeBase
from askpatients.patient.profile import build_profile_from_case
from askpatients.patient.simulator import PatientSimulator
from askpatients.rag.embeddings import MockEmbedder
from askpatients.rag.retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "data" / "knowledge_base" / "sample_diseases.json"
CASES_PATH = ROOT / "data" / "remedi" / "synthetic_sample.jsonl"


def _mock_config() -> Config:
    cfg = Config.load(str(ROOT / "configs" / "default.yaml"), overrides={
        "llm": {"provider": "mock"},
        "rag": {"embedding_provider": "mock"},
        "algo": {"max_dialogue_turns": 2, "num_candidate_questions": 3,
                 "min_responses_per_question": 2, "max_responses_per_question": 3},
    })
    cfg.apply_seed()
    return cfg


def test_full_app_pipeline_runs_end_to_end_offline():
    cfg = _mock_config()
    llm = build_llm(cfg.llm)
    kb = KnowledgeBase.from_json_file(KB_PATH)
    retriever = Retriever.build_from_knowledge_base(kb, MockEmbedder(dim=128),
                                                      chunk_size=cfg.rag.chunk_size,
                                                      overlap=cfg.rag.chunk_overlap)

    cases = load_remedi_raw(CASES_PATH)
    case = cases[0]

    profile = build_profile_from_case(
        llm, case_id=case.case_id, transcript=case.full_transcript_text(),
        presenting_complaint=case.opening_patient_turn(), symptoms=[],
        ground_truth_diagnosis=case.ground_truth_diagnosis,
    )
    patient = PatientSimulator(llm, profile)

    app = DrAPP(llm, kb, retriever, cfg.algo)
    orchestrator = DialogueOrchestrator(app, max_turns=cfg.algo.max_dialogue_turns)
    result = orchestrator.run(case.case_id, case.opening_doctor_turn(), patient,
                               case.ground_truth_diagnosis)

    # Structural assertions -- the point of the smoke test is that the full
    # pipeline (LLM calls -> RAG -> Bayesian math -> entropy selection ->
    # human-centric rewrite -> patient response -> Bayesian update) runs
    # without exceptions and produces well-formed output.
    assert len(result.turns) == cfg.algo.max_dialogue_turns
    assert len(result.entropy_trajectory) == cfg.algo.max_dialogue_turns + 1  # +1 for initial
    assert result.final_diagnosis in result.final_distribution
    assert math.isclose(sum(result.final_distribution.values()), 1.0, rel_tol=1e-4)
    for h in result.entropy_trajectory:
        assert h >= 0.0
    for t in result.turns:
        assert t.clinical_question
        assert t.human_question
        assert t.patient_response
        # The patient must never leak the exact ground-truth diagnosis string.
        assert case.ground_truth_diagnosis.lower() not in t.patient_response.lower()


def test_one_shot_baseline_runs_offline():
    cfg = _mock_config()
    llm = build_llm(cfg.llm)
    kb = KnowledgeBase.from_json_file(KB_PATH)
    cases = load_remedi_raw(CASES_PATH)
    case = cases[0]
    profile = build_profile_from_case(
        llm, case_id=case.case_id, transcript=case.full_transcript_text(),
        presenting_complaint=case.opening_patient_turn(), symptoms=[],
        ground_truth_diagnosis=case.ground_truth_diagnosis,
    )
    patient = PatientSimulator(llm, profile)
    baseline = OneShotBaseline(llm, kb)
    ranked = baseline.diagnose(patient.opening_statement())
    assert isinstance(ranked, list)
    assert len(ranked) > 0


def test_naive_multi_turn_baseline_runs_offline():
    cfg = _mock_config()
    llm = build_llm(cfg.llm)
    kb = KnowledgeBase.from_json_file(KB_PATH)
    cases = load_remedi_raw(CASES_PATH)
    case = cases[0]
    profile = build_profile_from_case(
        llm, case_id=case.case_id, transcript=case.full_transcript_text(),
        presenting_complaint=case.opening_patient_turn(), symptoms=[],
        ground_truth_diagnosis=case.ground_truth_diagnosis,
    )
    patient = PatientSimulator(llm, profile)
    baseline = NaiveMultiTurnBaseline(llm, kb, max_turns=2)
    orch = NaiveMultiTurnOrchestrator(baseline, max_turns=2)
    result = orch.run(case.case_id, case.opening_doctor_turn(), patient, case.ground_truth_diagnosis)
    assert len(result["transcript"]) == 2
    assert isinstance(result["ranked_diagnoses"], list)


def test_dataset_preprocessing_pipeline():
    from askpatients.data.remedi_loader import filter_diagnosis_only, sample_eval_subset
    cases = load_remedi_raw(CASES_PATH)
    diag_only = filter_diagnosis_only(cases)
    assert len(diag_only) == len(cases)  # synthetic sample is all Diagnosis-labeled
    subset = sample_eval_subset(diag_only, n=3, seed=42)
    assert len(subset) == 3
    subset2 = sample_eval_subset(diag_only, n=3, seed=42)
    assert [c.case_id for c in subset] == [c.case_id for c in subset2]  # deterministic
