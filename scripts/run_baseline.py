#!/usr/bin/env python
"""
Runs a baseline (one-shot or naive multi-turn) on a single case, for
comparison against scripts/run_app.py.

Usage:
    python scripts/run_baseline.py --mock --method one_shot \
        --case data/remedi/synthetic_sample.jsonl --case-id SYN-001
    python scripts/run_baseline.py --mock --method naive_multi_turn \
        --case data/remedi/synthetic_sample.jsonl --case-id SYN-001
"""
from __future__ import annotations

import argparse
import logging

from _common import add_common_args, build_knowledge_base, build_llm_from_config, load_config
from askpatients.data.remedi_loader import load_remedi_raw
from askpatients.dialogue.baselines import NaiveMultiTurnBaseline, OneShotBaseline
from askpatients.dialogue.orchestrator import NaiveMultiTurnOrchestrator
from askpatients.patient.profile import build_profile_from_case
from askpatients.patient.simulator import PatientSimulator

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--case", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--method", choices=["one_shot", "naive_multi_turn"], required=True)
    args = parser.parse_args()

    cfg = load_config(args)
    cases = load_remedi_raw(args.case)
    case = next((c for c in cases if c.case_id == args.case_id), cases[0]) if args.case_id else cases[0]
    logger.info("Running case %s (ground truth: %s) with method=%s",
                case.case_id, case.ground_truth_diagnosis, args.method)

    llm = build_llm_from_config(cfg)
    kb = build_knowledge_base(cfg, args.kb)
    profile = build_profile_from_case(
        llm, case_id=case.case_id, transcript=case.full_transcript_text(),
        presenting_complaint=case.opening_patient_turn(), symptoms=[],
        ground_truth_diagnosis=case.ground_truth_diagnosis,
    )
    patient = PatientSimulator(llm, profile)

    if args.method == "one_shot":
        baseline = OneShotBaseline(llm, kb)
        ranked = baseline.diagnose(patient.opening_statement())
        print(f"Patient: {patient.opening_statement()}")
        print(f"Ranked diagnoses (one-shot, no follow-up): {ranked}")
        print(f"Ground truth: {case.ground_truth_diagnosis}")
    else:
        baseline = NaiveMultiTurnBaseline(llm, kb, max_turns=cfg.algo.max_dialogue_turns)
        orch = NaiveMultiTurnOrchestrator(baseline, max_turns=cfg.algo.max_dialogue_turns)
        result = orch.run(case.case_id, case.opening_doctor_turn(), patient, case.ground_truth_diagnosis)
        print(f"Doctor: {case.opening_doctor_turn()}")
        print(f"Patient: {patient.opening_statement()}")
        for t in result["transcript"]:
            print(f"\n[Turn {t['turn']}] Doctor: {t['question']}")
            print(f"Patient: {t['response']}")
        print(f"\nRanked diagnoses (naive multi-turn, no Bayesian selection): "
              f"{result['ranked_diagnoses']}")
        print(f"Ground truth: {case.ground_truth_diagnosis}")


if __name__ == "__main__":
    main()
