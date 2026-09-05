#!/usr/bin/env python
"""
Runs the full Dr.APP pipeline on a single case against the patient
simulator, and prints the resulting dialogue, per-turn entropy, and
final diagnosis.

Usage (fully offline):
    python scripts/run_app.py --mock --case data/remedi/synthetic_sample.jsonl --case-id SYN-001

With a real LLM (after configuring .env):
    python scripts/run_app.py --config configs/default.yaml \
        --case data/processed/remedi_eval_100.jsonl --case-id 12345
"""
from __future__ import annotations

import argparse
import json
import logging

from _common import (
    add_common_args, build_knowledge_base, build_llm_from_config, build_retriever, load_config,
)
from askpatients.data.remedi_loader import ReMeDiCase, load_remedi_raw
from askpatients.dialogue.app_agent import DrAPP
from askpatients.dialogue.orchestrator import DialogueOrchestrator
from askpatients.patient.profile import build_profile_from_case
from askpatients.patient.simulator import PatientSimulator

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--case", required=True, help="Path to a ReMeDi-schema .jsonl file")
    parser.add_argument("--case-id", default=None, help="Which case_id to run (default: first case)")
    parser.add_argument("--turns", type=int, default=None, help="Override algo.max_dialogue_turns")
    args = parser.parse_args()

    cfg = load_config(args)
    if args.turns:
        cfg.algo.max_dialogue_turns = args.turns

    cases = load_remedi_raw(args.case)
    if not cases:
        raise SystemExit(f"No cases found in {args.case}")
    case: ReMeDiCase = next((c for c in cases if c.case_id == args.case_id), cases[0]) \
        if args.case_id else cases[0]
    logger.info("Running case %s (ground truth: %s)", case.case_id, case.ground_truth_diagnosis)

    llm = build_llm_from_config(cfg)
    kb = build_knowledge_base(cfg, args.kb)
    retriever = build_retriever(cfg, kb)

    profile = build_profile_from_case(
        llm, case_id=case.case_id, transcript=case.full_transcript_text(),
        presenting_complaint=case.opening_patient_turn(), symptoms=[],
        ground_truth_diagnosis=case.ground_truth_diagnosis,
    )
    patient = PatientSimulator(llm, profile)

    app = DrAPP(llm, kb, retriever, cfg.algo)
    orchestrator = DialogueOrchestrator(app, max_turns=cfg.algo.max_dialogue_turns)
    result = orchestrator.run(case.case_id, case.opening_doctor_turn(), patient, case.ground_truth_diagnosis)

    print("=" * 70)
    print(f"Case: {case.case_id}   Ground truth: {case.ground_truth_diagnosis}")
    print("=" * 70)
    print(f"Doctor: {case.opening_doctor_turn()}")
    print(f"Patient: {patient.opening_statement()}")
    for t in result.turns:
        print(f"\n--- Turn {t.turn} (H before ask: prior top={t.top_diagnosis_before!r} "
              f"p={t.top_diagnosis_prob_before:.3f}) ---")
        print(f"[clinical question selected by min-entropy]: {t.clinical_question}")
        print(f"Dr.APP: {t.human_question}")
        print(f"Patient: {t.patient_response}")
        print(f"Entropy after update: {t.entropy_after:.4f}")

    print("\n" + "=" * 70)
    print(f"Final diagnosis: {result.final_diagnosis} (p={result.final_diagnosis_prob:.3f})")
    print(f"Entropy trajectory: {[round(h, 4) for h in result.entropy_trajectory]}")
    print("Final distribution:")
    for d, p in sorted(result.final_distribution.items(), key=lambda kv: -kv[1]):
        print(f"  {d}: {p:.4f}")


if __name__ == "__main__":
    main()
