#!/usr/bin/env python
"""
Section 3.1 dataset preprocessing: ReMeDi-base -> Diagnosis-only cases -> 100-case
evaluation subset.

Usage:
    python scripts/prepare_dataset.py \
        --raw data/remedi/raw/remedi_base.jsonl \
        --out data/processed/remedi_diagnosis_only.jsonl \
        --eval-out data/processed/remedi_eval_100.jsonl \
        --eval-n 100 --seed 42

For a quick offline smoke test without the real dataset:
    python scripts/prepare_dataset.py --raw data/remedi/synthetic_sample.jsonl \
        --out /tmp/diag_only.jsonl --eval-out /tmp/eval_subset.jsonl --eval-n 5
"""
from __future__ import annotations

import argparse
import json
import logging
import hashlib
from collections import Counter
from pathlib import Path

from _common import _SRC  # noqa: F401  (ensures src is on sys.path)
from askpatients.data.remedi_loader import (
    filter_diagnosis_only, load_remedi_raw, sample_eval_subset,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Raw ReMeDi export (.jsonl or .json)")
    parser.add_argument("--out", required=True, help="Where to write the Diagnosis-only cases")
    parser.add_argument("--eval-out", required=True, help="Where to write the sampled eval subset")
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--filter', choices=['strict', 'single_diagnosis'], default='strict',
        help='single_diagnosis is a documented approximation, not the paper\'s unpublished subset')
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s %(levelname)s: %(message)s")

    logger.info("Loading raw ReMeDi cases from %s", args.raw)
    cases = load_remedi_raw(args.raw)
    logger.info("Loaded %d raw cases", len(cases))

    diagnosis_only = filter_diagnosis_only(cases, args.filter)
    logger.info('%d cases meet filter policy %s (exact paper subset unverified)',
                len(diagnosis_only), args.filter)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in diagnosis_only:
            f.write(json.dumps(c.to_dict()) + "\n")
    logger.info("Wrote diagnosis-only cases to %s", out_path)

    eval_subset = sample_eval_subset(diagnosis_only, n=args.eval_n, seed=args.seed)
    eval_path = Path(args.eval_out)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        for c in eval_subset:
            f.write(json.dumps(c.to_dict()) + "\n")
    logger.info("Wrote %d-case evaluation subset (seed=%d) to %s",
                len(eval_subset), args.seed, eval_path)
    report = {'raw_sha256': hashlib.sha256(Path(args.raw).read_bytes()).hexdigest(),
        'raw_cases': len(cases), 'filter_policy': args.filter,
        'cases_with_diagnosis_action': sum('Diagnosis' in c.doctor_actions() for c in cases),
        'strict_diagnosis_only': len(filter_diagnosis_only(cases)),
        'single_diagnosis': len(filter_diagnosis_only(cases, 'single_diagnosis')),
        'filtered_cases': len(diagnosis_only), 'sampled_cases': len(eval_subset),
        'seed': args.seed, 'selected_ids': [c.case_id for c in eval_subset],
        'exact_paper_subset_verified': False,
        'notes': 'Official release is Chinese. Labels require clinical review and terminology alignment; no translation is performed.'}
    eval_path.with_suffix('.audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if len(eval_subset) < args.eval_n:
        logger.warning('Requested %d cases but only %d meet this policy.', args.eval_n, len(eval_subset))


if __name__ == "__main__":
    main()
