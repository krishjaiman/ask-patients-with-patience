#!/usr/bin/env python
"""
Runs the full evaluation (Section 3.5-3.7): one-shot, naive multi-turn,
and Dr.APP over a set of cases, writes per-case CSV/JSON results and an
entropy-trajectory plot, and prints a summary table.

Usage (offline smoke test):
    python scripts/evaluate.py --mock \
        --cases data/remedi/synthetic_sample.jsonl \
        --methods one_shot naive_multi_turn app \
        --out-dir outputs/smoke_eval

Usage (real run over the paper's 100-case subset):
    python scripts/evaluate.py \
        --cases data/processed/remedi_eval_100.jsonl \
        --methods one_shot naive_multi_turn app \
        --out-dir outputs/full_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import hashlib
from dataclasses import asdict
from pathlib import Path

from _common import (
    add_common_args, build_knowledge_base, build_llm_from_config, build_retriever, load_config,
)
from askpatients.data.remedi_loader import load_remedi_raw
from askpatients.evaluation.run_eval import EvaluationRunner

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--cases", required=True, help="ReMeDi-schema .jsonl file to evaluate")
    parser.add_argument("--methods", nargs="+", default=["one_shot", "naive_multi_turn", "app"],
                         choices=["one_shot", "naive_multi_turn", "app"])
    parser.add_argument("--out-dir", default=None, help="Overrides eval.output_dir")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N cases")
    args = parser.parse_args()

    cfg = load_config(args)
    out_dir = Path(args.out_dir or cfg.eval.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_remedi_raw(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    logger.info("Evaluating %d cases with methods=%s", len(cases), args.methods)

    llm = build_llm_from_config(cfg)
    judge_llm = build_llm_from_config(cfg, 'judge_llm')
    kb = build_knowledge_base(cfg, args.kb)
    manifest = {'config': asdict(cfg),
        'dataset_sha256': hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
        'knowledge_base': [asdict(e) for e in kb.entries],
        'methods': args.methods, 'case_ids': [c.case_id for c in cases],
        'exact_label_kb_coverage': sum(c.ground_truth_diagnosis in kb.all_disease_names() for c in cases),
        'evaluation': 'LLM judge approximation; mock scores are not research results'}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    retriever = build_retriever(cfg, kb)

    runner = EvaluationRunner(cfg, llm, judge_llm, kb, retriever,
        profile_llm=build_llm_from_config(cfg, 'profile_llm'),
        patient_llm=build_llm_from_config(cfg, 'patient_llm'))
    results = runner.run_all(cases, args.methods)

    runner.write_csv(results, out_dir / "results.csv")
    runner.write_json(results, out_dir / "results.json")

    summary = runner.summarize(results)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for method, stats in summary.items():
        print(f"\n{method}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    _maybe_plot_entropy(results, out_dir)
    logger.info("Wrote results to %s", out_dir)
    if any(r.status == 'failed' for r in results):
        raise SystemExit('Evaluation incomplete: failed cases are recorded in results.json')


def _maybe_plot_entropy(results, out_dir: Path) -> None:
    """Section 3.6 / Figure 4 style entropy-across-iterations plot for the
    'app' method (only method with a real per-turn entropy trajectory)."""
    app_results = [r for r in results if r.method == "app" and r.entropy_trajectory]
    if not app_results:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping entropy plot.")
        return

    max_len = max(len(r.entropy_trajectory) for r in app_results)
    avg = []
    for i in range(max_len):
        vals = [r.entropy_trajectory[i] for r in app_results if len(r.entropy_trajectory) > i]
        avg.append(sum(vals) / len(vals))

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(avg) + 1), avg, marker="o", label="Dr.APP (this reproduction)")
    plt.xlabel("Iteration")
    plt.ylabel("Average entropy")
    plt.title("Average Entropy Across Iterations")
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / "entropy_across_iterations.png"
    plt.savefig(plot_path)
    logger.info("Saved entropy plot to %s", plot_path)


if __name__ == "__main__":
    main()
