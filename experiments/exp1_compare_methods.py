"""Exp1: CoT-distilled vs answer-only vs ground-truth fine-tune.

Three training signals, same student, same n training items, same test set:

  cot           : teacher CoT rationale + answer, alpha=0.5
  answer_only   : teacher's data but loss on answer only, alpha=1.0
  ground_truth  : dataset's own rationale + answer, alpha=0.5 (no teacher)

Note on answer_only: the rationale is still present in the input sequence
(the student sees it as context) but does not contribute to the loss. This
isolates the effect of the *training signal* from the effect of *seeing*
the rationale.

Output: results/exp1_compare_methods.json
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from experiments.common import run_eval, run_train, save_json, slice_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("exp1")

RESULT_PATH = Path("results/exp1_compare_methods.json")
EVAL_DIR = Path("results/exp1_eval")
SLICE_DIR = Path("results/exp1_slices")

METHODS = {
    "cot":          {"train_file": "gsm8k_train.filtered.jsonl",     "alpha": 0.5},
    "answer_only":  {"train_file": "gsm8k_train.filtered.jsonl",     "alpha": 1.0},
    "ground_truth": {"train_file": "gsm8k_train.ground_truth.jsonl", "alpha": 0.5},
}


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000,
                   help="Number of training items per method.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--test-file", type=Path,
                   default=Path("data/datasets/processed/gsm8k_test.jsonl"))
    p.add_argument("--data-dir", type=Path,
                   default=Path("data/datasets/processed"))
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--methods", nargs="+", choices=list(METHODS),
                   default=list(METHODS))
    args = p.parse_args()

    table: dict[str, dict] = {}

    for method in args.methods:
        spec = METHODS[method]
        train_src = args.data_dir / spec["train_file"]
        if not train_src.exists():
            LOG.error("Missing training file for %s: %s", method, train_src)
            if method == "ground_truth":
                LOG.error("Run: python -m scripts.build_ground_truth")
            return 1

        sliced = SLICE_DIR / f"{method}_n{args.n}.jsonl"
        slice_jsonl(train_src, sliced, args.n)

        run_name = f"exp1_{method}_n{args.n}"
        LOG.info("=== %s (n=%d, alpha=%.2f) ===", method, args.n, spec["alpha"])

        adapter = run_train(
            train_file=sliced,
            run_name=run_name,
            alpha=spec["alpha"],
            epochs=args.epochs,
        )
        result = run_eval(
            adapter=adapter,
            domain="gsm8k",
            test_file=args.test_file,
            run_name=run_name,
            save_dir=EVAL_DIR,
            max_new_tokens=args.max_new_tokens,
        )
        table[method] = {
            "n_train": args.n,
            "alpha": spec["alpha"],
            "accuracy": result["accuracy"],
            "n_correct": result["n_correct"],
            "n_total": result["n_total"],
            "adapter": str(adapter),
        }

    save_json(RESULT_PATH, table)

    LOG.info("=== Exp1 summary (n=%d) ===", args.n)
    for m, row in table.items():
        LOG.info("  %-14s acc=%.4f  (%d/%d)",
                 m, row["accuracy"], row["n_correct"], row["n_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())