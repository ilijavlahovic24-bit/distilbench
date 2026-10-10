"""Exp4: cross-domain transfer (QA <-> code debugging).

Train a student on one domain, evaluate on both. The diagonal cells are
in-domain references; the off-diagonal cells are the cross-domain transfer.

  train \\ eval   gsm8k         humaneval_derivat
  gsm8k          in-domain     QA-trained, tested on code
  humaneval      code-trained, tested on QA   in-domain

We use the humaneval_derivat 'test' split as *both* train and eval for the
code path because that's the only split we have (it's a synthetic set).

Output: results/exp4_cross_domain.json as a 2x2 accuracy matrix.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from experiments.common import run_eval, run_train, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("exp4")

RESULT_PATH = Path("results/exp4_cross_domain.json")
EVAL_DIR = Path("results/exp4_eval")

TRAIN_SPECS = {
    "gsm8k": {
        "file": Path("data/datasets/processed/gsm8k_train.filtered.jsonl"),
    },
    "humaneval": {
        "file": Path("data/datasets/processed/humaneval_derivat_test.filtered.jsonl"),
    },
}

TEST_SPECS = {
    "gsm8k": {
        "file": Path("data/datasets/processed/gsm8k_test.jsonl"),
        "domain": "gsm8k",
    },
    "humaneval": {
        "file": Path("data/datasets/processed/humaneval_derivat_test.jsonl"),
        "domain": "humaneval_derivat",
    },
}


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--skip-humaneval-train", action="store_true",
                   help="Only evaluate the gsm8k-trained student on both domains.")
    args = p.parse_args()

    matrix: dict[str, dict[str, float]] = {}

    train_domains = list(TRAIN_SPECS)
    if args.skip_humaneval_train:
        train_domains = ["gsm8k"]

    for train_key in train_domains:
        spec = TRAIN_SPECS[train_key]
        if not spec["file"].exists():
            LOG.error("Missing train file for %s: %s", train_key, spec["file"])
            continue

        run_name = f"exp4_train_{train_key}"
        LOG.info("=== Train on %s ===", train_key)
        adapter = run_train(
            train_file=spec["file"],
            run_name=run_name,
            alpha=args.alpha,
            epochs=args.epochs,
        )

        matrix[train_key] = {}
        for test_key, test_spec in TEST_SPECS.items():
            if not test_spec["file"].exists():
                LOG.warning("Skip eval %s->%s: missing %s",
                            train_key, test_key, test_spec["file"])
                continue
            eval_run_name = f"exp4_{train_key}_to_{test_key}"
            result = run_eval(
                adapter=adapter,
                domain=test_spec["domain"],
                test_file=test_spec["file"],
                run_name=eval_run_name,
                save_dir=EVAL_DIR,
            )
            matrix[train_key][test_key] = result["accuracy"]
            LOG.info("  %s -> %s: acc=%.4f", train_key, test_key, result["accuracy"])

    save_json(RESULT_PATH, matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())