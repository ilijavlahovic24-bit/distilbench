"""CLI: run data-efficiency sweep (multiple training sizes, one eval each).

For each (method, n) pair:
  1. Slice the filtered training file to n items
  2. Train a fresh student
  3. Evaluate on the same test set
  4. Save accuracy

Output: results/data_efficiency/<method>_n<N>.json
The metrics.data_efficiency_curve() helper aggregates these into plot data.

Examples:
  # CoT distillation, sizes 100/500/1000/2000
  python -m scripts.data_efficiency \
      --method cot --sizes 100 500 1000 2000 \
      --train-file data/datasets/processed/gsm8k_train.filtered.jsonl \
      --test-file data/datasets/processed/gsm8k_test.jsonl \
      --domain gsm8k

  # Answer-only baseline, same sizes
  python -m scripts.data_efficiency \
      --method answer_only --sizes 100 500 1000 2000 ...
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("data_efficiency")

RESULT_ROOT = Path("results/data_efficiency")


def _slice_jsonl(src: Path, dst: Path, n: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            if i >= n:
                break
            fo.write(line)


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["cot", "answer_only", "ground_truth"])
    p.add_argument("--sizes", nargs="+", type=int, required=True)
    p.add_argument("--train-file", required=True, type=Path)
    p.add_argument("--test-file", required=True, type=Path)
    p.add_argument("--domain", required=True,
                   choices=["gsm8k", "strategyqa", "humaneval_derivat"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--resume", action="store_true",
                   help="Skip (method, n) pairs that already have results.")
    args = p.parse_args()

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    for n in args.sizes:
        run_name = f"{args.domain}_{args.method}_n{n}"
        result_path = RESULT_ROOT / f"{run_name}.json"
        if args.resume and result_path.exists():
            LOG.info("Skip %s (already done)", run_name)
            continue

        sliced = RESULT_ROOT / "_slices" / f"{args.method}_n{n}.jsonl"
        _slice_jsonl(args.train_file, sliced, n)

        # alpha per method: cot=0.5, answer_only=1.0, ground_truth=0.5
        alpha = {"cot": 0.5, "answer_only": 1.0, "ground_truth": 0.5}[args.method]

        LOG.info("=== Training %s (n=%d, alpha=%.2f) ===", run_name, n, alpha)
        rc = subprocess.call([
            sys.executable, "-m", "scripts.train_student",
            "--train-file", str(sliced),
            "--run-name", run_name,
            "--alpha", str(alpha),
            "--epochs", str(args.epochs),
        ])
        if rc != 0:
            LOG.error("Training failed for %s (rc=%d); aborting sweep", run_name, rc)
            return rc

        LOG.info("=== Evaluating %s ===", run_name)
        adapter = Path("results/checkpoints") / run_name / "adapter"
        rc = subprocess.call([
            sys.executable, "-m", "scripts.eval_student",
            "--adapter", str(adapter),
            "--domain", args.domain,
            "--test-file", str(args.test_file),
            "--run-name", run_name,
            "--save-dir", str(RESULT_ROOT),
        ])
        if rc != 0:
            LOG.error("Eval failed for %s (rc=%d); aborting sweep", run_name, rc)
            return rc

        # Copy the result under the flat name so the curve helper finds it.
        src = RESULT_ROOT / run_name / f"{args.domain}_test.json"
        if src.exists():
            with src.open("r", encoding="utf-8") as f:
                data = json.load(f)
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            LOG.info("Saved %s", result_path)

    LOG.info("Data-efficiency sweep complete. Results in %s", RESULT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())