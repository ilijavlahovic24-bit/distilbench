"""Exp3: student size ablation (0.5B vs 1.5B).

Same method (CoT distillation), same data, two students. On GTX 1650 the
1.5B variant is tight — if training fails with OOM, the script logs the
failure and continues with 0.5B only. Use --skip-large to skip it outright.

Output: results/exp3_student_size.json
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
LOG = logging.getLogger("exp3")

RESULT_PATH = Path("results/exp3_student_size.json")
EVAL_DIR = Path("results/exp3_eval")

STUDENTS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", type=Path,
                   default=Path("data/datasets/processed/gsm8k_train.filtered.jsonl"))
    p.add_argument("--test-file", type=Path,
                   default=Path("data/datasets/processed/gsm8k_test.jsonl"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--skip-large", action="store_true",
                   help="Skip the 1.5B variant (recommended if VRAM is tight).")
    p.add_argument("--students", nargs="+", choices=list(STUDENTS),
                   default=list(STUDENTS))
    args = p.parse_args()

    if args.skip_large and "1.5b" in args.students:
        args.students = [s for s in args.students if s != "1.5b"]

    table: dict[str, dict] = {}

    for key in args.students:
        model_name = STUDENTS[key]
        run_name = f"exp3_student_{key}"
        LOG.info("=== Student %s (%s) ===", key, model_name)

        try:
            adapter = run_train(
                train_file=args.train_file,
                run_name=run_name,
                alpha=args.alpha,
                epochs=args.epochs,
                student_model=model_name,
            )
        except RuntimeError as exc:
            LOG.error("Training failed for %s: %s", key, exc)
            table[key] = {"model": model_name, "error": str(exc)}
            continue

        result = run_eval(
            adapter=adapter,
            domain="gsm8k",
            test_file=args.test_file,
            run_name=run_name,
            save_dir=EVAL_DIR,
        )
        table[key] = {
            "model": model_name,
            "accuracy": result["accuracy"],
            "n_correct": result["n_correct"],
            "n_total": result["n_total"],
            "adapter": str(adapter),
        }

    save_json(RESULT_PATH, table)
    LOG.info("=== Exp3 summary ===")
    for k, row in table.items():
        if "accuracy" in row:
            LOG.info("  %s: acc=%.4f", k, row["accuracy"])
        else:
            LOG.info("  %s: FAILED (%s)", k, row.get("error", "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())