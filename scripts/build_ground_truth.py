"""Build a ground-truth training file from raw GSM8K.

The "no teacher" baseline for exp1: the student trains on the dataset's own
rationale (stored in metadata.ground_truth_rationale when the raw file was
built). Same schema as *.filtered.jsonl so the trainer needs no changes.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data.datasets.common import read_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("build_ground_truth")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", type=Path,
        default=Path("data/datasets/processed/gsm8k_train.jsonl"),
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("data/datasets/processed/gsm8k_train.ground_truth.jsonl"),
    )
    args = p.parse_args()

    records = []
    for rec in read_jsonl(args.input):
        gt = (rec.get("metadata") or {}).get("ground_truth_rationale") or ""
        if not gt.strip():
            continue
        records.append({
            "id": rec["id"],
            "source": rec["source"],
            "split": rec["split"],
            "input": rec["input"],
            "rationale": gt,
            "answer": rec["answer"],
            "answer_type": rec["answer_type"],
            "provider": "ground_truth",
            "model": "gsm8k_canonical",
        })

    write_jsonl(args.output, records)
    LOG.info("Wrote %d ground-truth records -> %s", len(records), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())