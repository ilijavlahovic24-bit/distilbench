"""Build raw JSONL datasets (input + ground truth answer, no teacher yet).

Usage (from repo root):
    python -m scripts.prepare_data --domain gsm8k
    python -m scripts.prepare_data --domain strategyqa
    python -m scripts.prepare_data --all
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from data.datasets.gsm8k import build_gsm8k_jsonl
from data.datasets.strategyqa import build_strategyqa_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("prepare_data")

OUT_DIR = Path("data/datasets/processed")


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["gsm8k", "strategyqa"], default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-train-gsm8k", type=int, default=2500)
    p.add_argument("--n-train-strategyqa", type=int, default=2000)
    args = p.parse_args()

    if not args.all and args.domain is None:
        p.error("Specify --domain or --all")

    if args.all or args.domain == "gsm8k":
        paths = build_gsm8k_jsonl(OUT_DIR, n_train=args.n_train_gsm8k)
        for split, path in paths.items():
            LOG.info("Wrote GSM8K %s -> %s", split, path)

    if args.all or args.domain == "strategyqa":
        paths = build_strategyqa_jsonl(OUT_DIR, n_train=args.n_train_strategyqa)
        for split, path in paths.items():
            LOG.info("Wrote StrategyQA %s -> %s", split, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())