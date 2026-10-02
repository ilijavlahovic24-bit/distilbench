"""CLI: filter teacher rationales by ground-truth correctness.

Examples:
    python -m scripts.filter_rationales --domain gsm8k --split train
    python -m scripts.filter_rationales --all
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from data.generation.filter_rationales import filter_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("filter_rationales")

IN_DIR = Path("data/generation/output")
OUT_DIR = Path("data/datasets/processed")
STATS_PATH = Path("data/generation/filter_stats.jsonl")

_DOMAINS = ("gsm8k", "strategyqa", "humaneval_derivat")
_SPLITS = ("train", "test")


def _run_one(domain: str, split: str) -> dict | None:
    in_path = IN_DIR / f"{domain}_{split}.rationales.jsonl"
    if not in_path.exists():
        LOG.warning("Skip %s/%s: %s not found", domain, split, in_path)
        return None
    filtered = OUT_DIR / f"{domain}_{split}.filtered.jsonl"
    rejected = OUT_DIR / f"{domain}_{split}.rejected.jsonl"
    stats = filter_file(in_path, filtered, rejected)
    stats["domain"] = domain
    stats["split"] = split
    return stats


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=_DOMAINS, default=None)
    p.add_argument("--split", choices=_SPLITS, default="train")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if not args.all and args.domain is None:
        p.error("Specify --domain or --all")

    all_stats: list[dict] = []
    if args.all:
        for domain in _DOMAINS:
            for split in _SPLITS:
                s = _run_one(domain, split)
                if s:
                    all_stats.append(s)
    else:
        s = _run_one(args.domain, args.split)
        if s:
            all_stats.append(s)

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a", encoding="utf-8") as f:
        for s in all_stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    LOG.info("Done. Stats written to %s", STATS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())