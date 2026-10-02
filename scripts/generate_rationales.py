"""CLI: run teacher chain over a raw JSONL to produce rationales.

Examples (from repo root):
    python -m scripts.generate_rationales --domain gsm8k --split train
    python -m scripts.generate_rationales --domain gsm8k --split test
    python -m scripts.generate_rationales --domain strategyqa --split train
    python -m scripts.generate_rationales --domain humaneval_derivat --split test
    python -m scripts.generate_rationales --all-train
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from data.generation.generate_rationales import generate_rationales
from models.teacher import build_default_chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("generate_rationales")

DATA_DIR = Path("data/datasets/processed")
OUT_DIR = Path("data/generation/output")
STATS_PATH = Path("data/generation/generation_stats.jsonl")

_DOMAIN_FILES = {
    "gsm8k": "gsm8k_{split}.jsonl",
    "strategyqa": "strategyqa_{split}.jsonl",
    "humaneval_derivat": "humaneval_derivat_{split}.jsonl",
}


def _run_one(domain: str, split: str, chain, max_items: int | None) -> dict:
    in_path = DATA_DIR / _DOMAIN_FILES[domain].format(split=split)
    out_path = OUT_DIR / f"{domain}_{split}.rationales.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input not found: {in_path}. Run scripts.prepare_data first."
        )
    stats = generate_rationales(
        input_path=in_path,
        output_path=out_path,
        chain=chain,
        max_items=max_items,
    )
    stats["domain"] = domain
    stats["split"] = split
    stats["output"] = str(out_path)
    return stats


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--domain",
        choices=list(_DOMAIN_FILES),
        default=None,
    )
    p.add_argument("--split", default="train")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument(
        "--all-train",
        action="store_true",
        help="Run train split for gsm8k + strategyqa + humaneval_derivat.",
    )
    args = p.parse_args()

    if not args.all_train and args.domain is None:
        p.error("Specify --domain or --all-train")

    chain = build_default_chain()
    LOG.info(
        "Teacher chain: %s",
        [(c.provider, c.model) for c in chain.clients],
    )

    all_stats: list[dict] = []
    if args.all_train:
        for domain in ("gsm8k", "strategyqa", "humaneval_derivat"):
            try:
                all_stats.append(_run_one(domain, "train", chain, args.max_items))
            except FileNotFoundError as exc:
                LOG.warning("Skipping %s: %s", domain, exc)
    else:
        all_stats.append(
            _run_one(args.domain, args.split, chain, args.max_items)
        )

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a", encoding="utf-8") as f:
        for s in all_stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    LOG.info("Done. Stats written to %s", STATS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())