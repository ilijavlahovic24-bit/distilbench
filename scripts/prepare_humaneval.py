"""Build the HumanEval-derivat code-debugging JSONL."""
from __future__ import annotations

import logging

from dotenv import load_dotenv

from data.datasets.humaneval_derivat import build_humaneval_derivat_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("prepare_humaneval")

OUT_DIR = "data/datasets/processed"


def main() -> int:
    load_dotenv()
    path = build_humaneval_derivat_jsonl(OUT_DIR, n=None)
    LOG.info("Wrote HumanEval-derivat -> %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())