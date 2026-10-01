"""GSM8K loader -> unified raw JSONL.

GSM8K answers come formatted as '<rationale>\n#### <number>'. We split the
final number out (for filtering) and keep the rationale as ground-truth
metadata (useful for the 'direct fine-tune on ground truth' baseline in Faza 3).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from datasets import load_dataset

from data.datasets.common import write_jsonl

LOG = logging.getLogger(__name__)

_FINAL_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def _split_gsm8k_answer(text: str) -> tuple[str, str]:
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        rationale = text[: m.start()].strip()
        final = m.group(1).strip().replace(",", "")
        return rationale, final
    return text.strip(), ""


def load_gsm8k(split: str, n: int | None = None, seed: int = 42) -> list[dict]:
    LOG.info("Loading GSM8K split='%s' (n=%s)", split, n)
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))

    items: list[dict] = []
    for idx, row in enumerate(ds):
        gt_rationale, final = _split_gsm8k_answer(row["answer"])
        items.append({
            "id": f"gsm8k_{split}_{idx:05d}",
            "domain": "qa",
            "source": "gsm8k",
            "split": split,
            "input": row["question"].strip(),
            "answer": final,
            "answer_type": "numeric",
            "metadata": {"ground_truth_rationale": gt_rationale},
        })
    LOG.info("GSM8K '%s': %d items", split, len(items))
    return items


def build_gsm8k_jsonl(
    out_dir: Path,
    n_train: int = 2500,
    include_test: bool = True,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    train_items = load_gsm8k("train", n=n_train)
    train_path = out_dir / "gsm8k_train.jsonl"
    write_jsonl(train_path, train_items)
    paths["train"] = train_path

    if include_test:
        test_items = load_gsm8k("test", n=None)
        test_path = out_dir / "gsm8k_test.jsonl"
        write_jsonl(test_path, test_items)
        paths["test"] = test_path

    return paths