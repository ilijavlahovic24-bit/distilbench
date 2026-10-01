"""StrategyQA loader -> unified raw JSONL.

StrategyQA is a yes/no multi-hop QA dataset. The 'answer' field is boolean
in some HF mirrors and 'yes'/'no' string in others; we normalise to lowercase
'yes'/'no'. The dataset's own 'facts' field is kept as ground-truth metadata,
but we still generate teacher rationales (the point of distillation is to
teach the student the *reasoning chain*, not just the label).
"""
from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset

from data.datasets.common import write_jsonl

LOG = logging.getLogger(__name__)

# Try a few mirrors; the canonical one occasionally moves.
_CANDIDATES: list[tuple[str, str | None]] = [
    ("voidful/StrategyQA", "StrategyQA"),
    ("voidful/StrategyQA", None),
    ("ChilleD/StrategyQA", None),
]


def _normalize_bool(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    s = str(v).strip().lower()
    if s in {"true", "yes", "1"}:
        return "yes"
    if s in {"false", "no", "0"}:
        return "no"
    return s


def _load_hf_split(split: str):
    last_err: Exception | None = None
    for name, config in _CANDIDATES:
        try:
            if config:
                ds = load_dataset(name, config, split=split)
            else:
                ds = load_dataset(name, split=split)
            LOG.info("Loaded StrategyQA from '%s' (config=%s)", name, config)
            return ds
        except Exception as e:  # noqa: BLE001
            last_err = e
            LOG.debug("StrategyQA candidate '%s' failed: %s", name, e)
    raise RuntimeError(
        f"Could not load StrategyQA split='{split}' from any known mirror. "
        f"Last error: {last_err}"
    )


def load_strategyqa(split: str, n: int | None = None, seed: int = 42) -> list[dict]:
    LOG.info("Loading StrategyQA split='%s' (n=%s)", split, n)
    ds = _load_hf_split(split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))

    items: list[dict] = []
    for idx, row in enumerate(ds):
        question = row.get("question") or row.get("q") or ""
        answer_raw = row.get("answer", None)
        facts = row.get("facts", None)
        items.append({
            "id": f"strategyqa_{split}_{idx:05d}",
            "domain": "qa",
            "source": "strategyqa",
            "split": split,
            "input": str(question).strip(),
            "answer": _normalize_bool(answer_raw),
            "answer_type": "boolean",
            "metadata": {
                "ground_truth_facts": list(facts) if facts else [],
            },
        })
    LOG.info("StrategyQA '%s': %d items", split, len(items))
    return items


def build_strategyqa_jsonl(
    out_dir: Path,
    n_train: int = 2000,
    include_test: bool = True,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    train_items = load_strategyqa("train", n=n_train)
    train_path = out_dir / "strategyqa_train.jsonl"
    write_jsonl(train_path, train_items)
    paths["train"] = train_path

    if include_test:
        test_items = load_strategyqa("test", n=None)
        test_path = out_dir / "strategyqa_test.jsonl"
        write_jsonl(test_path, test_items)
        paths["test"] = test_path

    return paths