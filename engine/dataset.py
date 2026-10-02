"""Tokenized distillation dataset with rationale/answer segment masks.

Sequence format (identical at train and eval time):

    ### Question:
    {input}

    ### Solution:
    {rationale}

    ### Answer:
    {answer}<eos>

Segments:
  prompt    : "### Question:\n{input}\n\n### Solution:\n"   -> no loss
  rationale : "{rationale}"                                  -> rationale CE
  answer    : "\n\n### Answer:\n{answer}<eos>"               -> answer CE

Any item whose prompt + answer alone exceeds max_seq_length is dropped.
Items with over-long rationales have the rationale truncated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data.datasets.common import read_jsonl

LOG = logging.getLogger(__name__)

PROMPT_TEMPLATE = "### Question:\n{input}\n\n### Solution:\n"
ANSWER_PREFIX = "\n\n### Answer:\n"


def build_prompt(input_text: str) -> str:
    return PROMPT_TEMPLATE.format(input=input_text.strip())


class DistillDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 1024,
        max_items: int | None = None,
        require_rationale: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.eos_id = tokenizer.eos_token_id

        items: list[dict] = []
        for rec in read_jsonl(Path(path)):
            if require_rationale and not (rec.get("rationale") or "").strip():
                continue
            items.append(rec)

        if max_items is not None:
            items = items[:max_items]

        # Pre-filter items that cannot fit even without a rationale
        self.items = [it for it in items if self._fits_minimum(it)]
        dropped = len(items) - len(self.items)
        if dropped:
            LOG.warning(
                "Dropped %d items (prompt+answer exceeds max_seq_length=%d)",
                dropped, max_seq_length,
            )
        LOG.info("DistillDataset: %d items from %s", len(self.items), path)

    def _tok(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _fits_minimum(self, item: dict) -> bool:
        prompt_len = len(self._tok(build_prompt(item["input"])))
        answer_len = len(self._tok(ANSWER_PREFIX + item["answer"])) + 1  # +eos
        return prompt_len + answer_len <= self.max_seq_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        item = self.items[idx]

        prompt_ids = self._tok(build_prompt(item["input"]))
        answer_ids = self._tok(ANSWER_PREFIX + item["answer"]) + [self.eos_id]

        rationale_ids = self._tok(item.get("rationale") or "")

        # Truncate rationale if needed.
        rationale_budget = self.max_seq_length - len(prompt_ids) - len(answer_ids)
        if rationale_budget < 0:
            rationale_budget = 0
        rationale_ids = rationale_ids[:rationale_budget]

        input_ids = prompt_ids + rationale_ids + answer_ids
        labels = (
            [-100] * len(prompt_ids)
            + rationale_ids
            + answer_ids
        )
        rationale_mask = (
            [0] * len(prompt_ids)
            + [1] * len(rationale_ids)
            + [0] * len(answer_ids)
        )
        answer_mask = (
            [0] * len(prompt_ids)
            + [0] * len(rationale_ids)
            + [1] * len(answer_ids)
        )

        assert len(input_ids) == len(labels) == len(rationale_mask) == len(answer_mask)
        assert len(input_ids) <= self.max_seq_length

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "rationale_mask": rationale_mask,
            "answer_mask": answer_mask,
        }


@dataclass
class DistillCollator:
    """Pad all fields per-batch. Mask fields become float32."""

    tokenizer: PreTrainedTokenizerBase
    label_pad_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        dtypes = {
            "input_ids": torch.long,
            "attention_mask": torch.long,
            "labels": torch.long,
            "rationale_mask": torch.float32,
            "answer_mask": torch.float32,
        }

        out: dict[str, list[list[Any]]] = {k: [] for k in dtypes}
        for f in features:
            pad = max_len - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [pad_id] * pad)
            out["attention_mask"].append(f["attention_mask"] + [0] * pad)
            out["labels"].append(f["labels"] + [self.label_pad_id] * pad)
            out["rationale_mask"].append(f["rationale_mask"] + [0] * pad)
            out["answer_mask"].append(f["answer_mask"] + [0] * pad)

        return {k: torch.tensor(v, dtype=dtypes[k]) for k, v in out.items()}