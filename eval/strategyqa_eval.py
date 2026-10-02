"""StrategyQA evaluation: yes/no boolean accuracy."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from data.datasets.common import read_jsonl
from data.generation.parsing import answers_match, extract_answer
from eval.metrics import EvalResult, accuracy, save_result
from models.student import StudentBundle, generate

LOG = logging.getLogger(__name__)

PROMPT_TEMPLATE = "### Question:\n{question}\n\n### Solution:\n"


@torch.no_grad()
def evaluate_strategyqa(
    bundle: StudentBundle,
    test_path: Path,
    run_name: str,
    max_new_tokens: int = 384,
    max_items: int | None = None,
    save_dir: Path | None = None,
) -> EvalResult:
    items = list(read_jsonl(Path(test_path)))
    if max_items is not None:
        items = items[:max_items]

    LOG.info("StrategyQA eval: %d items (run=%s)", len(items), run_name)

    records: list[dict] = []
    n_correct = 0

    for item in tqdm(items, desc="strategyqa-eval", unit="item"):
        prompt = PROMPT_TEMPLATE.format(question=item["input"].strip())
        try:
            output = generate(bundle, prompt, max_new_tokens=max_new_tokens,
                              temperature=0.0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            output = ""

        pred = extract_answer(output, answer_type="boolean")
        correct = answers_match(pred, item["answer"], "boolean")
        n_correct += int(correct)

        records.append({
            "id": item["id"],
            "correct": correct,
            "predicted": pred,
            "ground_truth": item["answer"],
            "output": output[:2000],
        })

    result = EvalResult(
        domain="strategyqa",
        split="test",
        run_name=run_name,
        n_total=len(records),
        n_correct=n_correct,
        accuracy=accuracy(n_correct, len(records)),
        per_provider={},
        per_answer_type={"boolean": {"n": len(records),
                                     "accuracy": accuracy(n_correct, len(records))}},
        extras={"max_new_tokens": max_new_tokens,
                "student_model": bundle.cfg.model_name},
    )

    if save_dir is not None:
        save_result(result, Path(save_dir) / run_name)
        records_path = Path(save_dir) / run_name / "strategyqa_test.records.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOG.info("StrategyQA eval done: %d/%d (accuracy=%.2f%%)",
             n_correct, len(records), 100 * result.accuracy)
    return result