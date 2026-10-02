"""GSM8K evaluation: numeric accuracy against ground-truth answer.

Consumes the same JSONL format as training (id, input, answer, answer_type)
and the same parsing module as the teacher filtering, so the pipeline is
consistent end to end.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from data.datasets.common import read_jsonl
from data.generation.parsing import answers_match, extract_answer
from eval.metrics import EvalResult, accuracy, compute_per_group, save_result
from models.student import StudentBundle, generate

LOG = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "### Question:\n{question}\n\n### Solution:\n"
)
ANSWER_SEED = "\n\n### Answer:\n"


def build_eval_prompt(question: str, include_answer_seed: bool = True) -> str:
    """Same prefix the student saw at training time, but with no rationale.

    For CoT-trained students we expect them to produce a rationale before the
    answer. For answer-only trained students, they go straight to the answer.
    """
    prompt = PROMPT_TEMPLATE.format(question=question.strip())
    if include_answer_seed:
        return prompt  # let the model choose; some generate "### Answer:" themselves
    return prompt


@torch.no_grad()
def evaluate_gsm8k(
    bundle: StudentBundle,
    test_path: Path,
    run_name: str,
    max_new_tokens: int = 384,
    max_items: int | None = None,
    save_dir: Path | None = None,
) -> EvalResult:
    test_path = Path(test_path)
    items = list(read_jsonl(test_path))
    if max_items is not None:
        items = items[:max_items]

    LOG.info("GSM8K eval: %d items from %s (run=%s)", len(items), test_path, run_name)

    records: list[dict] = []
    n_correct = 0

    for item in tqdm(items, desc="gsm8k-eval", unit="item"):
        prompt = build_eval_prompt(item["input"])
        try:
            output = generate(bundle, prompt, max_new_tokens=max_new_tokens,
                              temperature=0.0)
        except torch.cuda.OutOfMemoryError:
            LOG.error("OOM during generation for id=%s; skipping", item["id"])
            torch.cuda.empty_cache()
            output = ""

        pred = extract_answer(output, answer_type="numeric")
        correct = answers_match(pred, item["answer"], "numeric")
        n_correct += int(correct)

        records.append({
            "id": item["id"],
            "correct": correct,
            "predicted": pred,
            "ground_truth": item["answer"],
            "output": output[:2000],
            "source": item.get("source", "gsm8k"),
        })

    result = EvalResult(
        domain="gsm8k",
        split="test",
        run_name=run_name,
        n_total=len(records),
        n_correct=n_correct,
        accuracy=accuracy(n_correct, len(records)),
        per_provider={},  # not applicable (student, not teacher)
        per_answer_type={"numeric": {"n": len(records),
                                     "accuracy": accuracy(n_correct, len(records))}},
        extras={
            "max_new_tokens": max_new_tokens,
            "student_model": bundle.cfg.model_name,
        },
    )

    if save_dir is not None:
        save_result(result, Path(save_dir) / run_name)
        # also dump per-item records for error analysis in the report
        import json
        records_path = Path(save_dir) / run_name / "gsm8k_test.records.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        LOG.info("Saved per-item records -> %s", records_path)

    LOG.info(
        "GSM8K eval done: %d/%d correct (accuracy=%.2f%%)",
        n_correct, len(records), 100 * result.accuracy,
    )
    return result