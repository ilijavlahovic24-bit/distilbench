"""HumanEval-derivat evaluation: extract fixed code, run tests, count pass@1.

The student is expected to produce a debug rationale that ends with the
corrected function. We extract the first Python code block, prepend the
HumanEval test harness, and run it in a subprocess with a timeout.

Safety: we run in a subprocess with a hard timeout and no network. The
inputs are our own mutated functions so this is not a hostile-input
scenario, but the isolation also protects against infinite loops.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm

from data.datasets.common import read_jsonl
from eval.metrics import EvalResult, accuracy, save_result
from models.student import StudentBundle, generate

LOG = logging.getLogger(__name__)

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

PROMPT_TEMPLATE = (
    "### Task:\n{task}\n\n"
    "Buggy implementation:\n"
    "```python\n{buggy}\n```\n\n"
    "### Solution:\n"
)

# HumanEval test cases are stored in the dataset as a `test` field in the
# original HF dataset; we stored only the entry point and description in our
# JSONL. To run tests, we load them from the original HF dataset at eval time
# and cache by task_id.

_TEST_CACHE: dict[str, str] = {}


def _load_humaneval_tests() -> dict[str, str]:
    global _TEST_CACHE
    if _TEST_CACHE:
        return _TEST_CACHE
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    for row in ds:
        _TEST_CACHE[row["task_id"]] = row["test"]
    return _TEST_CACHE


def _extract_code_block(text: str) -> str | None:
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _run_tests(
    fixed_code: str,
    test_code: str,
    entry_point: str,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Run the fixed code against HumanEval tests in a subprocess."""
    harness = (
        fixed_code
        + "\n\n"
        + test_code
        + f"\n\ncheck({entry_point})\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(harness)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout)[-500:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # noqa: BLE001
        return False, f"harness error: {exc}"
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


@torch.no_grad()
def evaluate_humaneval_derivat(
    bundle: StudentBundle,
    test_path: Path,
    run_name: str,
    max_new_tokens: int = 512,
    max_items: int | None = None,
    save_dir: Path | None = None,
) -> EvalResult:
    items = list(read_jsonl(Path(test_path)))
    if max_items is not None:
        items = items[:max_items]

    tests = _load_humaneval_tests()
    LOG.info("HumanEval-derivat eval: %d items (run=%s)", len(items), run_name)

    records: list[dict] = []
    n_correct = 0

    for item in tqdm(items, desc="humaneval-eval", unit="item"):
        meta = item["metadata"]
        task_id = meta["task_id"]
        entry_point = meta["entry_point"]
        test_code = tests.get(task_id)
        if test_code is None:
            LOG.warning("No test code for %s; skipping", task_id)
            continue

        prompt = PROMPT_TEMPLATE.format(
            task=meta["task_description"],
            buggy=meta["buggy_code"],
        )
        try:
            output = generate(bundle, prompt, max_new_tokens=max_new_tokens,
                              temperature=0.0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            output = ""

        fixed = _extract_code_block(output)
        if fixed is None:
            passed, err = False, "no code block"
        else:
            passed, err = _run_tests(fixed, test_code, entry_point)

        n_correct += int(passed)
        records.append({
            "id": item["id"],
            "task_id": task_id,
            "bug_type": meta.get("bug_type"),
            "correct": passed,
            "error": err,
            "output": output[:3000],
        })

    result = EvalResult(
        domain="humaneval_derivat",
        split="test",
        run_name=run_name,
        n_total=len(records),
        n_correct=n_correct,
        accuracy=accuracy(n_correct, len(records)),
        per_provider={},
        per_answer_type={"sentinel": {"n": len(records),
                                      "accuracy": accuracy(n_correct, len(records))}},
        extras={"max_new_tokens": max_new_tokens,
                "student_model": bundle.cfg.model_name,
                "metric": "pass@1"},
    )

    if save_dir is not None:
        save_result(result, Path(save_dir) / run_name)
        records_path = Path(save_dir) / run_name / "humaneval_test.records.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOG.info("HumanEval-derivat eval done: %d/%d (pass@1=%.2f%%)",
             n_correct, len(records), 100 * result.accuracy)
    return result