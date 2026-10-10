"""Shared helpers for experiment scripts.

Every experiment is a small orchestrator: it slices data, calls
scripts.train_student and scripts.eval_student as subprocesses, and
aggregates results into a JSON file under results/.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)


def slice_jsonl(src: Path, dst: Path, n: int) -> None:
    """Copy the first n lines of src to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            if i >= n:
                break
            fo.write(line)


def run_train(
    train_file: Path,
    run_name: str,
    alpha: float,
    epochs: int = 3,
    student_model: str | None = None,
    output_root: Path = Path("results/checkpoints"),
) -> Path:
    """Invoke scripts.train_student; return adapter dir on success."""
    cmd = [
        sys.executable, "-m", "scripts.train_student",
        "--train-file", str(train_file),
        "--run-name", run_name,
        "--alpha", str(alpha),
        "--epochs", str(epochs),
        "--output-root", str(output_root),
    ]
    if student_model:
        cmd += ["--student-model", student_model]

    LOG.info("TRAIN: %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"train_student failed (rc={rc}) for run={run_name}")
    return output_root / run_name / "adapter"


def run_eval(
    adapter: Path,
    domain: str,
    test_file: Path,
    run_name: str,
    save_dir: Path,
    max_new_tokens: int | None = None,
) -> dict:
    """Invoke scripts.eval_student; return parsed result dict."""
    cmd = [
        sys.executable, "-m", "scripts.eval_student",
        "--adapter", str(adapter),
        "--domain", domain,
        "--test-file", str(test_file),
        "--run-name", run_name,
        "--save-dir", str(save_dir),
    ]
    if max_new_tokens:
        cmd += ["--max-new-tokens", str(max_new_tokens)]

    LOG.info("EVAL:  %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"eval_student failed (rc={rc}) for run={run_name}")

    result_path = Path(save_dir) / run_name / f"{domain}_test.json"
    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    LOG.info("Saved -> %s", path)