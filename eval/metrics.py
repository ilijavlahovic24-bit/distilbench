"""Metrics and aggregation helpers for DistillBench evaluation.

Provides:
  - accuracy: numeric match (with tolerance)
  - boolean_accuracy: yes/no match
  - exact_match: string match (for sentinel tasks)
  - pass_at_1: for HumanEval-derivat (fraction of tests that pass)
  - aggregate_runs: collect results across experiments for the report
  - data_efficiency_curve: turn per-run accuracies into a plottable series
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger(__name__)


@dataclass
class EvalResult:
    domain: str
    split: str
    run_name: str
    n_total: int
    n_correct: int
    accuracy: float
    per_provider: dict[str, dict[str, float]] = field(default_factory=dict)
    per_answer_type: dict[str, dict[str, float]] = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "split": self.split,
            "run_name": self.run_name,
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "per_provider": self.per_provider,
            "per_answer_type": self.per_answer_type,
            **self.extras,
        }


def accuracy(n_correct: int, n_total: int) -> float:
    return n_correct / n_total if n_total else 0.0


def compute_per_group(
    records: Iterable[dict],
    key: str,
) -> dict[str, dict[str, float]]:
    """Group records by a field, compute accuracy per group."""
    groups: dict[str, list[bool]] = {}
    for r in records:
        g = r.get(key) or "unknown"
        groups.setdefault(str(g), []).append(bool(r["correct"]))
    return {
        g: {"n": len(v), "accuracy": sum(v) / len(v) if v else 0.0}
        for g, v in groups.items()
    }


def save_result(result: EvalResult, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.domain}_{result.split}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    LOG.info("Saved eval result -> %s", path)
    return path


def load_all_results(root: Path) -> list[dict]:
    root = Path(root)
    results: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "run_meta.json":
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "accuracy" in data:
                results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def data_efficiency_curve(
    results: list[dict],
    domain: str,
    split: str,
    metric: str = "accuracy",
) -> dict[str, list[tuple[int, float]]]:
    """Group by run family, return [(n_train, metric_value), ...] sorted.

    Run names are expected to embed the training size as `n<digits>`,
    e.g. 'gsm8k_cot_n500_alpha0.5'. If not found, we look in `extras`
    for `num_train_items`.
    """
    import re
    families: dict[str, list[tuple[int, float]]] = {}
    for r in results:
        if r.get("domain") != domain or r.get("split") != split:
            continue
        name = r.get("run_name", "")
        n = None
        m = re.search(r"_n(\d+)_", name)
        if m:
            n = int(m.group(1))
        elif "num_train_items" in r:
            n = int(r["num_train_items"])
        if n is None:
            continue
        # Family = run name with the n-tag stripped
        family = re.sub(r"_n\d+_", "_", name)
        families.setdefault(family, []).append((n, float(r.get(metric, 0.0))))

    return {fam: sorted(points) for fam, points in families.items()}