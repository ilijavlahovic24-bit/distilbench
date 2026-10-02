"""Filter teacher rationales by correctness against ground truth.

An item passes if:
  - rationale is non-empty
  - extract_answer(rationale) matches the ground truth answer

Outputs two files:
  - <source>_<split>.filtered.jsonl  (passing items, ready for training)
  - <source>_<split>.rejected.jsonl  (failing items, kept for analysis)
Plus a summary stats dict.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from data.datasets.common import write_jsonl
from data.generation.parsing import answers_match, extract_answer

LOG = logging.getLogger(__name__)


def filter_file(
    rationales_path: Path,
    filtered_path: Path,
    rejected_path: Path,
) -> dict:
    from data.datasets.common import read_jsonl

    rationales_path = Path(rationales_path)
    filtered_path = Path(filtered_path)
    rejected_path = Path(rejected_path)

    passed: list[dict] = []
    rejected: list[dict] = []
    reasons: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    provider_pass: Counter[str] = Counter()

    for rec in read_jsonl(rationales_path):
        if rec.get("error"):
            reasons["teacher_error"] += 1
            rejected.append({**rec, "reject_reason": "teacher_error"})
            continue

        rationale = rec.get("rationale") or ""
        if not rationale.strip():
            reasons["empty_rationale"] += 1
            rejected.append({**rec, "reject_reason": "empty_rationale"})
            continue

        provider = rec.get("provider") or "unknown"
        provider_counts[provider] += 1

        pred = extract_answer(rationale, rec["answer_type"])
        if pred is None:
            reasons["no_answer_marker"] += 1
            rejected.append({**rec, "reject_reason": "no_answer_marker"})
            continue

        if not answers_match(pred, rec["answer"], rec["answer_type"]):
            reasons["answer_mismatch"] += 1
            rejected.append({
                **rec,
                "reject_reason": "answer_mismatch",
                "predicted_answer": pred,
            })
            continue

        provider_pass[provider] += 1
        passed.append({
            "id": rec["id"],
            "source": rec["source"],
            "split": rec["split"],
            "input": rec["input"],
            "rationale": rationale,
            "answer": rec["answer"],
            "answer_type": rec["answer_type"],
            "provider": rec.get("provider"),
            "model": rec.get("model"),
        })

    write_jsonl(filtered_path, passed)
    write_jsonl(rejected_path, rejected)

    total = len(passed) + len(rejected)
    pass_rate = len(passed) / total if total else 0.0
    per_provider_pass = {
        p: (provider_pass[p] / provider_counts[p] if provider_counts[p] else 0.0)
        for p in provider_counts
    }

    stats = {
        "input": str(rationales_path),
        "total": total,
        "passed": len(passed),
        "rejected": len(rejected),
        "pass_rate": pass_rate,
        "reject_reasons": dict(reasons),
        "provider_counts": dict(provider_counts),
        "provider_pass_rate": per_provider_pass,
    }

    LOG.info(
        "filter: %s -> passed=%d rejected=%d (pass_rate=%.1f%%)",
        rationales_path.name, len(passed), len(rejected), 100 * pass_rate,
    )
    if reasons:
        LOG.info("reject reasons: %s", dict(reasons))
    if per_provider_pass:
        LOG.info(
            "per-provider pass rate: %s",
            {p: f"{r*100:.1f}%" for p, r in per_provider_pass.items()},
        )

    return stats