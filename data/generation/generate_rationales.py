"""Batch-generate teacher rationales through the fallback chain.

Design
------
- Input: raw JSONL (e.g. gsm8k_train.jsonl) with fields id, input, source, ...
- Output: append-only JSONL with one record per input:
      {id, source, split, provider, model, rationale, generated_at, error?}
- Resume-safe: on startup, we read the existing output and skip any id that
  already has a successful record. Failed records (error field) are retried.
- The SQLite cache in teacher.py provides a second layer of dedup at the
  prompt level, so re-running the same split is cheap even if we delete the
  output file.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from data.datasets.common import append_jsonl, read_jsonl
from data.generation.prompts import build_prompt
from models.teacher import TeacherChain, TeacherError

LOG = logging.getLogger(__name__)


def _load_done_ids(out_path: Path) -> set[str]:
    """Ids that have a successful (error-free) record in the output file."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    for rec in read_jsonl(out_path):
        if rec.get("id") and not rec.get("error"):
            done.add(rec["id"])
    return done


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_rationales(
    input_path: Path,
    output_path: Path,
    chain: TeacherChain,
    max_items: int | None = None,
    log_every: int = 25,
) -> dict:
    """Generate rationales for every item in input_path, skipping done ids.

    Returns a small stats dict for the caller.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(read_jsonl(input_path))
    if max_items is not None:
        items = items[:max_items]

    done_ids = _load_done_ids(output_path)
    todo = [it for it in items if it["id"] not in done_ids]

    LOG.info(
        "generate_rationales: input=%s total=%d done=%d todo=%d",
        input_path.name, len(items), len(done_ids), len(todo),
    )

    stats = {"total": len(items), "skipped": len(done_ids), "ok": 0, "failed": 0}
    t0 = time.time()

    for i, item in enumerate(tqdm(todo, desc=output_path.name, unit="item"), 1):
        prompt, system = build_prompt(item)
        record = {
            "id": item["id"],
            "source": item["source"],
            "split": item["split"],
            "input": item["input"],
            "answer": item["answer"],
            "answer_type": item["answer_type"],
            "generated_at": _utc_now(),
        }
        try:
            resp = chain.generate(prompt, system=system)
            record["provider"] = resp.provider
            record["model"] = resp.model
            record["rationale"] = resp.text
            record["cached"] = resp.cached
            stats["ok"] += 1
        except TeacherError as exc:
            record["provider"] = None
            record["model"] = None
            record["rationale"] = None
            record["error"] = str(exc)[:500]
            stats["failed"] += 1
            LOG.warning("Failed id=%s: %s", item["id"], exc)

        append_jsonl(output_path, record)

        if i % log_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            LOG.info(
                "progress: %d/%d (%.2f items/s) ok=%d failed=%d",
                i, len(todo), rate, stats["ok"], stats["failed"],
            )

    return stats