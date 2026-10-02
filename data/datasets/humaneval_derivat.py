"""HumanEval -> buggy code-debugging dataset via deterministic mutations.

We do NOT ask the teacher to invent bugs (that would be circular). Instead we
apply AST-based mutations to the canonical HumanEval solution, producing a
verifiably buggy variant. The teacher then generates a debug rationale.

Mutation types (see BUG_TYPES in prompts.py):
  - off_by_one:       range(n) -> range(n+1) or range(n-1)
  - wrong_operator:   <  -> <=,  + -> -,  * -> //, etc.
  - wrong_variable:   swap a local variable with another of the same scope
  - missing_return:   drop a return statement (function returns None)
  - wrong_initial_value: 0 -> 1, 1 -> 0, "" -> None, [] -> {}

The task_description is the docstring; the ground truth 'answer' is the
sentinel string 'fixed' (the only correct answer for a debug task), and the
'rationale' is the teacher's debug explanation.
"""
from __future__ import annotations

import ast
import logging
import random
import textwrap
from pathlib import Path

from datasets import load_dataset

from data.datasets.common import write_jsonl

LOG = logging.getLogger(__name__)

_BINOP_SWAPS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
}
_CMPOP_SWAPS = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
}
_NUM_SWAPS = {0: 1, 1: 0, -1: 1}


def _extract_docstring_and_body(full_code: str) -> tuple[str, str]:
    """Return (signature+docstring as task_description, function source)."""
    tree = ast.parse(full_code)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return "", full_code
    doc = ast.get_docstring(fn) or ""
    return doc.strip(), full_code


class _Mutator(ast.NodeTransformer):
    def __init__(self, kind: str, rng: random.Random):
        self.kind = kind
        self.rng = rng
        self.applied = False

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if self.kind == "wrong_operator" and not self.applied:
            swap = _BINOP_SWAPS.get(type(node.op))
            if swap is not None:
                node.op = swap()
                self.applied = True
        return node

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if self.kind == "wrong_operator" and not self.applied:
            if node.ops:
                swap = _CMPOP_SWAPS.get(type(node.ops[0]))
                if swap is not None:
                    node.ops[0] = swap()
                    self.applied = True
        return node

    def visit_Constant(self, node: ast.Constant):
        if self.kind == "wrong_initial_value" and not self.applied:
            if isinstance(node.value, int) and node.value in _NUM_SWAPS:
                node.value = _NUM_SWAPS[node.value]
                self.applied = True
        if self.kind == "off_by_one" and not self.applied:
            if isinstance(node.value, int) and node.value > 0:
                node.value = node.value + 1
                self.applied = True
        return node

    def visit_Return(self, node: ast.Return):
        self.generic_visit(node)
        if self.kind == "missing_return" and not self.applied:
            # Replace return value with None
            node.value = ast.Constant(value=None)
            self.applied = True
        return node


def _apply_mutation(code: str, kind: str, rng: random.Random) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    mutator = _Mutator(kind, rng)
    new_tree = mutator.visit(tree)
    if not mutator.applied:
        return None
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:  # noqa: BLE001
        return None


def load_humaneval_derivat(
    n: int | None = None,
    seed: int = 42,
    mutation_kinds: tuple[str, ...] | None = None,
) -> list[dict]:
    """Build a code-debugging dataset from HumanEval with injected bugs."""
    from data.generation.prompts import BUG_TYPES

    rng = random.Random(seed)
    kinds = mutation_kinds or BUG_TYPES

    LOG.info("Loading HumanEval (openai/openai_humaneval, split=test)")
    ds = load_dataset("openai/openai_humaneval", split="test")
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))

    items: list[dict] = []
    for idx, row in enumerate(ds):
        canonical = row["prompt"] + row["canonical_solution"]
        task_description, _ = _extract_docstring_and_body(canonical)

        mutated_code: str | None = None
        used_kind: str | None = None
        # Try mutation kinds in a shuffled order until one applies.
        for kind in rng.sample(list(kinds), k=len(kinds)):
            mutated_code = _apply_mutation(canonical, kind, rng)
            if mutated_code is not None:
                used_kind = kind
                break

        if mutated_code is None:
            LOG.debug("No mutation applied for task %s; skipping", row["task_id"])
            continue

        items.append({
            "id": f"humaneval_derivat_{idx:04d}",
            "domain": "code",
            "source": "humaneval_derivat",
            "split": "test",
            "input": row["prompt"].strip(),
            "answer": "fixed",
            "answer_type": "sentinel",
            "metadata": {
                "task_id": row["task_id"],
                "task_description": task_description,
                "canonical_code": canonical,
                "buggy_code": mutated_code,
                "bug_type": used_kind,
                "entry_point": row.get("entry_point"),
            },
        })

    LOG.info("HumanEval-derivat: %d items (of %d candidates)", len(items), len(ds))
    return items


def build_humaneval_derivat_jsonl(
    out_dir: Path,
    n: int | None = None,
    seed: int = 42,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_humaneval_derivat(n=n, seed=seed)
    path = out_dir / "humaneval_derivat_test.jsonl"
    write_jsonl(path, items)
    return path