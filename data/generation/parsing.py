"""Answer extraction from teacher rationales.

The teacher is free to format the final answer in several common ways:
    "Answer: 72"
    "#### 72"
    "The answer is 72."
    "...so the answer is 72"
We accept all of them and normalise.
"""
from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"####\s*(.+?)\s*$", re.MULTILINE),
    re.compile(r"(?:final\s+)?answer\s*[:：]\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"answer\s+is\s+(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
]


def extract_answer(text: str, answer_type: str = "numeric") -> str | None:
    if not text:
        return None
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return _normalize(m.group(1), answer_type)
    return None


def _normalize(raw: str, answer_type: str) -> str:
    s = raw.strip().rstrip(". ").strip()
    if answer_type == "numeric":
        s = s.replace(",", "").replace("$", "").strip()
        # If teacher wrote "72 clips" or "72." -> keep leading number
        m = re.match(r"(-?\d+(?:\.\d+)?)", s)
        if m:
            return m.group(1)
        return s
    if answer_type == "boolean":
        low = s.lower()
        if low.startswith("yes"):
            return "yes"
        if low.startswith("no"):
            return "no"
        return low
    return s


def answers_match(pred: str | None, gt: str, answer_type: str) -> bool:
    if pred is None:
        return False
    if answer_type == "numeric":
        try:
            return abs(float(pred) - float(gt)) < 1e-6
        except (TypeError, ValueError):
            return pred.strip() == gt.strip()
    if answer_type == "boolean":
        return pred.lower() == gt.lower()
    return pred.strip() == gt.strip()