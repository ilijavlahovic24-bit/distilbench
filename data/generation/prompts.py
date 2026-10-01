"""Prompt templates per source/domain.

Teachers are given explicit format hints, but the parser accepts multiple
formats (see parsing.py) so we tolerate drift. Rationale is expected to be
a step-by-step chain-of-thought; the final answer should be on its own line.
"""
from __future__ import annotations

GSM8K_SYSTEM = (
    "You are a careful math tutor. You solve grade-school math problems by "
    "reasoning step by step. Show every calculation. Keep the reasoning "
    "concise but complete."
)

GSM8K_TEMPLATE = (
    "Solve the following math problem step by step.\n\n"
    "At the very end, put the final answer on its own line in one of these "
    "formats:\n"
    "    Answer: <number>\n"
    "or\n"
    "    #### <number>\n\n"
    "Problem: {question}\n"
)

STRATEGYQA_SYSTEM = (
    "You are a careful reasoning assistant. You answer yes/no questions by "
    "decomposing them into intermediate facts and reasoning step by step. "
    "Do not guess — justify each intermediate step."
)

STRATEGYQA_TEMPLATE = (
    "Answer the following yes/no question by reasoning step by step.\n\n"
    "At the very end, put the final answer on its own line in one of these "
    "formats:\n"
    "    Answer: yes\n"
    "    Answer: no\n"
    "or\n"
    "    #### yes\n"
    "    #### no\n\n"
    "Question: {question}\n"
)


def build_prompt(item: dict) -> tuple[str, str]:
    """Return (user_prompt, system_prompt) for a raw dataset item."""
    source = item["source"]
    if source == "gsm8k":
        return GSM8K_TEMPLATE.format(question=item["input"]), GSM8K_SYSTEM
    if source == "strategyqa":
        return STRATEGYQA_TEMPLATE.format(question=item["input"]), STRATEGYQA_SYSTEM
    raise ValueError(f"No prompt template for source='{source}'")