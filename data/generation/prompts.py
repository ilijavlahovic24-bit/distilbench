"""Prompt templates per source/domain."""
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

CODE_DEBUG_SYSTEM = (
    "You are a senior software engineer debugging Python code. Identify the "
    "bug, explain why it fails, and give the corrected implementation. Be "
    "precise: name the function, the line, and the fix."
)

CODE_DEBUG_TEMPLATE = (
    "The following Python function is supposed to solve the task below, but "
    "it contains a bug. The function docstring describes the intended "
    "behaviour.\n\n"
    "Task description (from docstring):\n"
    "{task_description}\n\n"
    "Buggy implementation:\n"
    "```python\n"
    "{buggy_code}\n"
    "```\n\n"
    "Debug the function step by step:\n"
    "1. Identify what the function actually does vs. what it should do.\n"
    "2. Point out the specific bug (line and reason).\n"
    "3. Provide the corrected function.\n\n"
    "At the very end, on its own line, state whether the bug is fixed:\n"
    "    Answer: fixed\n"
)

BUG_TYPES = (
    "off_by_one",
    "wrong_operator",
    "wrong_variable",
    "missing_return",
    "wrong_initial_value",
)


def build_prompt(item: dict) -> tuple[str, str]:
    """Return (user_prompt, system_prompt) for a raw dataset item."""
    source = item["source"]
    if source == "gsm8k":
        return GSM8K_TEMPLATE.format(question=item["input"]), GSM8K_SYSTEM
    if source == "strategyqa":
        return STRATEGYQA_TEMPLATE.format(question=item["input"]), STRATEGYQA_SYSTEM
    if source == "humaneval_derivat":
        return (
            CODE_DEBUG_TEMPLATE.format(
                task_description=item["metadata"]["task_description"],
                buggy_code=item["metadata"]["buggy_code"],
            ),
            CODE_DEBUG_SYSTEM,
        )
    raise ValueError(f"No prompt template for source='{source}'")