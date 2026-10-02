"""Multi-task distillation loss.

L = alpha * CE(answer) + (1 - alpha) * CE(rationale)

Where CE(...) is per-token cross-entropy averaged over the positions
belonging to that segment only. Rationale and answer segments are marked
by `rationale_mask` and `answer_mask` tensors of the same shape as `labels`.

The default ignore_index=-100 handles padding; masked-out positions
contribute 0 to both the numerator and are excluded from the denominators.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LossBreakdown:
    total: torch.Tensor
    ce_rationale: torch.Tensor
    ce_answer: torch.Tensor


def multi_task_distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    rationale_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    alpha: float = 0.5,
    ignore_index: int = -100,
) -> LossBreakdown:
    """Compute the weighted multi-task loss.

    Shapes
    ------
    logits          : [B, T, V]
    labels          : [B, T]
    rationale_mask  : [B, T]  (1 where label is a rationale token)
    answer_mask     : [B, T]  (1 where label is an answer token)
    """
    # Causal shift: logits at position t predict labels at position t+1.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_rationale = rationale_mask[..., 1:].contiguous().float()
    shift_answer = answer_mask[..., 1:].contiguous().float()

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())

    rationale_denom = shift_rationale.sum().clamp(min=1.0)
    answer_denom = shift_answer.sum().clamp(min=1.0)

    ce_rationale = (per_token * shift_rationale).sum() / rationale_denom
    ce_answer = (per_token * shift_answer).sum() / answer_denom

    total = alpha * ce_answer + (1.0 - alpha) * ce_rationale

    return LossBreakdown(total=total, ce_rationale=ce_rationale, ce_answer=ce_answer)


def answer_only_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard causal LM loss — used as a degenerate check in tests."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )