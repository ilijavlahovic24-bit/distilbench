"""Sanity-check the multi-task loss on a synthetic batch.

Verifies:
  1. Total loss equals alpha*CE_answer + (1-alpha)*CE_rationale
  2. alpha=1.0 matches the standard causal-LM loss on the answer segment
  3. Gradients flow to logits and are finite
  4. Zero-mask edge case doesn't produce NaN

Run: python -m scripts.test_loss
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from engine.losses import answer_only_loss, multi_task_distillation_loss


def _synthetic_batch(B=2, T=12, V=50, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(1, V, (B, T))
    # Mask last 3 tokens of prompt (positions 0..2) and pad (none here)
    rationale_mask = torch.zeros(B, T)
    rationale_mask[:, 3:6] = 1
    answer_mask = torch.zeros(B, T)
    answer_mask[:, 6:9] = 1
    # Positions 9..11 are "prompt tail" — no loss
    labels[:, 9:] = -100
    return logits, labels, rationale_mask, answer_mask


def main() -> int:
    logits, labels, rmask, amask = _synthetic_batch()

    # ---- 1) Manual recomputation vs formula -----------------------------
    bd = multi_task_distillation_loss(logits, labels, rmask, amask, alpha=0.5)
    manual = 0.5 * bd.ce_answer + 0.5 * bd.ce_rationale
    assert torch.allclose(bd.total, manual, atol=1e-6), "weighting mismatch"
    print(f"[ok] formula   total={bd.total.item():.4f} "
          f"ce_r={bd.ce_rationale.item():.4f} ce_a={bd.ce_answer.item():.4f}")

    # ---- 2) alpha=1.0 should match plain CE on the answer segment -------
    bd1 = multi_task_distillation_loss(logits, labels, rmask, amask, alpha=1.0)
    # Recompute plain CE restricted to answer tokens:
    sl = logits[..., :-1, :].reshape(-1, logits.size(-1))
    tl = labels[..., 1:].reshape(-1)
    am = amask[..., 1:].reshape(-1).bool()
    ce_manual = F.cross_entropy(sl[am], tl[am])
    assert torch.allclose(bd1.ce_answer, ce_manual, atol=1e-6)
    assert torch.allclose(bd1.total, ce_manual, atol=1e-6)
    print(f"[ok] alpha=1.0 -> answer-only loss matches manual CE ({ce_manual.item():.4f})")

    # ---- 3) Gradients flow and are finite -------------------------------
    bd.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), "non-finite gradient"
    gnorm = logits.grad.norm().item()
    print(f"[ok] backward   grad_norm={gnorm:.4f}")

    # ---- 4) Zero answer-mask must not produce NaN -----------------------
    zero_mask = torch.zeros_like(amask)
    bd_zero = multi_task_distillation_loss(logits, labels, rmask, zero_mask, alpha=0.5)
    assert torch.isfinite(bd_zero.total), "NaN in zero-mask edge case"
    print(f"[ok] zero-mask  total={bd_zero.total.item():.4f} (no NaN)")

    print("\nAll loss checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())