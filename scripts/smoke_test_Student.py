"""Smoke test for the QLoRA student setup on GTX 1650.

Verifies:
  1. Student loads in 4-bit without CUDA OOM
  2. LoRA adapters are attached and trainable
  3. One forward + backward pass fits in 4GB VRAM
  4. Peak VRAM is reported so we know how much headroom remains

Run from repo root:
    python -m scripts.smoke_test_student
"""
from __future__ import annotations

import logging
import sys

import torch
from dotenv import load_dotenv

from engine.config import DistillBenchConfig
from models.student import generate, load_student

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("smoke_test_student")


PROMPT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether "
    "in April and May? Show your reasoning step by step, then give the final "
    "answer on a line starting with 'Answer:'.\n"
)


def main() -> int:
    load_dotenv()

    if not torch.cuda.is_available():
        LOG.error("CUDA not available. This project requires a CUDA GPU.")
        return 1

    LOG.info("GPU: %s | VRAM total: %.2f GB",
             torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1024**3)
    torch.cuda.reset_peak_memory_stats()

    cfg = DistillBenchConfig().student
    bundle = load_student(cfg)

    # --- 1) Generation sanity check ----------------------------------------
    LOG.info("--- Greedy generation sanity check ---")
    try:
        out = generate(bundle, PROMPT, max_new_tokens=128, temperature=0.0)
        LOG.info("Model output:\n%s", out[:500])
    except torch.cuda.OutOfMemoryError:
        LOG.error("OOM during generation with max_new_tokens=128.")
        return 1
    finally:
        torch.cuda.empty_cache()

    # --- 2) Simulated training step (fwd + bwd) ----------------------------
    LOG.info("--- Simulated training step (forward + backward) ---")
    tokenizer = bundle.tokenizer
    model = bundle.model
    model.train()

    enc = tokenizer(
        PROMPT + "Answer: 72",
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_seq_length if hasattr(cfg, "max_seq_length") else 512,
    ).to(model.device)

    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(**enc, labels=enc["input_ids"])
            loss = out.loss
        loss.backward()
    except torch.cuda.OutOfMemoryError:
        LOG.error(
            "OOM on a single training step. Try reducing max_seq_length, "
            "or drop FFN modules (gate/up/down_proj) from lora_target_modules."
        )
        return 1

    LOG.info("Loss: %.4f", loss.item())
    peak = torch.cuda.max_memory_allocated() / 1024**3
    LOG.info("Peak VRAM during training step: %.2f GB", peak)

    if peak > 3.5:
        LOG.warning(
            "Peak VRAM %.2f GB is close to the 4GB limit. Expect OOM with "
            "longer sequences or larger batch. Keep batch=1 and grad_accum>1.",
            peak,
        )
    else:
        LOG.info("Headroom OK — %.2f GB left for activations/optimizer.", 4.0 - peak)

    LOG.info("OK: student QLoRA setup is functional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())