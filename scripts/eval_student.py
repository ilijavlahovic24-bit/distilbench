"""CLI: evaluate a trained student checkpoint on a held-out split.

Examples:
  python -m scripts.eval_student \
      --adapter results/checkpoints/gsm8k_cot_alpha0.5/adapter \
      --domain gsm8k \
      --test-file data/datasets/processed/gsm8k_test.jsonl \
      --run-name gsm8k_cot_alpha0.5

  # Smoke: 50 items only
  python -m scripts.eval_student ... --max-items 50

  # All three domains sequentially (if corresponding adapters exist)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.config import DistillBenchConfig
from eval.gsm8k_eval import evaluate_gsm8k
from eval.humaneval_eval import evaluate_humaneval_derivat
from eval.strategyqa_eval import evaluate_strategyqa
from models.student import StudentBundle, build_bnb_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("eval_student")

DEFAULT_RESULTS = Path("results/eval")


def load_trained_student(adapter_dir: Path) -> StudentBundle:
    """Load base model in 4-bit, then attach the trained LoRA adapter."""
    cfg = DistillBenchConfig().student
    adapter_dir = Path(adapter_dir)

    LOG.info("Loading tokenizer from %s", adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir), trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOG.info("Loading base model in 4-bit: %s", cfg.model_name)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=build_bnb_config(cfg),
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    LOG.info("Attaching adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    model.config.use_cache = True   # generation speed

    return StudentBundle(model=model, tokenizer=tokenizer, cfg=cfg)


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--domain",
                   choices=["gsm8k", "strategyqa", "humaneval_derivat"],
                   required=True)
    p.add_argument("--test-file", required=True, type=Path)
    p.add_argument("--run-name", required=True)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--save-dir", type=Path, default=DEFAULT_RESULTS)
    args = p.parse_args()

    if not torch.cuda.is_available():
        LOG.error("CUDA not available.")
        return 1

    bundle = load_trained_student(args.adapter)

    # Sensible defaults per domain
    defaults = {"gsm8k": 384, "strategyqa": 384, "humaneval_derivat": 512}
    max_new = args.max_new_tokens or defaults[args.domain]

    if args.domain == "gsm8k":
        evaluate_gsm8k(bundle, args.test_file, args.run_name,
                       max_new_tokens=max_new, max_items=args.max_items,
                       save_dir=args.save_dir)
    elif args.domain == "strategyqa":
        evaluate_strategyqa(bundle, args.test_file, args.run_name,
                            max_new_tokens=max_new, max_items=args.max_items,
                            save_dir=args.save_dir)
    elif args.domain == "humaneval_derivat":
        evaluate_humaneval_derivat(bundle, args.test_file, args.run_name,
                                   max_new_tokens=max_new, max_items=args.max_items,
                                   save_dir=args.save_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())