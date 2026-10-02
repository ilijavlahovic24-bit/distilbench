"""CLI: train the QLoRA student with multi-task distillation loss.

Examples (from repo root):

  # Full CoT distillation on GSM8K
  python -m scripts.train_student \
      --train-file data/datasets/processed/gsm8k_train.filtered.jsonl \
      --run-name gsm8k_cot_alpha0.5 \
      --alpha 0.5 --epochs 3

  # Answer-only baseline (same code, alpha=1.0)
  python -m scripts.train_student \
      --train-file data/datasets/processed/gsm8k_train.filtered.jsonl \
      --run-name gsm8k_answer_only \
      --alpha 1.0 --epochs 3

  # Quick smoke run (100 items, 1 epoch)
  python -m scripts.train_student \
      --train-file data/datasets/processed/gsm8k_train.filtered.jsonl \
      --run-name debug_run --max-train-items 100 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from dotenv import load_dotenv

from engine.config import DistillBenchConfig
from engine.dataset import DistillDataset
from engine.trainer import build_trainer
from models.student import load_student

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("train_student")


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True, type=Path)
    p.add_argument("--eval-file", type=Path, default=None)
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-root", type=Path, default=Path("results/checkpoints"))
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-train-items", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        LOG.error("CUDA not available.")
        return 1

    cfg = DistillBenchConfig()
    if args.epochs is not None:
        cfg.training.num_train_epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.max_seq_length is not None:
        cfg.training.max_seq_length = args.max_seq_length
    cfg.training.alpha = args.alpha

    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Run: %s | alpha=%.2f | out=%s",
             args.run_name, args.alpha, output_dir)
    LOG.info("Training config: %s", json.dumps(
        {k: str(v) for k, v in vars(cfg.training).items()}, indent=2))

    bundle = load_student(cfg.student)

    train_ds = DistillDataset(
        path=args.train_file,
        tokenizer=bundle.tokenizer,
        max_seq_length=cfg.training.max_seq_length,
        max_items=args.max_train_items,
    )
    eval_ds = None
    if args.eval_file is not None:
        eval_ds = DistillDataset(
            path=args.eval_file,
            tokenizer=bundle.tokenizer,
            max_seq_length=cfg.training.max_seq_length,
        )

    trainer = build_trainer(
        bundle=bundle,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        cfg=cfg.training,
        output_dir=output_dir,
        run_name=args.run_name,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume)

    # Save LoRA adapter + tokenizer (small, ~20MB)
    trainer.save_model(str(output_dir / "adapter"))
    bundle.tokenizer.save_pretrained(str(output_dir / "adapter"))

    # Save training metadata for the report
    meta = {
        "run_name": args.run_name,
        "alpha": args.alpha,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file) if args.eval_file else None,
        "num_train_items": len(train_ds),
        "epochs": cfg.training.num_train_epochs,
        "lr": cfg.training.learning_rate,
        "batch_size": cfg.training.per_device_train_batch_size,
        "grad_accum": cfg.training.gradient_accumulation_steps,
        "max_seq_length": cfg.training.max_seq_length,
        "student_model": cfg.student.model_name,
        "lora_r": cfg.student.lora_r,
        "lora_alpha": cfg.student.lora_alpha,
        "train_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in train_result.metrics.items()},
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    LOG.info("Done. Adapter: %s | Meta: %s",
             output_dir / "adapter", output_dir / "run_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())