"""DistillTrainer: transformers.Trainer with multi-task compute_loss.

Logs `ce_rationale` and `ce_answer` alongside the total loss so that
per-component contribution is visible during training (per plan, Faza 2).
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from transformers import Trainer, TrainingArguments

from engine.config import TrainingConfig
from engine.dataset import DistillCollator, DistillDataset
from engine.losses import multi_task_distillation_loss
from models.student import StudentBundle

LOG = logging.getLogger(__name__)


class DistillTrainer(Trainer):
    def __init__(self, *args, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self._last_ce_rationale: float | None = None
        self._last_ce_answer: float | None = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rationale_mask = inputs.pop("rationale_mask")
        answer_mask = inputs.pop("answer_mask")

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        breakdown = multi_task_distillation_loss(
            logits=logits,
            labels=labels,
            rationale_mask=rationale_mask,
            answer_mask=answer_mask,
            alpha=self.alpha,
        )

        # Cache for logging (approximate: last micro-batch).
        self._last_ce_rationale = breakdown.ce_rationale.detach().item()
        self._last_ce_answer = breakdown.ce_answer.detach().item()

        return (breakdown.total, outputs) if return_outputs else breakdown.total

    def log(self, logs, *args, **kwargs):
        if self._last_ce_rationale is not None:
            logs["ce_rationale"] = self._last_ce_rationale
            logs["ce_answer"] = self._last_ce_answer
        return super().log(logs, *args, **kwargs)


def build_training_arguments(
    cfg: TrainingConfig,
    output_dir: Path,
    run_name: str,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=[],
        remove_unused_columns=False,   # critical: keeps our custom mask columns
        dataloader_num_workers=0,      # Windows + small dataset: no benefit
        dataloader_pin_memory=False,   # saves VRAM on 4GB cards
    )


def build_trainer(
    bundle: StudentBundle,
    train_dataset: DistillDataset,
    cfg: TrainingConfig,
    output_dir: Path,
    run_name: str,
    eval_dataset: DistillDataset | None = None,
) -> DistillTrainer:
    args = build_training_arguments(cfg, output_dir, run_name)
    collator = DistillCollator(tokenizer=bundle.tokenizer)

    return DistillTrainer(
        model=bundle.model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        alpha=cfg.alpha,
    )