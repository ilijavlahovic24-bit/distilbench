"""Centralised hyperparameters for DistillBench.

All experiment knobs live here so that experiments/*.py scripts only need to
override a few fields.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TeacherConfig:
    cache_path=Path("data/cache/teacher.sqlite3")
    temperature:float=0.2
    max_tokens=1024
    max_retries_per_client: int = 2

@dataclass
class StudentConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "float16" # my GPU does not support bf16
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

@dataclass
class TrainingConfig:
    output_dir: Path = Path("results/checkpoints")
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    max_seq_length: int = 1024

    # Multi-task loss: L = alpha * CE(answer) + (1 - alpha) * CE(rationale)
    # alpha = 1.0 -> answer-only distillation (baseline control group)
    alpha: float = 0.5


@dataclass
class DistillBenchConfig:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42