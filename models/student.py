"""Student model setup: 4-bit QLoRA for GTX 1650 (4GB VRAM).

Design notes
------------
- BitsAndBytesConfig: NF4 + double quant + fp16 compute. bf16 is NOT used because GTX 1650 (Turing, SM 7.5)
 does not support it.
- LoRA covers attention *and* FFN projections, since we want the student to
  adapt its reasoning path, not just its attention pattern.
- Gradient checkpointing is enabled here already; `trainer.py` only needs to
  call `model.enable_input_require_grads()` (handled by
  `prepare_model_for_kbit_training`).
- Attention implementation is "sdpa" — FlashAttention-2 requires Ampere+
  (SM 8.0) and would silently fall back on Turing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from engine.config import StudentConfig

LOG = logging.getLogger(__name__)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Expected one of {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


@dataclass
class StudentBundle:
    """Everything downstream code (trainer, eval) needs from a student."""

    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    cfg: StudentConfig

    @property
    def is_peft(self) -> bool:
        return isinstance(self.model, PeftModel)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #

def build_bnb_config(cfg: StudentConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_resolve_dtype(cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #

def build_lora_config(cfg: StudentConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_student(cfg: StudentConfig, device_index: int = 0) -> StudentBundle:
    """Load Qwen2.5 (or any HF causal LM) in 4-bit and attach LoRA adapters.

    Parameters
    ----------
    cfg : StudentConfig
        Populated from `DistillBenchConfig.student`.
    device_index : int
        CUDA device index. Default 0 (single-GPU setup on GTX 1650).
    """
    LOG.info("Loading tokenizer: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="right",     # required for causal LM training
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    LOG.info("Loading base model in 4-bit (this may take a minute on first run)...")
    bnb_config = build_bnb_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        torch_dtype=_resolve_dtype(cfg.torch_dtype),
        device_map={"": device_index},   # explicit: avoids accelerate placing layers on CPU
        attn_implementation="sdpa",      # FA2 unsupported on Turing (SM 7.5)
        trust_remote_code=True,
    )

    # Freeze base, unfreeze LoRA later; also enables input grads for grad checkpointing
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    LOG.info("Attaching LoRA (r=%d, alpha=%d) on: %s",
             cfg.lora_r, cfg.lora_alpha, ", ".join(cfg.lora_target_modules))
    model = get_peft_model(model, build_lora_config(cfg))
    model.config.use_cache = False  # incompatible with gradient checkpointing

    bundle = StudentBundle(model=model, tokenizer=tokenizer, cfg=cfg)
    _log_param_counts(bundle)
    _log_vram("after load")

    return bundle


def _log_param_counts(bundle: StudentBundle) -> None:
    total = bundle.total_parameters()
    trainable = bundle.trainable_parameters()
    pct = 100.0 * trainable / total if total else 0.0
    LOG.info(
        "Parameters: total=%.2fM | trainable=%.2fM (%.2f%%)",
        total / 1e6, trainable / 1e6, pct,
    )


def _log_vram(stage: str) -> None:
    if not torch.cuda.is_available():
        LOG.warning("CUDA not available — skipping VRAM report (%s)", stage)
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    LOG.info("VRAM %s: allocated=%.2f GB | reserved=%.2f GB", stage, alloc, reserved)


# --------------------------------------------------------------------------- #
# Inference helper (used by smoke test and eval/)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate(
    bundle: StudentBundle,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool | None = None,
) -> str:
    """Quick generation helper for sanity checks and eval loops.

    Temperature 0.0 -> greedy decoding (deterministic). Set temperature > 0
    and do_sample=True for stochastic generation.
    """
    if do_sample is None:
        do_sample = temperature > 0.0

    model = bundle.model
    model.eval()
    tokenizer = bundle.tokenizer

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=0.95 if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)