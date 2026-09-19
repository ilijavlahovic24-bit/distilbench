# DistillBench

A custom Knowledge Distillation engine focused on **Chain-of-Thought (reasoning) distillation** — distilling a large language model's reasoning process into a smaller, faster model, not just its final answer.

## Motivation

Standard distillation (soft targets, feature matching) teaches a student to imitate a teacher's *output*. CoT distillation goes further: it teaches the student to reproduce the *process* by which the teacher arrives at an answer (step-by-step reasoning), which yields much better results on reasoning-heavy tasks with a far smaller model and less training data.

This direction is directly relevant as it gives:
- smaller, faster models that still reason well directly cut latency/cost in a search + LLM pipeline
- local/on-device AI Assistant tooling needs models that run fast without a GPU farm while retaining reasoning quality (e.g. for code debugging)

## Architecture

```
distillbench/
├── data/          # teacher rationale generation and caching, dataset loaders
├── engine/        # multi-task loss, QLoRA trainer, config
├── models/        # teacher API wrapper, student (HF + PEFT)
├── eval/          # GSM8K/HumanEval eval harness, metrics
├── experiments/   # scripts for each experiment
├── results/       # raw results + plots
└── report/        # write-up
```

## Key decisions

- **Teacher**: large open-weight model accessed via API (Together/Groq/OpenRouter), with response caching (SQLite) to avoid paying twice
- **Student**: Qwen2.5-0.5B (primary), Qwen2.5-1.5B (stretch ablation) - QLoRA/4-bit training due to the GTX 1650 4GB constraint
- **Loss**: multi-task - combination of answer CE and rationale CE, with a tunable α
- **Domains**: multi-step QA (GSM8K, StrategyQA) and code-debugging reasoning (HumanEval-derived), covered simultaneously
- **Data filtering**: rationale examples where the teacher's final answer is incorrect are discarded

## Experiments

1. CoT-distilled vs. answer-only-distilled vs. no distillation (direct ground-truth fine-tuning)
2. Data-efficiency curve (accuracy as a function of number of training examples)
3. Student size ablation (0.5B vs 1.5B)
4. Cross-domain transfer check (QA → code and vice versa)


## Resources

Hsieh et al. 2023 (Distilling Step-by-Step), Magister et al. 2023, Wei et al. 2022 (CoT Prompting), Chen et al. 2023 (Self-Debug) 
