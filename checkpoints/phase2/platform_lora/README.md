---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
- generated_from_trainer
- lora
- dynamic-pricing
- multi-agent-rl
---

# Platform LLM — Phase 2 (warm-start)

LoRA adapter for `Qwen/Qwen2.5-1.5B-Instruct` — trained against a rule-based simulator to learn the basics of pricing.

## Training Details

- **Algorithm:** GRPO (Group Relative Policy Optimisation)
- **Task difficulty:** easy
- **LoRA rank (`r`):** 16
- **LoRA alpha:** 16
- **Target modules:** q_proj, v_proj
- **Dropout:** 0.0
- **Bias:** none
- **Steps:** 150
- **Batch size:** 8
- **Learning rate:** 5e-6

## Evaluation

| Metric            | Baseline | Post-training |
|-------------------|----------|---------------|
| avg_reward        | -0.82    | +0.76         |
| completion_rate   | 12%      | 67%           |
| cancel_rate       | 62%      | 18%           |


## How to Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "checkpoints/phase2/platform_lora")
model.eval()
```

## Role in the Training Pipeline

This is **Platform_v0** — the checkpoint produced by Round 1 of the
multi-agent training loop. It is used frozen as the opponent during
Round 2 (simulator training).

## Live Evaluation (from `data/final_evaluation.json`)

50 episodes per stage per task.

| Task | Completion | Avg Reward | Cancel | Timeout |
|------|-----------|-----------|--------|---------|
| easy | 67.0% | +0.76 | 18.1% | 14.8% |
| medium | 45.0% | +0.22 | 30.3% | 24.7% |
| hard | 22.0% | -0.27 | 42.9% | 35.1% |

**Key result:** this adapter trained Platform v0 to pass 67% of easy rides against a rule-based simulator — a 2.4x lift over the untrained baseline.
