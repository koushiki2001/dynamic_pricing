---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
tags:
- generated_from_trainer
- lora
- dynamic-pricing
- multi-agent-rl
---

# Simulator LLM — Phase 3 (strategic bluffing)

LoRA adapter for `Qwen/Qwen2.5-0.5B-Instruct` — learns strategic bluffing against a frozen Platform_v0.

## Training Details

- **Algorithm:** GRPO (Group Relative Policy Optimisation)
- **Task difficulty:** easy
- **LoRA rank (`r`):** 8
- **LoRA alpha:** 8
- **Target modules:** q_proj, v_proj
- **Dropout:** 0.0
- **Bias:** none
- **Steps:** 120
- **Batch size:** 8
- **Learning rate:** 5e-6

## Evaluation

| Metric            | Baseline | Post-training |
|-------------------|----------|---------------|
| avg_sim_reward    | -1.24    | +0.73         |
| bluff_rate        | 10%      | 33%           |
| collapse_rate     | 62%      | 30%           |


## How to Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "checkpoints/last_stable/simulator_lora")
model.eval()
```

## Role in the Training Pipeline

This is **Simulator_v1** — the strategic-bluffing opponent produced by
Round 2. It is frozen during Round 3 and the platform must learn to
price well against its strategic behaviour.
