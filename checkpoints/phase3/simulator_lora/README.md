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
model = PeftModel.from_pretrained(base, "checkpoints/phase3/simulator_lora")
model.eval()
```

## Role in the Training Pipeline

This is **Simulator_v1** — the strategic-bluffing opponent produced by
Round 2. It is frozen during Round 3 and the platform must learn to
price well against its strategic behaviour.

## Live Evaluation (from `data/final_evaluation.json`)

50 episodes per stage per task.

| Task | Completion | Avg Reward | Cancel | Timeout |
|------|-----------|-----------|--------|---------|
| easy | 44.0% | -0.31 | 38.1% | 17.9% |
| medium | 28.0% | -0.65 | 49.0% | 23.0% |
| hard | 13.0% | -1.02 | 59.2% | 27.8% |

**Key result:** when Platform v0 faces this adapter, its completion rate drops from 67% to 44% on easy (Stage C) — proof the simulator learned genuine strategic bluffing rather than being a passive rule-based opponent.
