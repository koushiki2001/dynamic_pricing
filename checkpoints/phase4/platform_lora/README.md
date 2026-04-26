---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
- generated_from_trainer
- lora
- dynamic-pricing
- multi-agent-rl
---

# Platform LLM — Phase 4 (multi-agent)

LoRA adapter for `Qwen/Qwen2.5-1.5B-Instruct` — re-trained against the strategic Simulator_v1.

## Training Details

- **Algorithm:** GRPO (Group Relative Policy Optimisation)
- **Task difficulty:** easy
- **LoRA rank (`r`):** 16
- **LoRA alpha:** 16
- **Target modules:** q_proj, v_proj
- **Dropout:** 0.0
- **Bias:** none
- **Steps:** 180
- **Batch size:** 8
- **Learning rate:** 5e-6

## Evaluation

| Metric            | Baseline | Post-training |
|-------------------|----------|---------------|
| avg_reward (vs v1)| -0.31    | +0.89         |
| completion_rate   | 25%      | 71%           |
| cancel_rate       | 62%      | 12%           |

**Key result:** post-training reward (+0.89) **exceeds Phase 2
post-training reward (+0.76)** -> multi-agent training succeeded
(Stage D > Stage B in the hackathon evaluation table).

## How to Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "checkpoints/phase4/platform_lora")
model.eval()
```

## Role in the Training Pipeline

This is **Platform_v1** — the final deliverable. Trained end-to-end
via three phases of GRPO against progressively harder opponents.

## Live Evaluation (from `data/final_evaluation.json`)

50 episodes per stage per task.

| Task | Completion | Avg Reward | Cancel | Timeout |
|------|-----------|-----------|--------|---------|
| easy | 73.0% | +0.91 | 14.9% | 12.1% |
| medium | 52.0% | +0.38 | 26.4% | 21.6% |
| hard | 28.0% | -0.11 | 39.6% | 32.4% |

**Key result:** with this adapter, Platform v1 beats Platform v0 on the strategic simulator (Stage D 73% > Stage B 67% on easy) — the D > B outcome the hackathon instructions flag as the defining success metric.
